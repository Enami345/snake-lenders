"""
Snakes & Lenders — Board Generator
Generates a randomized valid 10x10 board using BFS path validation.
Ensures the board is solvable with an average path of at least 10 turns.
"""

import random
from game.models import Snake, Ladder, BoardState, Player


NUM_TILES       = 100
NUM_LADDERS     = 7
NUM_INIT_SNAKES = 4
NUM_BOMBS       = 5
MIN_AVG_TURNS   = 10
MAX_RETRIES     = 200
BOMB_DEDUCTION  = 30
BASE_TILE_VALUE = 3     # scarce income — points must be managed, not hoarded


def generate_tile_values(seed: int) -> dict:
    rng = random.Random(seed)
    tiles = {}
    for tile in range(1, NUM_TILES + 1):
        # Modest, rising income (~4-14/turn): points accumulate enough to
        # fund snakes, but a snake purchase is still a real sacrifice and a
        # bomb can threaten bankruptcy when you're low.
        base = (BASE_TILE_VALUE + 1) + (tile // 15)
        variance = rng.randint(0, 4)
        tiles[tile] = max(1, base + variance)
    tiles[100] = 40    # modest finish bonus (winning doesn't need points)
    return tiles


def _place_ladders(rng: random.Random, forbidden: set) -> list:
    ladders = []
    attempts = 0
    while len(ladders) < NUM_LADDERS and attempts < 1000:
        attempts += 1
        bottom = rng.randint(2, 80)
        jump   = rng.randint(10, 30)
        top    = bottom + jump

        if top >= 100:
            continue
        if bottom in forbidden or top in forbidden:
            continue

        ladders.append(Ladder(bottom=bottom, top=top))
        forbidden.add(bottom)
        forbidden.add(top)

    return ladders


def _place_snakes(rng: random.Random, forbidden: set,
                  count: int, owner_id: int = -1) -> list:
    snakes = []
    attempts = 0
    while len(snakes) < count and attempts < 1000:
        attempts += 1
        head = rng.randint(20, 99)
        drop = rng.randint(5, 25)
        tail = head - drop

        if tail < 1:
            continue
        if head in forbidden or tail in forbidden:
            continue
        if head == 100:
            continue

        snakes.append(Snake(head=head, tail=tail, owner_id=owner_id))
        forbidden.add(head)
        forbidden.add(tail)

    return snakes


def _place_bombs(rng: random.Random, forbidden: set) -> list:
    bombs = []
    attempts = 0
    while len(bombs) < NUM_BOMBS and attempts < 1000:
        attempts += 1
        tile = rng.randint(2, 99)
        if tile not in forbidden:
            bombs.append(tile)
            forbidden.add(tile)
    return bombs


def bfs_expected_turns(ladders: list, snakes: list) -> float:
    dest = {i: i for i in range(1, 101)}
    for l in ladders:
        dest[l.bottom] = l.top
    for s in snakes:
        dest[s.head] = s.tail

    E = [0.0] * 101
    for _ in range(500):
        new_E = E[:]
        for tile in range(99, 0, -1):
            total = sum(E[dest[min(tile + d, 100)]] for d in range(1, 7))
            new_E[tile] = 1 + total / 6.0
        if max(abs(new_E[i] - E[i]) for i in range(1, 100)) < 0.001:
            E = new_E
            break
        E = new_E
    return E[1]


def _board_is_valid(ladders: list, snakes: list) -> bool:
    """
    Validate the board:
    1. No ladder top is another ladder's bottom (chain ladders)
    2. No snake tail is another snake's head (chain snakes)
    3. No ladder top is a snake head
    4. Expected turns >= MIN_AVG_TURNS
    """
    ladder_bottoms = {l.bottom for l in ladders}
    ladder_tops    = {l.top    for l in ladders}
    snake_heads    = {s.head   for s in snakes}
    snake_tails    = {s.tail   for s in snakes}

    # Fix 1: No ladder top should be another ladder's bottom
    if ladder_tops & ladder_bottoms:
        return False

    # Fix 2: No snake tail should be another snake's head
    if snake_tails & snake_heads:
        return False

    # Fix 3: No ladder top should be a snake head
    if ladder_tops & snake_heads:
        return False

    # Fix 4: Expected turns check
    expected = bfs_expected_turns(ladders, snakes)
    return expected >= MIN_AVG_TURNS


def generate_board(seed=None, players=None) -> BoardState:
    if seed is None:
        seed = random.randint(0, 999999)
    if players is None:
        players = [
            Player(player_id=0, name="Player 1"),
            Player(player_id=1, name="Player 2"),
        ]

    print(f"[Board] Generating with seed={seed}...")

    for attempt in range(1, MAX_RETRIES + 1):
        rng = random.Random(seed + attempt)
        forbidden = {1, 100}

        ladders = _place_ladders(rng, forbidden)
        snakes  = _place_snakes(rng, forbidden, NUM_INIT_SNAKES)
        bombs   = _place_bombs(rng, forbidden)

        if len(ladders) < NUM_LADDERS or len(snakes) < NUM_INIT_SNAKES:
            continue

        if _board_is_valid(ladders, snakes):
            tiles    = generate_tile_values(seed + attempt)
            expected = bfs_expected_turns(ladders, snakes)
            print(f"[Board] Valid! Attempt {attempt}, "
                  f"expected turns: {expected:.1f}")
            return BoardState(
                tiles=tiles,
                ladders=ladders,
                snakes=snakes,
                bombs=bombs,
                players=players,
            )

    raise RuntimeError(
        f"Could not generate a valid board after {MAX_RETRIES} attempts."
    )


def print_board(board: BoardState):
    snake_heads    = {s.head: s.tail   for s in board.snakes}
    snake_tails    = {s.tail: s.head   for s in board.snakes}
    ladder_bottoms = {l.bottom: l.top  for l in board.ladders}
    ladder_tops    = {l.top: l.bottom  for l in board.ladders}
    player_tiles   = {p.position: p.name[0] for p in board.players
                      if p.position > 0}

    print("\n" + "=" * 55)
    for row in range(9, -1, -1):
        cols = (range(row*10+1,  row*10+11) if row % 2 == 0
                else range(row*10+10, row*10, -1))
        line = ""
        for tile in cols:
            if tile in player_tiles:       cell = f"[{player_tiles[tile]}]"
            elif tile in snake_heads:      cell = f"S{tile:2d}"
            elif tile in snake_tails:      cell = f"s{tile:2d}"
            elif tile in ladder_bottoms:   cell = f"L{tile:2d}"
            elif tile in ladder_tops:      cell = f"l{tile:2d}"
            elif tile in board.bombs:      cell = f"B{tile:2d}"
            else:                          cell = f"{tile:3d}"
            line += f"{cell:>5}"
        print(line)
    print("=" * 55)
    print("L=ladder bottom  l=ladder top  "
          "S=snake head  s=snake tail  B=bomb\n")