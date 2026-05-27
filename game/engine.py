"""
Snakes & Lenders — Game Engine
Handles the full turn loop: rolling, moving, points, shop, and win condition.
"""

import random
from game.models import BoardState, Player, Snake
from game.board import BOMB_DEDUCTION

# Bombs: a flat hit that grows with board depth (deeper = nastier).
# Big enough to bankrupt a low-on-points player (→ reset to tile 0), but
# flat (not %-of-wealth) so points can still accumulate for the economy.
BOMB_BASE  = 18
BOMB_DEPTH = 6      # +1 point of damage per this many tiles of depth


# ── Economy tuning (manuscript: exponential pricing) ────────────────
# Cost = BASE_SNAKE_PRICE x purchase_count x length^PRICE_ALPHA
# Short snakes cheap; long snakes scale up fast. Price rises each
# purchase. PRICE_ALPHA is the FIX-2 tuning knob (raise if AI always
# buys all 3 — see acceptance sim).
BASE_SNAKE_PRICE = 2
PRICE_ALPHA      = 0.9   # sub-linear: long, devastating snakes stay affordable
MIN_SNAKE_COST   = 12    # so you can SAVE UP for a big knockback (strategy)
MAX_SNAKE_HEAD   = 90    # player-placed snakes (board snakes may go higher)

# Snakes bite ONLY on landing exactly on the head tile. STRIKE_ZONE adds
# this many tiles below the head to the bite range (0 = exact-head only;
# the game's chosen rule). Larger values make snakes fire more often (and
# the AI much stronger) at the cost of the clean "step on the head" feel.
STRIKE_ZONE = 0

# Anti-softlock / anti-wall: don't let snake heads form a long run of
# adjacent tiles (overlapping strike ranges) that gets too sticky to pass.
MAX_HEAD_RUN = 2


def roll_dice() -> int:
    return random.randint(1, 6)


def calculate_snake_cost(player: Player, head: int, tail: int) -> int:
    """
    Manuscript exponential pricing:
        Cost = BASE_SNAKE_PRICE x purchase_count x length^PRICE_ALPHA
    purchase_count = which snake this is (1st, 2nd, 3rd) so each
    additional snake costs more.
    """
    length         = head - tail
    purchase_count = player.snake_count + 1
    cost           = int(BASE_SNAKE_PRICE * purchase_count
                         * (length ** PRICE_ALPHA))
    return max(cost, MIN_SNAKE_COST)


def _head_run_length(existing_heads: set, new_head: int) -> int:
    """
    Length of the run of consecutive snake-head tiles that `new_head`
    would belong to, given the existing head tiles. Used to stop players
    from building an impassable wall of adjacent heads.
    """
    heads = existing_heads | {new_head}
    left = new_head - 1
    while left in heads:
        left -= 1
    right = new_head + 1
    while right in heads:
        right += 1
    return right - left - 1


def can_place_snake(board: BoardState, buyer: Player,
                    head: int, tail: int) -> tuple[bool, str]:
    """
    Validate a snake placement.
    Returns (True, "") if valid, or (False, reason) if not.
    """
    if head <= tail:
        return False, "Head must be higher than tail."
    if not (1 <= tail <= 100 and 1 <= head <= 100):
        return False, "Tiles must be between 1 and 100."
    if head > MAX_SNAKE_HEAD:
        return False, f"Snake head cannot be placed above tile {MAX_SNAKE_HEAD}."
    if head in board.occupied_tiles():
        return False, f"Tile {head} is occupied by a player."
    if tail in board.occupied_tiles():
        return False, f"Tile {tail} is occupied by a player."
    if board.get_snake_at(head):
        return False, f"A snake already exists at tile {head}."
    if board.get_ladder_at(head) or board.get_ladder_at(tail):
        return False, "Cannot place snake on a ladder tile."
    if not buyer.can_buy_snake:
        return False, "You already own the maximum of 3 snakes."

    # Fix: Cannot place snake head at another snake's tail
    snake_tails = {s.tail for s in board.snakes}
    if head in snake_tails:
        return False, (f"Tile {head} is the tail of another snake. "
                       f"Cannot chain snakes.")

    # Fix: Cannot place snake tail at another snake's head
    snake_heads = {s.head for s in board.snakes}
    if tail in snake_heads:
        return False, (f"Tile {tail} is the head of another snake. "
                       f"Cannot chain snakes.")

    # Anti-softlock: don't let this head extend a run of adjacent snake
    # heads beyond MAX_HEAD_RUN (would risk an impassable wall).
    if _head_run_length(snake_heads, head) > MAX_HEAD_RUN:
        return False, (f"Tile {head} would create too long a wall of "
                       f"snake heads. Leave a gap so players can pass.")

    cost = calculate_snake_cost(buyer, head, tail)
    if buyer.points < cost:
        return False, f"Not enough points. Need {cost}, have {buyer.points}."

    return True, ""


# ── Movement ──────────────────────────────────────────────────────────────────

def _consume_snake(board: BoardState, snake: Snake) -> None:
    """
    Player-placed snakes are single-use: once they fire they're removed,
    freeing a slot so the owner can re-earn points and place another.
    Board snakes (owner_id == -1) are fixed terrain and persist.
    """
    if snake.owner_id < 0:
        return
    if snake in board.snakes:
        board.snakes.remove(snake)
    for p in board.players:
        if p.player_id == snake.owner_id and snake in p.snakes_owned:
            p.snakes_owned.remove(snake)
            break


def _striking_snake(board: BoardState, tile: int, mover_id: int = -99):
    """
    Return a snake whose strike range covers `tile` for the moving player.

    Player-placed snakes (owner_id >= 0) are active traps with a strike
    range: they bite on the head OR the STRIKE_ZONE tiles just below it,
    so a well-placed trap reliably catches a passing opponent (~50%).
    Board snakes (owner_id == -1) are fixed terrain and bite on the exact
    head only — keeping the base board from dragging games out.

    Owner immunity: your own snakes never bite you (otherwise you'd walk
    into traps you set ahead of an opponent you're chasing).

    Landing above a head (a clean jump over) is always safe. Picks the
    snake with the highest head (biggest setback) if several overlap.
    """
    best = None
    for s in board.snakes:
        if s.owner_id == mover_id:
            continue  # own trap — immune
        zone = STRIKE_ZONE if s.owner_id >= 0 else 0
        if s.head - zone <= tile <= s.head:
            if best is None or s.head > best.head:
                best = s
    return best


def _apply_snakes(board: BoardState, player: Player,
                  logs: list) -> None:
    """
    Slide the player down if they landed in a snake's strike range (head
    or the STRIKE_ZONE tiles just below it). Jumping clean over a head is
    safe. Player-placed snakes are consumed after biting (single-use).
    Each slide moves strictly downward, so the loop always terminates.
    """
    while True:
        snake = _striking_snake(board, player.position, player.player_id)
        if snake:
            logs.append(f"  🐍 Snake! {player.name} landed on snake "
                        f"{snake.head} and slides to {snake.tail}")
            player.position = snake.tail
            _consume_snake(board, snake)
        else:
            break


def _apply_ladders(board: BoardState, player: Player,
                   logs: list) -> None:
    """
    Keep climbing chained ladders until the player lands
    on a tile with no ladder bottom.
    """
    while True:
        ladder = board.get_ladder_at(player.position)
        if ladder:
            logs.append(f"  🪜 Ladder! {player.name} climbs "
                        f"from {player.position} to {ladder.top}")
            player.position = ladder.top
        else:
            break


def move_player(board: BoardState, player: Player,
                roll: int) -> list[str]:
    """
    Move a player by roll amount, apply snakes/ladders/bombs.

    Overshooting tile 100:
        If position + roll > 100, the move is invalid — the player
        stays in place. An exact roll is required to land on tile 100.
        e.g. tile 97 + roll 5 = would be 102 → invalid, stays at 97.

    Players start at tile 0 — first roll enters the board.
    """
    logs    = []
    old_pos = player.position

    # ── Overshoot: invalid move, stay put (exact roll needed) ─────
    raw = player.position + roll
    if raw > 100:
        logs.append(f"{player.name} rolled a {roll} "
                    f"— overshoots tile 100! Move invalid, stays at "
                    f"{old_pos}. (Exact roll needed to win.)")
        return logs

    new_pos = raw
    player.position = new_pos

    if old_pos == 0:
        logs.append(f"{player.name} rolled a {roll} "
                    f"— enters the board at tile {new_pos}")
    else:
        logs.append(f"{player.name} rolled a {roll} "
                    f"— moves from {old_pos} to {new_pos}")

    # Exact-landing triggers (applied on entry too): climb a ladder
    # bottom, slide on a snake head. Jumping over a head is safe.
    _apply_ladders(board, player, logs)
    _apply_snakes(board, player, logs)

    # Collect tile points
    tile_pts = board.tiles.get(player.position, 10)
    player.add_points(tile_pts)
    logs.append(f"  💰 Earned {tile_pts} points "
                f"— total: {player.points}")

    # Check bomb — flat hit scaling with depth; can trigger bankruptcy
    if board.is_bomb(player.position):
        loss = BOMB_BASE + player.position // BOMB_DEPTH
        before = player.points
        solvent = player.deduct_points(loss)
        logs.append(f"  💣 Bomb! {player.name} loses {loss} points "
                    f"(had {before}) — total: {player.points}")
        if not solvent:
            logs.append(f"  ☠️ {player.name} went BANKRUPT — reset to tile 0!")

    return logs


# ── Shop Phase ────────────────────────────────────────────────────────────────

def buy_snake(board: BoardState, buyer: Player,
              head: int, tail: int) -> tuple[bool, str]:
    valid, reason = can_place_snake(board, buyer, head, tail)
    if not valid:
        return False, reason

    cost = calculate_snake_cost(buyer, head, tail)
    buyer.deduct_points(cost)

    new_snake = Snake(head=head, tail=tail, owner_id=buyer.player_id)
    board.snakes.append(new_snake)
    buyer.snakes_owned.append(new_snake)

    length = head - tail
    return True, (f"{buyer.name} placed a snake {head}→{tail} "
                  f"(length {length}) for {cost} points.")


# ── Win Check ─────────────────────────────────────────────────────────────────

def check_winner(board: BoardState):
    for player in board.players:
        if player.position >= 100:
            return player
    return None


# ── Full Turn ─────────────────────────────────────────────────────────────────

def do_turn(board: BoardState, shop_decision=None) -> dict:
    player = board.active_player
    logs   = []
    bought = False

    # Shop phase — only if player is on the board
    if shop_decision and player.can_buy_snake and player.position > 0:
        head = shop_decision.get("head")
        tail = shop_decision.get("tail")
        if head and tail:
            success, msg = buy_snake(board, player, head, tail)
            logs.append(f"  🛒 {msg}")
            bought = success

    # Roll and move
    from_pos  = player.position
    roll      = roll_dice()
    move_logs = move_player(board, player, roll)
    logs.extend(move_logs)

    # Move breakdown for UI animation: walk from_pos → landing (dice), then
    # any jump to final (ladder climb / snake slide / bankruptcy reset).
    landing = from_pos + roll if from_pos + roll <= 100 else from_pos
    move = {"player_id": player.player_id, "name": player.name,
            "roll": roll, "from": from_pos, "landing": landing,
            "final": player.position}

    # Check win
    winner = check_winner(board)
    if winner:
        logs.append(f"\n🏆 {winner.name} wins the game!")
        return {"logs": logs, "winner": winner, "bought": bought, "move": move}

    # Advance turn
    board.next_turn()
    return {"logs": logs, "winner": None, "bought": bought, "move": move}