"""
Snakes & Lenders — Game Engine
Handles the full turn loop: rolling, moving, points, shop, and win condition.
"""

import random
from game.models import BoardState, Player, Snake
from game.board import BOMB_DEDUCTION


def roll_dice() -> int:
    return random.randint(1, 6)


def calculate_snake_cost(player: Player, head: int, tail: int) -> int:
    """
    Cost = length x 10 x multiplier
    1st snake = 1.0x, 2nd = 1.5x, 3rd = 2.0x
    """
    length     = head - tail
    multiplier = 1.0 + (player.snake_count * 0.5)
    cost       = int(length * 10 * multiplier)
    return max(cost, 30)


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
    if head > 80:
        return False, "Snake head cannot be placed above tile 80."
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

    cost = calculate_snake_cost(buyer, head, tail)
    if buyer.points < cost:
        return False, f"Not enough points. Need {cost}, have {buyer.points}."

    return True, ""


# ── Movement ──────────────────────────────────────────────────────────────────

def _apply_snakes(board: BoardState, player: Player,
                  logs: list) -> None:
    """
    Keep sliding the player down chained snakes until
    they land on a tile with no snake head.
    """
    while True:
        snake = board.get_snake_at(player.position)
        if snake:
            logs.append(f"  🐍 Snake! {player.name} slides "
                        f"from {player.position} to {snake.tail}")
            player.position = snake.tail
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
        If position + roll > 100, the player bounces back.
        e.g. tile 98 + roll 3 = would be 101 → bounces to 99.

    Players start at tile 0 — first roll enters the board.
    """
    logs    = []
    old_pos = player.position

    # ── Calculate new position with bounce-back ───────────────────
    raw = player.position + roll
    if raw > 100:
        # Bounce back from tile 100
        overshoot      = raw - 100
        new_pos        = 100 - overshoot
        bounce_message = (f"{player.name} rolled a {roll} "
                          f"— overshoots! Bounces back to tile {new_pos}")
    else:
        new_pos        = raw
        bounce_message = None

    # ── Entering the board from tile 0 ────────────────────────────
    if old_pos == 0:
        player.position = new_pos
        if bounce_message:
            logs.append(bounce_message)
        else:
            logs.append(f"{player.name} rolled a {roll} "
                        f"— enters the board at tile {new_pos}")

        # No ladder/snake check on very first entry
        # (snake heads start at tile 20+ so this is fine)
        tile_pts = board.tiles.get(player.position, 10)
        player.add_points(tile_pts)
        logs.append(f"  💰 Earned {tile_pts} points "
                    f"— total: {player.points}")
        return logs

    # ── Normal movement ───────────────────────────────────────────
    if bounce_message:
        logs.append(bounce_message)
    else:
        logs.append(f"{player.name} rolled a {roll} "
                    f"— moves from {old_pos} to {new_pos}")

    player.position = new_pos

    # Apply ladder first, then snake (order matters)
    _apply_ladders(board, player, logs)
    _apply_snakes(board, player, logs)

    # Collect tile points
    tile_pts = board.tiles.get(player.position, 10)
    player.add_points(tile_pts)
    logs.append(f"  💰 Earned {tile_pts} points "
                f"— total: {player.points}")

    # Check bomb
    if board.is_bomb(player.position):
        player.deduct_points(BOMB_DEDUCTION)
        logs.append(f"  💣 Bomb! {player.name} loses "
                    f"{BOMB_DEDUCTION} points "
                    f"— total: {player.points}")

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
    roll      = roll_dice()
    move_logs = move_player(board, player, roll)
    logs.extend(move_logs)

    # Check win
    winner = check_winner(board)
    if winner:
        logs.append(f"\n🏆 {winner.name} wins the game!")
        return {"logs": logs, "winner": winner, "bought": bought}

    # Advance turn
    board.next_turn()
    return {"logs": logs, "winner": None, "bought": bought}