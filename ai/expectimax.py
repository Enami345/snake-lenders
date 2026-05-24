"""
Snakes & Lenders — Expectimax AI (Easy Mode)
Calculates the best action each turn by averaging all 6 dice outcomes
and evaluating whether buying a snake is worth it right now.
"""

from game.models import BoardState, Player, Snake
from game.engine import calculate_snake_cost, can_place_snake


# How many dice outcomes to look ahead (always 6 for a standard die)
DICE_OUTCOMES = list(range(1, 7))


# ── Board Evaluation ──────────────────────────────────────────────────────────

def evaluate_state(board: BoardState, player: Player) -> float:
    """
    Score the board from the perspective of one player.
    Higher = better for that player.

    Factors considered:
    - Player's position (closer to 100 = better)
    - Player's points (more = better)
    - Opponent positions (further behind = better for us)
    - Active snakes ahead of opponents (more danger for them = better for us)
    """
    score = 0.0

    # 1. Our position — heavily weighted
    score += player.position * 10.0

    # 2. Our points — useful for buying snakes
    score += player.points * 0.1

    # 3. Penalize opponents who are ahead of us
    for other in board.players:
        if other.player_id == player.player_id:
            continue
        position_diff = player.position - other.position
        score += position_diff * 5.0  # Positive if we're ahead, negative if behind

    # 4. Bonus for each snake we own that threatens opponents
    for snake in board.snakes:
        if snake.owner_id != player.player_id:
            continue
        for other in board.players:
            if other.player_id == player.player_id:
                continue
            # Snake is valuable if it's ahead of the opponent
            if snake.head > other.position:
                tiles_away = snake.head - other.position
                # More valuable if close to the opponent (1-6 tiles away = rollable)
                if tiles_away <= 6:
                    score += 50.0
                elif tiles_away <= 12:
                    score += 20.0
                else:
                    score += 5.0

    return score


# ── Expected Value of a Dice Roll ─────────────────────────────────────────────

def expected_value_after_roll(board: BoardState, player: Player) -> float:
    """
    Calculate the average board score across all 6 dice outcomes.
    This is the 'chance node' in the Expectimax tree.
    """
    total = 0.0

    for die in DICE_OUTCOMES:
        # Simulate moving without modifying the real board
        new_pos = min(player.position + die, 100)

        # Apply ladder
        for ladder in board.ladders:
            if ladder.bottom == new_pos:
                new_pos = ladder.top
                break

        # Apply snake
        for snake in board.snakes:
            if snake.head == new_pos:
                new_pos = snake.tail
                break

        # Temporarily update position to evaluate
        old_pos = player.position
        player.position = new_pos
        score = evaluate_state(board, player)
        player.position = old_pos  # Restore

        total += score

    return total / 6.0  # Average across all 6 outcomes


# ── Snake Placement Evaluation ────────────────────────────────────────────────

def evaluate_snake_placement(board: BoardState, player: Player,
                              head: int, tail: int) -> float:
    """
    Calculate the Return on Investment (ROI) of placing a snake at head→tail.

    ROI = (probability opponent lands on it) x (setback value) - cost
    """
    cost = calculate_snake_cost(player, head, tail)

    # Calculate probability opponent lands on the snake head within 3 turns
    total_prob = 0.0
    for other in board.players:
        if other.player_id == player.player_id:
            continue

        # Simulate 3 turns of dice rolls (look-ahead)
        for die1 in DICE_OUTCOMES:
            pos1 = min(other.position + die1, 100)
            prob1 = 1 / 6.0
            if pos1 == head:
                total_prob += prob1
            for die2 in DICE_OUTCOMES:
                pos2 = min(pos1 + die2, 100)
                prob2 = prob1 / 6.0
                if pos2 == head:
                    total_prob += prob2
                for die3 in DICE_OUTCOMES:
                    pos3 = min(pos2 + die3, 100)
                    prob3 = prob2 / 6.0
                    if pos3 == head:
                        total_prob += prob3

    # Setback value = how far back the snake sends the opponent
    setback = (head - tail) * 10.0

    # ROI = expected damage - cost
    roi = (total_prob * setback) - (cost * 0.01)
    return roi


def find_best_snake_to_buy(board: BoardState,
                            player: Player) -> tuple[int, int, float] | None:
    """
    Search all valid snake placements and return the best one.
    Returns (head, tail, roi) or None if no profitable placement exists.
    """
    if not player.can_buy_snake:
        return None

    best_roi = -1.0  # Only buy if ROI is positive
    best_head = None
    best_tail = None

    # Check snake placements across the board
    # Focus on tiles ahead of opponents
    for other in board.players:
        if other.player_id == player.player_id:
            continue

        # Try placing snakes 1 to 15 tiles ahead of the opponent
        for offset in range(1, 16):
            head = other.position + offset
            if head > 80 or head > 99:
                continue

            # Try various snake lengths
            for length in range(5, 26, 5):
                tail = head - length
                if tail < 1:
                    continue

                valid, _ = can_place_snake(board, player, head, tail)
                if not valid:
                    continue

                roi = evaluate_snake_placement(board, player, head, tail)
                if roi > best_roi:
                    best_roi = roi
                    best_head = head
                    best_tail = tail

    if best_head is None:
        return None
    return (best_head, best_tail, best_roi)


# ── Main Decision Function ────────────────────────────────────────────────────

def expectimax_decision(board: BoardState, player: Player) -> dict | None:
    """
    Main entry point for the Expectimax AI.
    Called each turn to decide whether to buy a snake or just roll.

    Returns:
        dict with 'head' and 'tail' if buying a snake, or None to just roll.
    """
    if not player.can_buy_snake or player.points < 100:
        return None  # Can't afford anything or already at max snakes

    # Calculate expected value of just rolling (baseline)
    ev_roll = expected_value_after_roll(board, player)

    # Find best snake to buy
    result = find_best_snake_to_buy(board, player)

    if result is None:
        return None

    head, tail, roi = result

    # Only buy if the snake placement improves our position meaningfully
    cost = calculate_snake_cost(player, head, tail)
    if roi > 0 and player.points >= cost:
        print(f"  [Expectimax] Buying snake {head}→{tail} "
              f"(ROI: {roi:.2f}, cost: {cost})")
        return {"head": head, "tail": tail}

    return None