"""
Snakes & Lenders — Expectimax AI (Easy Mode)
Calculates the best action each turn by averaging all 6 dice outcomes
and evaluating whether buying a snake is worth it right now.
"""

import random

from game.log import gprint
from game.models import BoardState, Player, Snake
from game.engine import (calculate_snake_cost, can_place_snake,
                         MAX_SNAKE_HEAD, MIN_SNAKE_COST, STRIKE_ZONE)


# How many dice outcomes to look ahead (always 6 for a standard die)
DICE_OUTCOMES = list(range(1, 7))

# Point-value of one tile of board progress — expresses snake damage
# (a setback) in points. Tunable for balance.
TILE_VALUE = 10.0

# Snakes trigger on an EXACT-head landing (~1/6 per pass). A cunning AI
# beats that low rate by (a) placing heads in the opponent's immediate
# dice range so they get a shot every turn while approaching, (b) using
# maximum setback so each rare hit erases a brutal amount of progress,
# and (c) forcing re-climbs: a hit drops them back through the lower
# snakes, giving those extra passes. Hit estimates by distance ahead:
# Strike-range snakes catch a passing opponent ~50%, so a well-placed
# trap is reliably worth its cost — strategy, not luck.
HIT_IN_RANGE   = 0.50      # head within ~1 die of the opponent
HIT_NEAR       = 0.35      # 7–12 tiles ahead (reached in ~2 turns)
HIT_FAR        = 0.20      # further out

# Aggressive harassment: sabotage early, spend down to a thin cushion.
# Point-theft on hits self-funds this, so heavy aggression pays off.
SABOTAGE_MIN_POS  = 10
MIN_DAMAGE_TO_BUY = 6.0
SAFETY_BUFFER     = 12

# Win-denial: a snake parked near the goal against an opponent who is
# almost home is worth extra — knocking them back from the finish line
# is the most demoralizing play available.
WIN_DENIAL_TILE = 85       # head at/above this = near-goal lurk
WIN_DENIAL_OPP  = 72       # opponent at/above this = almost winning
WIN_DENIAL_MULT = 1.8

# ── EASY-mode handicap ────────────────────────────────────────────────────────
# expectimax_decision is the EASY opponent. It plays deliberately weakly so the
# trained Hard (PPO) agent has a beatable target: it acts late, hesitates, hoards
# too much, and only ever places cheap short traps (never big knockbacks or
# win-denial lurks — those are the Hard agent's tools).
EASY_SABOTAGE_MIN_POS = 35
EASY_BUFFER           = 45
EASY_SKIP_PROB        = 0.35


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
        # Simulate moving without modifying the real board.
        # Overshoot past 100 is an invalid move — player stays put.
        raw = player.position + die
        new_pos = player.position if raw > 100 else raw

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
    Expected DAMAGE (in points) of placing a snake at head→tail —
    how much opponent progress we expect to undo.

        damage = setback_tiles × TILE_VALUE × expected_landings

    Cunning valuation: reward heads placed in the opponent's immediate
    path (caught soon), big setbacks (brutal knockback), and targeting an
    advanced opponent (erasing more progress is worth more).

    Note we return damage, not damage−cost: points are abundant (tile
    income ≫ snake cost), so the binding constraint is the 3-snake cap,
    not affordability. The buy decision spends a slot on the best target.
    """
    setback_tiles = head - tail
    best_damage = 0.0

    for other in board.players:
        if other.player_id == player.player_id:
            continue
        dist = head - other.position
        if dist <= 0:
            continue  # snake is behind this opponent — useless against them

        if dist <= 6:
            p_hit = HIT_IN_RANGE
        elif dist <= 12:
            p_hit = HIT_NEAR
        else:
            p_hit = HIT_FAR

        # Erasing an advanced opponent's progress is worth more (1.0–2.0x).
        progress_weight = 1.0 + other.position / 100.0
        damage = setback_tiles * TILE_VALUE * p_hit * progress_weight

        # Win-denial: parking a snake near the goal against an
        # almost-finished opponent is the most valuable sabotage.
        if head >= WIN_DENIAL_TILE and other.position >= WIN_DENIAL_OPP:
            damage *= WIN_DENIAL_MULT

        best_damage = max(best_damage, damage)

    return best_damage


def find_best_snake_to_buy(board: BoardState,
                            player: Player) -> tuple[int, int, float] | None:
    """
    Search all valid snake placements and return the highest-damage one.
    Returns (head, tail, expected_damage) or None if none placeable.
    """
    if not player.can_buy_snake:
        return None

    best_damage = 0.0
    best_head = None
    best_tail = None

    # Check snake placements across the board
    # Focus on tiles ahead of opponents
    for other in board.players:
        if other.player_id == player.player_id:
            continue

        # Try placing snakes just ahead of the opponent (favor the
        # immediate dice range so they get a shot next turn).
        for offset in range(1, 16):
            head = other.position + offset
            if head > MAX_SNAKE_HEAD:
                continue

            # Try a range of setbacks, including the maximum (tail=1) so
            # a hit is as brutal as possible.
            tails = {head - L for L in (5, 10, 15, 20, 25)}
            tails.add(max(1, head - 40))
            tails.add(1)
            for tail in sorted(tails):
                if tail < 1 or tail >= head:
                    continue

                valid, _ = can_place_snake(board, player, head, tail)
                if not valid:
                    continue

                # Budget-aware: skip placements we can't afford right now
                # (income is scarce — the AI must manage points too).
                if calculate_snake_cost(player, head, tail) > player.points:
                    continue

                damage = evaluate_snake_placement(board, player, head, tail)
                if damage > best_damage:
                    best_damage = damage
                    best_head = head
                    best_tail = tail

    if best_head is None:
        return None
    return (best_head, best_tail, best_damage)


# ── Main Decision Function ────────────────────────────────────────────────────

def expectimax_decision(board: BoardState, player: Player) -> dict | None:
    """
    EASY-mode AI. A deliberately weak saboteur: it reacts late, hesitates,
    hoards too many points, and only places cheap short traps. This is the
    beatable baseline the Hard (PPO) agent trains against and outplays.

    Returns a {'head','tail'} dict to place a snake, or None to just roll.
    """
    if not player.can_buy_snake:
        return None

    # Reacts only once an opponent is well advanced, and hesitates often.
    leader_pos = max((o.position for o in board.players
                      if o.player_id != player.player_id), default=0)
    if leader_pos < EASY_SABOTAGE_MIN_POS:
        return None
    if random.random() < EASY_SKIP_PROB:
        return None

    # Only ever a cheap short trap — never a big knockback or win-denial lurk.
    shop = propose_cheap_trap(board, player)
    if shop is None:
        return None

    cost = calculate_snake_cost(player, shop["head"], shop["tail"])
    if player.points - cost < EASY_BUFFER:
        return None   # over-cautious hoarding

    gprint(f"  [Easy] Buying snake {shop['head']}→{shop['tail']} (cost: {cost})")
    return shop


# ── Placement strategies (for the PPO agent's expanded action space) ──────────
#
# Each returns a {"head","tail"} dict for a placeable, affordable snake of a
# given STYLE, or None. The PPO policy chooses WHICH style to use each turn,
# so it controls strategy/economy timing itself (unlike expectimax_decision,
# which applies one fixed heuristic).

def _leading_opponent(board: BoardState, player: Player):
    opps = [o for o in board.players if o.player_id != player.player_id]
    return max(opps, key=lambda o: o.position, default=None)


def _placeable(board: BoardState, player: Player, head: int, tail: int) -> bool:
    if tail < 1 or tail >= head:
        return False
    valid, _ = can_place_snake(board, player, head, tail)
    return valid and calculate_snake_cost(player, head, tail) <= player.points


# Catch-optimal offsets: placing the head this far ahead of the opponent
# puts the whole strike zone [head-Z, head] inside their dice range
# [pos+1, pos+6], maximizing the chance they land in it (~(Z+1)/6).
def _catch_offsets():
    return range(STRIKE_ZONE + 1, 7)   # e.g. Z=3 → 4,5,6


def propose_cheap_trap(board: BoardState, player: Player) -> dict | None:
    """Short, cheap snake placed for maximum catch chance just ahead."""
    opp = _leading_opponent(board, player)
    if opp is None:
        return None
    for offset in _catch_offsets():
        head = opp.position + offset
        for length in (8, 6, 5):
            tail = head - length
            if _placeable(board, player, head, tail):
                return {"head": head, "tail": tail}
    return None


def propose_big_snake(board: BoardState, player: Player) -> dict | None:
    """Catch-optimal placement with the biggest affordable setback."""
    opp = _leading_opponent(board, player)
    if opp is None:
        return None
    best = None
    for offset in _catch_offsets():
        head = opp.position + offset
        if head > MAX_SNAKE_HEAD:
            continue
        for tail in (1, head - 40, head - 30, head - 20, head - 10):
            if _placeable(board, player, head, tail):
                if best is None or (head - tail) > (best["head"] - best["tail"]):
                    best = {"head": head, "tail": tail}
    return best


def propose_lurk(board: BoardState, player: Player) -> dict | None:
    """
    Win-denial: against an almost-finished opponent, drop a catch-optimal,
    maximum-setback snake in their path near the goal — a hit resets them
    to near the start.
    """
    opp = _leading_opponent(board, player)
    if opp is None or opp.position < 60:
        return None
    for offset in _catch_offsets():
        head = min(opp.position + offset, MAX_SNAKE_HEAD)
        if head <= opp.position:
            continue
        for tail in (1, head - 40, head - 30, head - 20):
            if _placeable(board, player, head, tail):
                return {"head": head, "tail": tail}
    return None