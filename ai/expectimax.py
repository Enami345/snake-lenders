"""Snakes & Lenders — Expectimax AI (Easy Mode)"""

import random

from game.log import gprint
from game.models import BoardState, Player, Snake
from game.engine import (calculate_snake_cost, can_place_snake,
                         MAX_SNAKE_HEAD, MIN_SNAKE_COST, STRIKE_ZONE)

DICE_OUTCOMES = list(range(1, 7))

TILE_VALUE     = 10.0   # 1 tile setback ≈ 10 pts damage

HIT_IN_RANGE   = 0.50   # hit prob: head 1-6 tiles ahead of opponent
HIT_NEAR       = 0.35   # hit prob: 7-12 tiles ahead
HIT_FAR        = 0.20   # hit prob: >12 tiles ahead

SABOTAGE_MIN_POS  = 10  # strong_decision: don't attack before this tile
MIN_DAMAGE_TO_BUY = 6.0
SAFETY_BUFFER     = 12  # strong_decision: min points kept after buy

WIN_DENIAL_TILE = 85    # lurk zone: snake head at/above = near-goal trap
WIN_DENIAL_OPP  = 72    # lurk trigger: opponent this close to winning
WIN_DENIAL_MULT = 1.8   # damage multiplier for win-denial placements

# Easy-mode handicaps: late, hesitant, hoards, cheap traps only
EASY_SABOTAGE_MIN_POS = 35
EASY_BUFFER           = 45
EASY_SKIP_PROB        = 0.35


def evaluate_state(board: BoardState, player: Player) -> float:
    """Board score from player's perspective (position + points + lead + snake threats)."""
    score = player.position * 10.0 + player.points * 0.1
    for other in board.players:
        if other.player_id == player.player_id:
            continue
        score += (player.position - other.position) * 5.0
        for snake in board.snakes:
            if snake.owner_id != player.player_id:
                continue
            if snake.head > other.position:
                d = snake.head - other.position
                score += 50.0 if d <= 6 else 20.0 if d <= 12 else 5.0
    return score


def expected_value_after_roll(board: BoardState, player: Player) -> float:
    """Expectimax chance node: average evaluate_state across all 6 dice outcomes."""
    total = 0.0
    for die in DICE_OUTCOMES:
        raw = player.position + die
        new_pos = player.position if raw > 100 else raw
        for ladder in board.ladders:
            if ladder.bottom == new_pos:
                new_pos = ladder.top
                break
        for snake in board.snakes:
            if snake.head == new_pos:
                new_pos = snake.tail
                break
        old_pos = player.position
        player.position = new_pos
        total += evaluate_state(board, player)
        player.position = old_pos
    return total / 6.0


def evaluate_snake_placement(board: BoardState, player: Player,
                              head: int, tail: int) -> float:
    """Expected damage of placing a snake: setback × TILE_VALUE × p_hit × progress_weight."""
    setback_tiles = head - tail
    best_damage = 0.0
    for other in board.players:
        if other.player_id == player.player_id:
            continue
        dist = head - other.position
        if dist <= 0:
            continue
        p_hit = HIT_IN_RANGE if dist <= 6 else HIT_NEAR if dist <= 12 else HIT_FAR
        progress_weight = 1.0 + other.position / 100.0
        damage = setback_tiles * TILE_VALUE * p_hit * progress_weight
        if head >= WIN_DENIAL_TILE and other.position >= WIN_DENIAL_OPP:
            damage *= WIN_DENIAL_MULT
        best_damage = max(best_damage, damage)
    return best_damage


def find_best_snake_to_buy(board: BoardState,
                            player: Player) -> tuple[int, int, float] | None:
    """Search all valid placements; return (head, tail, damage) with highest damage or None."""
    if not player.can_buy_snake:
        return None
    best_damage, best_head, best_tail = 0.0, None, None
    for other in board.players:
        if other.player_id == player.player_id:
            continue
        for offset in range(1, 16):
            head = other.position + offset
            if head > MAX_SNAKE_HEAD:
                continue
            tails = {head - L for L in (5, 10, 15, 20, 25)}
            tails.add(max(1, head - 40))
            tails.add(1)
            for tail in sorted(tails):
                if tail < 1 or tail >= head:
                    continue
                valid, _ = can_place_snake(board, player, head, tail)
                if not valid or calculate_snake_cost(player, head, tail) > player.points:
                    continue
                damage = evaluate_snake_placement(board, player, head, tail)
                if damage > best_damage:
                    best_damage, best_head, best_tail = damage, head, tail
    return (best_head, best_tail, best_damage) if best_head else None


def expectimax_decision(board: BoardState, player: Player) -> dict | None:
    """Easy bot: deliberately weak — reacts late, hesitates, hoards, cheap traps only."""
    if not player.can_buy_snake:
        return None
    leader_pos = max((o.position for o in board.players
                      if o.player_id != player.player_id), default=0)
    if leader_pos < EASY_SABOTAGE_MIN_POS:
        return None
    if random.random() < EASY_SKIP_PROB:
        return None
    shop = propose_cheap_trap(board, player)
    if shop is None:
        return None
    cost = calculate_snake_cost(player, shop["head"], shop["tail"])
    if player.points - cost < EASY_BUFFER:
        return None
    gprint(f"  [{player.name}] buys snake {shop['head']}→{shop['tail']} (cost: {cost})")
    return shop


def strong_decision(board: BoardState, player: Player) -> dict | None:
    """Strong training opponent: lurk near goal, big knockbacks, cheap traps as fallback."""
    if not player.can_buy_snake:
        return None
    opp = _leading_opponent(board, player)
    if opp is None or opp.position < 8:
        return None
    strategies = (propose_lurk,) if opp.position >= 60 else ()
    strategies += (propose_big_snake, propose_cheap_trap)
    for fn in strategies:
        shop = fn(board, player)
        if shop:
            cost = calculate_snake_cost(player, shop["head"], shop["tail"])
            if player.points - cost >= 35:
                return shop
    return None


def _leading_opponent(board: BoardState, player: Player):
    opps = [o for o in board.players if o.player_id != player.player_id]
    return max(opps, key=lambda o: o.position, default=None)


def _placeable(board: BoardState, player: Player, head: int, tail: int) -> bool:
    if tail < 1 or tail >= head:
        return False
    valid, _ = can_place_snake(board, player, head, tail)
    return valid and calculate_snake_cost(player, head, tail) <= player.points


def _catch_offsets():
    """Head offsets that land in opponent's dice range (1-6 tiles ahead)."""
    return range(STRIKE_ZONE + 1, 7)


def propose_cheap_trap(board: BoardState, player: Player) -> dict | None:
    """Short cheap snake in opponent's dice range for maximum catch chance."""
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
    """Longest affordable snake in opponent's dice range for max setback."""
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


def propose_combo(board: BoardState, player: Player) -> dict | None:
    """Snake with tail on a bomb tile — victim eats knockback + bomb damage."""
    opp = _leading_opponent(board, player)
    if opp is None or not board.bombs:
        return None
    best, best_setback = None, 0
    for tail in board.bombs:
        for offset in _catch_offsets():
            head = opp.position + offset
            if head > MAX_SNAKE_HEAD or head <= tail:
                continue
            if _placeable(board, player, head, tail):
                setback = head - tail
                if setback > best_setback:
                    best_setback, best = setback, {"head": head, "tail": tail}
    return best


def propose_lurk(board: BoardState, player: Player) -> dict | None:
    """Win-denial: max-setback snake near goal against almost-finished opponent."""
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
