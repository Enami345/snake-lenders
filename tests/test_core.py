"""
Core rule tests for Snakes & Lenders.

Run from the project root:
    python -m unittest tests.test_core
    (or)  python tests/test_core.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.models import Player, BoardState, Snake, Ladder
from game.board import (generate_board, bfs_expected_turns,
                        NUM_LADDERS, NUM_INIT_SNAKES, NUM_BOMBS, MIN_AVG_TURNS)
from game.engine import (move_player, calculate_snake_cost, can_place_snake,
                         MAX_SNAKE_HEAD, MIN_SNAKE_COST, STRIKE_ZONE)


def _board(snakes=None, ladders=None, bombs=None, players=None, tiles=None):
    return BoardState(
        tiles=tiles or {i: 5 for i in range(1, 101)},
        ladders=ladders or [],
        snakes=snakes or [],
        bombs=bombs or [],
        players=players or [Player(0, "X")],
    )


class TestBoardGeneration(unittest.TestCase):
    def test_counts_and_solvability(self):
        b = generate_board(seed=123, players=[Player(0, "A"), Player(1, "B")])
        self.assertEqual(len(b.ladders), NUM_LADDERS)
        self.assertEqual(len(b.snakes), NUM_INIT_SNAKES)
        self.assertEqual(len(b.bombs), NUM_BOMBS)
        self.assertGreaterEqual(
            bfs_expected_turns(b.ladders, b.snakes), MIN_AVG_TURNS)


class TestSnakeCost(unittest.TestCase):
    def test_monotonic_and_floor(self):
        p = Player(0, "A")
        c5 = calculate_snake_cost(p, 50, 45)    # length 5
        c20 = calculate_snake_cost(p, 50, 30)   # length 20
        self.assertGreater(c20, c5)
        self.assertGreaterEqual(c5, MIN_SNAKE_COST)

    def test_rises_per_purchase(self):
        p = Player(0, "A")
        first = calculate_snake_cost(p, 50, 40)
        p.snakes_owned = [Snake(60, 50, 0)]
        second = calculate_snake_cost(p, 50, 40)
        self.assertGreater(second, first)


class TestMovement(unittest.TestCase):
    def test_overshoot_stays_put(self):
        b = _board()
        p = b.players[0]; p.position = 97
        move_player(b, p, 5)            # 97+5=102 > 100
        self.assertEqual(p.position, 97)

    def test_exact_roll_wins(self):
        b = _board()
        p = b.players[0]; p.position = 97
        move_player(b, p, 3)
        self.assertEqual(p.position, 100)

    def test_entry_climbs_ladder(self):
        b = _board(ladders=[Ladder(3, 40)])
        p = b.players[0]
        move_player(b, p, 3)           # enter from 0 onto ladder bottom 3
        self.assertEqual(p.position, 40)


class TestSnakeTriggers(unittest.TestCase):
    def test_player_snake_strike_range(self):
        # Player snake (owner 1) head 50, victim is player 0.
        victim = Player(0, "V"); other = Player(1, "O")
        victim.points = 200                  # enough that theft won't bankrupt
        b = _board(snakes=[Snake(50, 10, owner_id=1)],
                   players=[victim, other])
        victim.position = 50 - STRIKE_ZONE   # land inside strike zone next roll
        move_player(b, victim, STRIKE_ZONE)  # lands exactly on head 50
        self.assertEqual(victim.position, 10)

    def test_jump_clean_over_is_safe(self):
        victim = Player(0, "V"); other = Player(1, "O")
        b = _board(snakes=[Snake(50, 10, owner_id=1)],
                   players=[victim, other])
        victim.position = 49
        move_player(b, victim, 5)            # 49->54, above head 50 = safe
        self.assertEqual(victim.position, 54)

    def test_owner_immunity(self):
        owner = Player(0, "Owner")
        b = _board(snakes=[Snake(50, 10, owner_id=0)], players=[owner])
        owner.position = 49
        move_player(b, owner, 1)             # lands on own head 50 — immune
        self.assertEqual(owner.position, 50)

    def test_board_snake_exact_head_only(self):
        p = Player(0, "X")
        b = _board(snakes=[Snake(50, 10, owner_id=-1)], players=[p])
        p.position = 50 - 1                  # within a player strike zone, but board=exact
        move_player(b, p, 1)                 # lands on 50 exactly -> bite
        self.assertEqual(p.position, 10)
        p.position = 50 - 2
        move_player(b, p, 1)                 # lands on 49, not head -> safe
        self.assertEqual(p.position, 49)


class TestPlacementRules(unittest.TestCase):
    def setUp(self):
        self.buyer = Player(0, "B"); self.buyer.points = 9999
        self.board = _board(players=[self.buyer, Player(1, "O")])

    def test_head_cap(self):
        ok, _ = can_place_snake(self.board, self.buyer, MAX_SNAKE_HEAD + 1, 50)
        self.assertFalse(ok)

    def test_no_wall(self):
        self.board.snakes = [Snake(51, 40, 0), Snake(52, 41, 0)]
        ok, _ = can_place_snake(self.board, self.buyer, 53, 42)  # run 51,52,53
        self.assertFalse(ok)

    def test_no_chain(self):
        self.board.snakes = [Snake(60, 50, 0)]
        ok, _ = can_place_snake(self.board, self.buyer, 50, 40)  # head on a tail
        self.assertFalse(ok)


class TestBankruptcy(unittest.TestCase):
    def test_bomb_can_bankrupt_to_tile_zero(self):
        p = Player(0, "X"); p.position = 19; p.points = 1
        b = _board(bombs=[20], players=[p])
        move_player(b, p, 1)                 # land on bomb tile 20 with ~1 pt
        self.assertEqual(p.position, 0)      # reset to start
        self.assertEqual(p.points, 0)
        self.assertEqual(p.bankrupt_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
