"""
Snakes & Lenders — Data Models
Defines all core data structures used throughout the game.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Snake:
    head: int           # Higher tile number (snake bites here)
    tail: int           # Lower tile number (player lands here after sliding)
    owner_id: int       # -1 = initial board snake, 0+ = player ID

    def __post_init__(self):
        if self.head <= self.tail:
            raise ValueError(f"Snake head ({self.head}) must be higher than tail ({self.tail})")

    @property
    def length(self) -> int:
        return self.head - self.tail


@dataclass
class Ladder:
    bottom: int         # Lower tile (player steps on this)
    top: int            # Higher tile (player jumps to this)

    def __post_init__(self):
        if self.bottom >= self.top:
            raise ValueError(f"Ladder bottom ({self.bottom}) must be lower than top ({self.top})")


@dataclass
class Player:
    player_id: int
    name: str
    position: int = 0        # Players start at tile 0 (off the board)
    points: int = 0
    snakes_owned: list = field(default_factory=list)
    is_ai: bool = False
    ai_difficulty: Optional[str] = None  # 'easy' or 'hard'
    bankrupt_count: int = 0

    @property
    def snake_count(self) -> int:
        return len(self.snakes_owned)

    @property
    def can_buy_snake(self) -> bool:
        return self.snake_count < 3

    def add_points(self, amount: int):
        self.points += amount

    def deduct_points(self, amount: int) -> bool:
        """Returns False if player goes bankrupt."""
        self.points -= amount
        if self.points < 0:
            self.go_bankrupt()
            return False
        return True

    def go_bankrupt(self):
        """Reset player to start due to bankruptcy."""
        self.position = 0        # Back to off the board
        self.points = 0
        self.bankrupt_count += 1
        print(f"  [BANKRUPT] {self.name} went bankrupt and returns to tile 0!")


@dataclass
class BoardState:
    tiles: dict           # tile_number -> point_value
    ladders: list         # List of Ladder objects
    snakes: list          # List of Snake objects
    bombs: list           # List of tile numbers
    players: list         # List of Player objects
    current_turn: int = 0
    turn_number: int = 1

    def get_snake_at(self, tile: int):
        for s in self.snakes:
            if s.head == tile:
                return s
        return None

    def get_ladder_at(self, tile: int):
        for l in self.ladders:
            if l.bottom == tile:
                return l
        return None

    def is_bomb(self, tile: int) -> bool:
        return tile in self.bombs

    def occupied_tiles(self) -> set:
        # Tile 0 is off the board so don't count it as occupied
        return {p.position for p in self.players if p.position > 0}

    @property
    def active_player(self):
        return self.players[self.current_turn % len(self.players)]

    def next_turn(self):
        self.current_turn = (self.current_turn + 1) % len(self.players)
        if self.current_turn == 0:
            self.turn_number += 1