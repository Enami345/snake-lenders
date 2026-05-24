"""
Snakes & Lenders — Pygame UI Renderer
Classic yellow/blue checkerboard board with side player info panel.
"""

import pygame
import sys

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_W   = 1000
WINDOW_H   = 700
BOARD_SIZE = 700
PANEL_W    = WINDOW_W - BOARD_SIZE   # 300px side panel
TILE_SIZE  = BOARD_SIZE // 10        # 70px per tile
FPS        = 60

# ── Colors ────────────────────────────────────────────────────────────────────

C_TILE_YELLOW  = (255, 230, 100)    # Classic yellow tile
C_TILE_BLUE    = (100, 170, 255)    # Classic blue tile
C_SNAKE_HEAD   = (220, 50,  50)     # Red — snake head
C_SNAKE_TAIL   = (180, 100, 60)     # Brown — snake tail
C_LADDER_BOT   = (80,  200, 100)    # Green — ladder bottom
C_LADDER_TOP   = (40,  140, 70)     # Dark green — ladder top
C_BOMB         = (70,  70,  70)     # Dark grey — bomb tile
C_GOAL         = (255, 200, 50)     # Gold — tile 100
C_PANEL_BG     = (30,  30,  50)     # Dark panel background
C_BORDER       = (60,  60,  90)     # Tile border
C_TEXT         = (240, 240, 240)    # White text
C_TEXT_DIM     = (150, 150, 170)    # Dimmed text
C_GOLD         = (255, 200, 50)     # Gold highlight
C_WHITE        = (255, 255, 255)
C_BLACK        = (0,   0,   0)
C_GREEN        = (80,  200, 100)
C_RED          = (220, 50,  50)
C_BG           = (20,  20,  35)

# Player token colors
PLAYER_COLORS = [
    (255, 80,  80),    # Red
    (80,  150, 255),   # Blue
    (80,  210, 110),   # Green
    (255, 190, 60),    # Yellow
]

FONT_TILE  = None
FONT_SMALL = None
FONT_MED   = None
FONT_LARGE = None
FONT_ICON  = None


def init_fonts():
    global FONT_TILE, FONT_SMALL, FONT_MED, FONT_LARGE, FONT_ICON
    FONT_TILE  = pygame.font.SysFont("Arial", 11, bold=True)
    FONT_SMALL = pygame.font.SysFont("Arial", 13)
    FONT_MED   = pygame.font.SysFont("Arial", 15, bold=True)
    FONT_LARGE = pygame.font.SysFont("Arial", 18, bold=True)
    FONT_ICON  = pygame.font.SysFont("Segoe UI Emoji", 16)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def tile_to_screen(tile: int) -> tuple[int, int]:
    """Convert tile number (1-100) to top-left pixel on screen."""
    tile -= 1
    row   = tile // 10
    col   = tile  % 10
    if row % 2 == 1:
        col = 9 - col
    screen_row = 9 - row
    x = screen_col = col
    x = screen_col * TILE_SIZE
    y = screen_row * TILE_SIZE
    return x, y


def tile_center(tile: int) -> tuple[int, int]:
    x, y = tile_to_screen(tile)
    return x + TILE_SIZE // 2, y + TILE_SIZE // 2


# ── Board drawing ─────────────────────────────────────────────────────────────

def draw_board(surface, board):
    """Draw the 10x10 classic checkerboard grid."""
    snake_heads  = {s.head   for s in board.snakes}
    snake_tails  = {s.tail   for s in board.snakes}
    ladder_bots  = {l.bottom for l in board.ladders}
    ladder_tops  = {l.top    for l in board.ladders}
    bombs        = set(board.bombs)

    for tile in range(1, 101):
        x, y = tile_to_screen(tile)
        rect  = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

        # Choose tile color
        if tile == 100:
            color = C_GOAL
        elif tile in snake_heads:
            color = C_SNAKE_HEAD
        elif tile in snake_tails:
            color = C_SNAKE_TAIL
        elif tile in ladder_bots:
            color = C_LADDER_BOT
        elif tile in ladder_tops:
            color = C_LADDER_TOP
        elif tile in bombs:
            color = C_BOMB
        else:
            # Classic yellow/blue checkerboard
            row = (tile - 1) // 10
            col = (tile - 1) % 10
            color = C_TILE_YELLOW if (row + col) % 2 == 0 else C_TILE_BLUE

        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, C_BORDER, rect, 1)

        # Tile number — top left corner
        num_surf = FONT_TILE.render(str(tile), True, C_BLACK)
        surface.blit(num_surf, (x + 3, y + 3))

        # Icon for special tiles
        cx = x + TILE_SIZE // 2 - 10
        cy = y + TILE_SIZE // 2 - 10
        if tile in snake_heads:
            surface.blit(FONT_ICON.render("🐍", True, C_WHITE), (cx, cy))
        elif tile in ladder_bots:
            surface.blit(FONT_ICON.render("🪜", True, C_WHITE), (cx, cy))
        elif tile in bombs:
            surface.blit(FONT_ICON.render("💣", True, C_WHITE), (cx, cy))
        elif tile == 100:
            surface.blit(FONT_ICON.render("🏆", True, C_BLACK), (cx, cy))


def draw_connections(surface, board):
    """Draw lines connecting snake head→tail and ladder bottom→top."""
    # Snakes — red lines
    for snake in board.snakes:
        hx, hy = tile_center(snake.head)
        tx, ty = tile_center(snake.tail)
        pygame.draw.line(surface, (190, 30, 30), (hx, hy), (tx, ty), 4)
        pygame.draw.circle(surface, C_SNAKE_HEAD, (hx, hy), 7)
        pygame.draw.circle(surface, C_SNAKE_TAIL, (tx, ty), 5)

    # Ladders — green lines
    for ladder in board.ladders:
        bx, by = tile_center(ladder.bottom)
        tx, ty = tile_center(ladder.top)
        pygame.draw.line(surface, (40, 160, 70), (bx, by), (tx, ty), 4)
        pygame.draw.circle(surface, C_LADDER_BOT, (bx, by), 7)
        pygame.draw.circle(surface, C_LADDER_TOP, (tx, ty), 5)


def draw_players(surface, board):
    """Draw player tokens on their current tiles."""
    tile_players = {}
    for p in board.players:
        if p.position > 0:
            tile_players.setdefault(p.position, []).append(p)

    for tile, players in tile_players.items():
        cx, cy = tile_center(tile)
        count  = len(players)
        for i, player in enumerate(players):
            offset_x = (i - count // 2) * 16
            color    = PLAYER_COLORS[player.player_id % len(PLAYER_COLORS)]
            pygame.draw.circle(surface, color,
                               (cx + offset_x, cy), 12)
            pygame.draw.circle(surface, C_WHITE,
                               (cx + offset_x, cy), 12, 2)
            initial = FONT_TILE.render(player.name[0], True, C_BLACK)
            surface.blit(initial, (cx + offset_x - 4, cy - 6))


# ── Legend ────────────────────────────────────────────────────────────────────

def draw_legend(surface):
    """Small legend strip at the very bottom of the board."""
    items = [
        (C_SNAKE_HEAD, "Snake head"),
        (C_SNAKE_TAIL, "Snake tail"),
        (C_LADDER_BOT, "Ladder"),
        (C_BOMB,       "Bomb"),
        (C_GOAL,       "Goal"),
    ]
    x = 4
    y = BOARD_SIZE - 14
    for color, label in items:
        pygame.draw.rect(surface, color, pygame.Rect(x, y, 10, 10))
        text = FONT_TILE.render(label, True, C_WHITE)
        surface.blit(text, (x + 13, y))
        x += 110


# ── Side panel ────────────────────────────────────────────────────────────────

def draw_panel(surface, board, active_msg=""):
    """Draw the right-side player info panel."""
    px         = BOARD_SIZE
    panel_rect = pygame.Rect(px, 0, PANEL_W, WINDOW_H)
    pygame.draw.rect(surface, C_PANEL_BG, panel_rect)
    pygame.draw.line(surface, C_BORDER, (px, 0), (px, WINDOW_H), 2)

    y = 20

    # Title
    title = FONT_LARGE.render("SNAKES & LENDERS", True, C_GOLD)
    surface.blit(title, (px + 10, y))
    y += 30

    sub = FONT_SMALL.render("First to tile 100 wins!", True, C_TEXT_DIM)
    surface.blit(sub, (px + 10, y))
    y += 25

    pygame.draw.line(surface, C_BORDER,
                     (px + 5, y), (WINDOW_W - 5, y), 1)
    y += 15

    # Turn info
    active    = board.active_player
    turn_surf = FONT_MED.render(
        f"Turn {board.turn_number}", True, C_GOLD)
    surface.blit(turn_surf, (px + 10, y))
    y += 25

    # Active message (instructions)
    if active_msg:
        # Word wrap at 28 chars
        words    = active_msg.split()
        line     = ""
        for word in words:
            if len(line) + len(word) + 1 <= 28:
                line += ("" if not line else " ") + word
            else:
                msg_s = FONT_SMALL.render(line, True, C_GREEN)
                surface.blit(msg_s, (px + 10, y))
                y    += 18
                line  = word
        if line:
            msg_s = FONT_SMALL.render(line, True, C_GREEN)
            surface.blit(msg_s, (px + 10, y))
            y += 18
    y += 10

    pygame.draw.line(surface, C_BORDER,
                     (px + 5, y), (WINDOW_W - 5, y), 1)
    y += 15

    # Player cards
    header = FONT_MED.render("PLAYERS", True, C_TEXT_DIM)
    surface.blit(header, (px + 10, y))
    y += 22

    for player in board.players:
        color     = PLAYER_COLORS[player.player_id % len(PLAYER_COLORS)]
        is_active = player.player_id == active.player_id

        # Card background
        card = pygame.Rect(px + 5, y, PANEL_W - 10, 90)
        pygame.draw.rect(surface, (40, 40, 65), card, border_radius=8)
        if is_active:
            pygame.draw.rect(surface, color, card, 2, border_radius=8)

        # Token dot
        pygame.draw.circle(surface, color, (px + 20, y + 20), 10)
        pygame.draw.circle(surface, C_WHITE, (px + 20, y + 20), 10, 2)
        initial = FONT_TILE.render(player.name[0], True, C_BLACK)
        surface.blit(initial, (px + 16, y + 14))

        # Name
        tag  = ""
        if player.ai_difficulty == "easy": tag = " [Easy AI]"
        if player.ai_difficulty == "hard": tag = " [Hard AI]"
        name = FONT_MED.render(
            player.name + tag,
            True, color if is_active else C_TEXT)
        surface.blit(name, (px + 35, y + 8))

        # Stats
        stats = f"Tile: {player.position:3d}   Pts: {player.points:5d}"
        surface.blit(FONT_SMALL.render(stats, True, C_TEXT_DIM),
                     (px + 35, y + 28))

        snakes_txt = f"Snakes owned: {player.snake_count}/3"
        surface.blit(FONT_SMALL.render(snakes_txt, True, C_TEXT_DIM),
                     (px + 35, y + 44))

        # Progress bar
        bar_w    = PANEL_W - 50
        bar_rect = pygame.Rect(px + 15, y + 62, bar_w, 10)
        fill_w   = int(bar_w * player.position / 100)
        fill     = pygame.Rect(px + 15, y + 62, fill_w, 10)
        pygame.draw.rect(surface, C_BORDER, bar_rect, border_radius=5)
        pygame.draw.rect(surface, color,    fill,     border_radius=5)

        # Progress label
        pct = FONT_TILE.render(
            f"{player.position}%", True, C_TEXT_DIM)
        surface.blit(pct, (px + 15 + fill_w + 4, y + 62))

        y += 102

    pygame.draw.line(surface, C_BORDER,
                     (px + 5, y), (WINDOW_W - 5, y), 1)
    y += 15

    # Controls hint
    controls = [
        "CONTROLS:",
        "SPACE — roll / continue",
        "B     — open snake shop",
        "ENTER — confirm input",
        "ESC   — cancel",
        "Q     — quit",
    ]
    for line in controls:
        col  = C_GOLD if line == "CONTROLS:" else C_TEXT_DIM
        surf = FONT_SMALL.render(line, True, col)
        surface.blit(surf, (px + 10, y))
        y += 18


# ── Main Renderer Class ───────────────────────────────────────────────────────

class GameRenderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Snakes & Lenders")
        self.clock  = pygame.time.Clock()
        self.active_msg = ""
        init_fonts()

    def render(self, board):
        self.screen.fill(C_BG)
        draw_board(self.screen, board)
        draw_connections(self.screen, board)
        draw_players(self.screen, board)
        draw_legend(self.screen)
        draw_panel(self.screen, board, self.active_msg)
        pygame.display.flip()

    def wait_for_space(self, board, message="Press SPACE to continue..."):
        self.active_msg = message
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
            self.render(board)
            self.clock.tick(FPS)
        self.active_msg = ""

    def run(self, board, mode="hvh", ppo_model=None):
        from game.engine import do_turn, calculate_snake_cost, can_place_snake
        from ai.expectimax import expectimax_decision

        while True:
            player = board.active_player

            self.wait_for_space(
                board,
                f"Turn {board.turn_number} — {player.name}'s turn. "
                f"Press SPACE to roll."
            )

            # Shop phase
            shop_decision = None
            if not player.is_ai and player.position > 0:
                shop_decision = self._human_shop(board, player)
            elif player.is_ai:
                if player.ai_difficulty == "easy":
                    shop_decision = expectimax_decision(board, player)
                elif player.ai_difficulty == "hard":
                    if ppo_model:
                        from ai.ppo_agent import ppo_decision
                        shop_decision = ppo_decision(
                            board, player, ppo_model)
                    else:
                        shop_decision = expectimax_decision(board, player)

            # Execute turn
            result = do_turn(board, shop_decision=shop_decision)
            self.render(board)

            # Win condition
            if result["winner"]:
                winner = result["winner"]
                self.wait_for_space(
                    board,
                    f"🏆 {winner.name} wins! Press SPACE to exit."
                )
                pygame.quit()
                return

        pygame.quit()

    def _human_shop(self, board, player) -> dict | None:
        from game.engine import calculate_snake_cost, can_place_snake

        if not player.can_buy_snake:
            return None

        # Ask to open shop
        self.active_msg = (
            f"Buy a snake? Press B to shop, "
            f"SPACE to skip. You have {player.points} pts."
        )
        open_shop = False
        waiting   = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        open_shop = True
                        waiting   = False
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
            self.render(board)
            self.clock.tick(FPS)

        if not open_shop:
            return None

        # Get head tile
        head = self._get_number_input(
            board,
            f"SNAKE HEAD tile (20-80). Type number + ENTER.",
            20, 80
        )
        if head is None:
            return None

        # Get tail tile
        tail = self._get_number_input(
            board,
            f"SNAKE TAIL tile (1-{head - 1}). Type number + ENTER.",
            1, head - 1
        )
        if tail is None:
            return None

        # Validate
        valid, reason = can_place_snake(board, player, head, tail)
        if not valid:
            self.active_msg = f"❌ {reason} — Press SPACE"
            self.wait_for_space(board)
            return None

        # Confirm
        cost = calculate_snake_cost(player, head, tail)
        self.active_msg = (
            f"Snake {head}→{tail} costs {cost} pts. "
            f"Press B to confirm, SPACE to cancel."
        )
        confirm = False
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        confirm = True
                        waiting = False
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
            self.render(board)
            self.clock.tick(FPS)

        return {"head": head, "tail": tail} if confirm else None

    def _get_number_input(self, board, prompt,
                           min_val, max_val) -> int | None:
        input_str = ""
        while True:
            self.active_msg = f"{prompt} [{input_str}]"
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_RETURN:
                        try:
                            val = int(input_str)
                            if min_val <= val <= max_val:
                                return val
                            input_str = ""
                        except ValueError:
                            input_str = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_str = input_str[:-1]
                    elif event.unicode.isdigit():
                        input_str += event.unicode
            self.render(board)
            self.clock.tick(FPS)