"""
Snakes & Lenders — Main Entry Point

Interactive (asks players / humans / difficulty, then plays):
    python main.py
    python main.py --console            # terminal instead of Pygame UI

Skip the prompts with flags:
    python main.py --players 4 --humans 1 --difficulty hard
    python main.py --players 2 --humans 2                 # local 2-human
    python main.py --players 4 --humans 0 --difficulty easy  # AI watch mode

Other:
    python main.py --phase 1            # board generation test
    python main.py --train              # train the PPO Hard AI
    python main.py --train --steps 300000
"""

import sys
import random
import argparse
sys.path.insert(0, ".")

# Windows consoles default to cp1252 and crash on the emoji / arrow glyphs
# in game logs (UnicodeEncodeError). Force UTF-8 so console + training run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _prompt_int(msg, lo, hi):
    while True:
        try:
            v = int(input(msg).strip())
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Enter a number between {lo} and {hi}.")


def _prompt_difficulty():
    while True:
        d = input("  AI difficulty (easy/hard): ").strip().lower()
        if d in ("easy", "hard"):
            return d
        print("  Type 'easy' or 'hard'.")


def resolve_setup(args):
    """Return (n_players, n_humans, difficulty) from flags, prompting for
    anything not supplied."""
    n_players = args.players if args.players is not None else \
        _prompt_int("  Number of players (2-4): ", 2, 4)

    n_humans = args.humans if args.humans is not None else \
        _prompt_int(f"  Human players (0-{n_players}): ", 0, n_players)
    n_humans = min(n_humans, n_players)

    n_ai = n_players - n_humans
    difficulty = args.difficulty
    if n_ai > 0 and difficulty is None:
        difficulty = _prompt_difficulty()
    return n_players, n_humans, (difficulty or "easy")


def build_players(n_players, n_humans, difficulty):
    """Build the player list (humans + AI of one difficulty), then SHUFFLE
    the turn order so no seat has a fixed first-mover advantage."""
    from game.models import Player

    players = []
    for i in range(n_humans):
        name = "You" if n_humans == 1 else f"Human {i + 1}"
        players.append(Player(0, name))

    n_ai = n_players - n_humans
    label = "Hard AI" if difficulty == "hard" else "Easy AI"
    for i in range(n_ai):
        ai_name = label if n_ai == 1 else f"{label} {i + 1}"
        players.append(Player(0, ai_name, is_ai=True, ai_difficulty=difficulty))

    random.shuffle(players)                 # randomized turn order
    for idx, p in enumerate(players):       # reassign stable ids 0..N-1
        p.player_id = idx
    return players


def load_ppo_if_needed(players):
    if not any(p.ai_difficulty == "hard" for p in players):
        return None
    try:
        from ai.ppo_agent import load_ppo_model
        model = load_ppo_model()
        print("[PPO] Hard AI model loaded.")
        return model
    except Exception as e:
        print(f"[PPO] Warning: {e}")
        print("[PPO] Hard AI will use Expectimax as fallback.")
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int, default=None, choices=[1, 2])
    parser.add_argument("--players",    type=int, default=None, choices=[2, 3, 4])
    parser.add_argument("--humans",     type=int, default=None, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--difficulty", type=str, default=None, choices=["easy", "hard"])
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--steps",      type=int, default=100_000)
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--console",    action="store_true",
                        help="Play in terminal instead of the UI")
    parser.add_argument("--web",        action="store_true",
                        help="Launch the web UI (browser) instead of Pygame")
    args = parser.parse_args()

    # ── Train PPO ─────────────────────────────────────────────────
    if args.train:
        from ai.ppo_agent import train_ppo
        train_ppo(total_timesteps=args.steps)
        return

    # ── Web UI (browser handles setup + game) ─────────────────────
    if args.web:
        from ui.web.server import run_server
        run_server()
        return

    # ── Phase 1 board test ────────────────────────────────────────
    if args.phase == 1:
        from game.models import Player
        from game.board import (generate_board, print_board,
                                bfs_expected_turns)
        n = args.players or 2
        players = [Player(i, f"P{i + 1}") for i in range(n)]
        board   = generate_board(seed=args.seed, players=players)
        print_board(board)
        expected = bfs_expected_turns(board.ladders, board.snakes)
        print(f"  BFS expected turns: {expected:.1f} (need >= 10)")
        print("  Phase 1 complete!")
        return

    # ── Setup: players / humans / difficulty (prompt or flags) ────
    n_players, n_humans, difficulty = resolve_setup(args)
    players   = build_players(n_players, n_humans, difficulty)
    ppo_model = load_ppo_if_needed(players)

    order = " → ".join(p.name for p in players)
    print(f"[Setup] {n_players} players, {n_humans} human(s). "
          f"Turn order (shuffled): {order}")

    # ── Generate board ────────────────────────────────────────────
    from game.board import generate_board
    board = generate_board(seed=args.seed, players=players)

    # ── Run ───────────────────────────────────────────────────────
    if args.console:
        from game.console_game import play_game
        play_game(board, ppo_model=ppo_model)
        return

    from ui.renderer import GameRenderer
    renderer = GameRenderer()
    renderer.run(board, ppo_model=ppo_model)


if __name__ == "__main__":
    main()
