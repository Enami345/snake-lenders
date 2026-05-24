"""
Snakes & Lenders — Main Entry Point

python main.py                         # Human vs Human (Pygame UI)
python main.py --mode hvai             # Human vs Easy AI
python main.py --mode hvai --hard      # Human vs Hard AI
python main.py --mode aivai            # Easy AI vs Hard AI
python main.py --console               # Play in terminal instead
python main.py --phase 1               # Phase 1 board test
python main.py --train                 # Train the PPO model
"""

import sys
import argparse
sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",   type=int,  default=None,
                        choices=[1, 2])
    parser.add_argument("--mode",    type=str,  default="hvh",
                        choices=["hvh", "hvai", "aivai"])
    parser.add_argument("--hard",    action="store_true")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--steps",   type=int,  default=100_000)
    parser.add_argument("--seed",    type=int,  default=None)
    parser.add_argument("--players", type=int,  default=2,
                        choices=[2, 3, 4])
    parser.add_argument("--console", action="store_true",
                        help="Play in terminal instead of Pygame UI")
    args = parser.parse_args()

    # ── Train PPO ─────────────────────────────────────────────────
    if args.train:
        from ai.ppo_agent import train_ppo
        train_ppo(total_timesteps=args.steps)
        return

    # ── Phase 1 test ──────────────────────────────────────────────
    if args.phase == 1:
        from game.models import Player
        from game.board import (generate_board, print_board,
                                bfs_expected_turns)
        names   = ["EL", "JC", "KINA", "MATTERS"]
        players = [Player(i, names[i]) for i in range(args.players)]
        board   = generate_board(seed=args.seed, players=players)
        print_board(board)
        expected = bfs_expected_turns(board.ladders, board.snakes)
        print(f"  BFS expected turns: {expected:.1f} (need >= 10)")
        print("  Phase 1 complete!")
        return

    # ── Build players ─────────────────────────────────────────────
    from game.models import Player

    if args.mode == "hvh":
        players = [
            Player(0, "EL"),
            Player(1, "JC"),
        ]
    elif args.mode == "hvai":
        diff  = "hard" if args.hard else "easy"
        label = "Hard AI" if args.hard else "Easy AI"
        players = [
            Player(0, "You"),
            Player(1, label, is_ai=True, ai_difficulty=diff),
        ]
    elif args.mode == "aivai":
        players = [
            Player(0, "Easy AI", is_ai=True, ai_difficulty="easy"),
            Player(1, "Hard AI", is_ai=True, ai_difficulty="hard"),
        ]

    # ── Load PPO if needed ────────────────────────────────────────
    ppo_model = None
    if any(p.ai_difficulty == "hard" for p in players):
        try:
            from ai.ppo_agent import load_ppo_model
            ppo_model = load_ppo_model()
            print("[PPO] Hard AI model loaded.")
        except FileNotFoundError as e:
            print(f"[PPO] Warning: {e}")
            print("[PPO] Hard AI will use Expectimax as fallback.")

    # ── Generate board ────────────────────────────────────────────
    from game.board import generate_board
    board = generate_board(seed=args.seed, players=players)

    # ── Console mode ──────────────────────────────────────────────
    if args.console:
        from game.console_game import play_game
        play_game(mode=args.mode,
                  use_hard_ai=args.hard,
                  seed=args.seed)
        return

    # ── Pygame UI (default) ───────────────────────────────────────
    from ui.renderer import GameRenderer
    renderer = GameRenderer()
    renderer.run(board, mode=args.mode, ppo_model=ppo_model)


if __name__ == "__main__":
    main()