"""
Snakes & Lenders — Console Game (Phase 3 update)
Now supports Human vs AI, AI vs AI, and Human vs Human.

Run options:
    python main.py                        # Human vs Human
    python main.py --mode hvai            # Human vs Easy AI
    python main.py --mode hvai --hard     # Human vs Hard AI
    python main.py --mode aivai           # Easy AI vs Hard AI
"""

import sys
sys.path.insert(0, ".")

from game.models import Player
from game.board import generate_board, print_board
from game.engine import do_turn, calculate_snake_cost, can_place_snake
from ai.expectimax import expectimax_decision


def get_int_input(prompt: str, min_val: int, max_val: int) -> int:
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def ask_shop_phase(board, player) -> dict | None:
    """Human player shop decision."""
    if not player.can_buy_snake:
        print(f"  (You already own 3 snakes — shop unavailable)")
        return None

    print(f"\n  💰 Your points : {player.points}")
    print(f"  🐍 Snakes owned: {player.snake_count}/3")
    answer = input("  Want to buy a snake? (y/n): ").strip().lower()
    if answer != "y":
        return None

    print("\n  Place your snake:")
    print("  - Head must be HIGHER than tail")
    print("  - Head tile max: 80")

    head = get_int_input("  Enter head tile (20-80): ", 20, 80)
    tail = get_int_input("  Enter tail tile (1-79): ", 1, head - 1)

    cost = calculate_snake_cost(player, head, tail)
    valid, reason = can_place_snake(board, player, head, tail)

    if not valid:
        print(f"  ❌ Cannot place: {reason}")
        return None

    print(f"  Cost: {cost} points (you have {player.points})")
    confirm = input("  Confirm? (y/n): ").strip().lower()
    return {"head": head, "tail": tail} if confirm == "y" else None


def get_ai_decision(board, player, ppo_model=None) -> dict | None:
    """Get the shop decision from whichever AI controls this player."""
    if player.ai_difficulty == "easy":
        return expectimax_decision(board, player)
    elif player.ai_difficulty == "hard":
        if ppo_model is None:
            # Fall back to Expectimax if PPO model not loaded
            print("  [PPO] No model loaded — falling back to Expectimax")
            return expectimax_decision(board, player)
        from ai.ppo_agent import ppo_decision
        return ppo_decision(board, player, ppo_model)
    return None


def print_status(board):
    print("\n" + "─" * 45)
    for p in board.players:
        tag = f"[{'AI-E' if p.ai_difficulty == 'easy' else 'AI-H' if p.ai_difficulty == 'hard' else 'YOU'}]"
        bar = "█" * (p.position // 5)
        print(f"  {tag} {p.name:10} | Tile {p.position:3d} | "
              f"{p.points:5d} pts | Snakes: {p.snake_count}/3 | {bar}")
    print("─" * 45)


def play_game(board, ppo_model=None):
    """Run the terminal game loop on a pre-built board (any mix of human
    and AI players, set up by the caller)."""
    print("\n" + "═" * 50)
    print("   🐍  SNAKES & LENDERS  🐍")
    print("═" * 50)

    print_board(board)
    print("Game start! First to tile 100 wins.\n")

    while True:
        player = board.active_player
        print(f"\n{'═' * 50}")
        print(f"  Turn {board.turn_number} — {player.name}'s turn"
              + (" 🤖" if player.is_ai else " 👤"))
        print_status(board)

        # ── Shop phase ────────────────────────────────────────────
        if player.is_ai:
            input(f"\n  [{player.name}] Press Enter to watch AI play...")
            shop = get_ai_decision(board, player, ppo_model)
        else:
            input(f"\n  [{player.name}] Press Enter to roll the dice...")
            shop = ask_shop_phase(board, player)

        # ── Execute turn ──────────────────────────────────────────
        result = do_turn(board, shop_decision=shop)

        print()
        for line in result["logs"]:
            print(line)

        if result["winner"]:
            print_status(board)
            winner = result["winner"]
            tag = "🤖" if winner.is_ai else "🏆"
            print(f"\n{tag} {winner.name} wins the game!\n")
            break

        show = input("\n  Show board? (y/n): ").strip().lower()
        if show == "y":
            print_board(board)