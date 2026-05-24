"""
Snakes & Lenders — PPO Agent (Hard Mode)
Uses Proximal Policy Optimization via stable-baselines3.
The agent learns by playing thousands of self-play games.

Install dependency first:
    pip install stable-baselines3 numpy gymnasium
"""

import os
import numpy as np
from game.models import BoardState, Player
from game.engine import calculate_snake_cost, can_place_snake
from ai.expectimax import find_best_snake_to_buy

# Path where the trained model is saved/loaded
MODEL_PATH = "ai/ppo_model.zip"


# ── State Encoder ─────────────────────────────────────────────────────────────

def encode_state(board: BoardState, player: Player) -> np.ndarray:
    """
    Convert the full board state into a flat numeric vector
    that the PPO neural network can read.

    Vector layout (total: 14 values):
    [0]     Our position (normalized 0-1)
    [1]     Our points (normalized, capped at 2000)
    [2]     Our snake count (0-3)
    [3-5]   Opponent positions (up to 3 opponents, normalized)
    [6-8]   Opponent points (normalized)
    [9]     Number of snakes ahead of nearest opponent
    [10]    Cost of best available snake (normalized)
    [11]    ROI of best available snake (normalized)
    [12]    Turns played (normalized, capped at 100)
    [13]    Can we afford a snake? (0 or 1)
    """
    state = np.zeros(14, dtype=np.float32)

    # Our info
    state[0] = player.position / 100.0
    state[1] = min(player.points / 2000.0, 1.0)
    state[2] = player.snake_count / 3.0

    # Opponent info (up to 3 opponents)
    opponents = [p for p in board.players if p.player_id != player.player_id]
    for i, opp in enumerate(opponents[:3]):
        state[3 + i] = opp.position / 100.0
        state[6 + i] = min(opp.points / 2000.0, 1.0)

    # Snakes ahead of nearest opponent
    if opponents:
        nearest_opp = min(opponents, key=lambda p: abs(p.position - player.position))
        snakes_ahead = sum(
            1 for s in board.snakes
            if s.head > nearest_opp.position
        )
        state[9] = min(snakes_ahead / 10.0, 1.0)

    # Best snake info
    best = find_best_snake_to_buy(board, player)
    if best:
        head, tail, roi = best
        cost = calculate_snake_cost(player, head, tail)
        state[10] = min(cost / 1000.0, 1.0)
        state[11] = min(max(roi / 100.0, -1.0), 1.0)
        state[13] = 1.0 if player.points >= cost else 0.0

    # Turn number
    state[12] = min(board.turn_number / 100.0, 1.0)

    return state


# ── PPO Training Environment ───────────────────────────────────────────────────

def build_training_env():
    """
    Build a Gymnasium-compatible environment for PPO training.
    Only imported when training — not needed just to play.
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from game.models import Player
        from game.board import generate_board
        from game.engine import do_turn
    except ImportError:
        raise ImportError(
            "Training requires: pip install stable-baselines3 numpy gymnasium"
        )

    class SnakesLendersEnv(gym.Env):
        """
        Custom Gym environment for Snakes & Lenders.

        Action space:
            0 = Just roll (no snake purchase)
            1 = Buy the best available snake

        Observation space:
            14-dimensional float vector (see encode_state)
        """

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(14,), dtype=np.float32
            )
            self.board = None
            self.ai_player = None
            self.max_turns = 200

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            players = [
                Player(0, "PPO_Agent", is_ai=True, ai_difficulty="hard"),
                Player(1, "Opponent",  is_ai=True, ai_difficulty="easy"),
            ]
            self.board = generate_board(players=players)
            self.ai_player = self.board.players[0]
            self.turn_count = 0
            return encode_state(self.board, self.ai_player), {}

        def step(self, action):
            # Action 0 = roll only, Action 1 = try to buy best snake then roll
            shop_decision = None
            if action == 1:
                best = find_best_snake_to_buy(self.board, self.ai_player)
                if best:
                    head, tail, _ = best
                    shop_decision = {"head": head, "tail": tail}

            result = do_turn(self.board, shop_decision=shop_decision)
            self.turn_count += 1

            # ── Reward function ──────────────────────────────────────
            reward = 0.0

            if result["winner"]:
                if result["winner"].player_id == self.ai_player.player_id:
                    reward = +100.0   # Big reward for winning
                else:
                    reward = -100.0   # Big penalty for losing

            else:
                # Small reward for making progress toward tile 100
                reward += self.ai_player.position * 0.1

                # Penalty for going bankrupt
                if self.ai_player.points < 0:
                    reward -= 50.0

                # Small reward for successfully placing a snake
                if result.get("bought"):
                    reward += 10.0

            terminated = result["winner"] is not None
            truncated  = self.turn_count >= self.max_turns

            obs = encode_state(self.board, self.ai_player)
            return obs, reward, terminated, truncated, {}

    return SnakesLendersEnv


# ── Training Function ─────────────────────────────────────────────────────────

def train_ppo(total_timesteps: int = 100_000):
    """
    Train the PPO agent via self-play and save the model.

    Args:
        total_timesteps: How many environment steps to train for.
                         100,000 = ~30 min. 500,000 = better quality.

    Run this once before playing on Hard mode:
        python -c "from ai.ppo_agent import train_ppo; train_ppo()"
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        raise ImportError(
            "PPO training requires: pip install stable-baselines3 gymnasium"
        )

    print(f"[PPO] Starting training for {total_timesteps:,} timesteps...")
    print("[PPO] This may take several minutes. Go grab a coffee ☕")

    EnvClass = build_training_env()

    # Use 4 parallel environments to speed up training
    vec_env = make_vec_env(EnvClass, n_envs=4)

    model = PPO(
        policy="MlpPolicy",     # Multi-layer perceptron neural network
        env=vec_env,
        learning_rate=3e-4,     # Standard PPO learning rate
        n_steps=2048,           # Steps before each policy update
        batch_size=64,
        n_epochs=10,            # PPO epochs per update
        gamma=0.99,             # Discount factor (values future rewards)
        clip_range=0.2,         # The PPO clipping value — keeps updates stable
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_PATH)
    print(f"[PPO] Training complete! Model saved to {MODEL_PATH}")


# ── Inference (Playing) ───────────────────────────────────────────────────────

def load_ppo_model():
    """Load the trained PPO model from disk."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("pip install stable-baselines3")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            f"Train first with: python -c \"from ai.ppo_agent import train_ppo; train_ppo()\""
        )

    return PPO.load(MODEL_PATH)


def ppo_decision(board: BoardState, player: Player, model) -> dict | None:
    """
    Main entry point for the PPO AI during gameplay.
    Uses the trained model to decide whether to buy a snake.

    Args:
        board:  Current board state
        player: The PPO player
        model:  Loaded PPO model (from load_ppo_model())

    Returns:
        dict with 'head' and 'tail' if buying, or None to just roll.
    """
    obs = encode_state(board, player)
    action, _ = model.predict(obs, deterministic=True)

    if action == 1:
        best = find_best_snake_to_buy(board, player)
        if best:
            head, tail, _ = best
            cost = calculate_snake_cost(player, head, tail)
            if player.points >= cost:
                print(f"  [PPO] Buying snake {head}→{tail} (cost: {cost})")
                return {"head": head, "tail": tail}

    return None