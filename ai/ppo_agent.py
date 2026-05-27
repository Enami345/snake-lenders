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
import random

from ai.expectimax import (find_best_snake_to_buy, expectimax_decision,
                           strong_decision, propose_cheap_trap,
                           propose_big_snake, propose_lurk)

# PPO action space — the policy picks a STRATEGY each turn (this is what
# gives Hard a real edge over Easy's single fixed heuristic):
#   0 = roll only (bank points)
#   1 = cheap short trap just ahead of the leader
#   2 = save up & drop the longest affordable snake (max knockback)
#   3 = win-denial lurk snake near the goal (tiles 85-90)
N_ACTIONS = 4


def _action_to_shop(board, player, action):
    """Map a PPO action to a shop decision (or None to just roll)."""
    if action == 1:
        return propose_cheap_trap(board, player)
    if action == 2:
        return propose_big_snake(board, player)
    if action == 3:
        return propose_lurk(board, player)
    return None

# Path where the trained model is saved/loaded. BACKUP_PATH is a stable
# copy used as a fallback when the main file is missing or mid-write
# (e.g. while a training run is overwriting it).
MODEL_PATH  = "ai/ppo_model.zip"
BACKUP_PATH = "ai/ppo_model_backup.zip"


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

def build_training_env(opponent_pool=False):
    """
    Build a Gymnasium-compatible environment for PPO training.

    opponent_pool=False → opponent is the weak Easy bot (original).
    opponent_pool=True  → each episode the opponent is drawn from a POOL
        {Easy, Strong heuristic, frozen current-best PPO} so the agent
        learns against varied/stronger play (self-play), not one weak bot.
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

    # Build the opponent decider pool once.
    def _easy(board, p):   return expectimax_decision(board, p)
    def _strong(board, p): return strong_decision(board, p)
    deciders = [_easy]
    if opponent_pool:
        deciders = [_easy, _strong]
        try:                                   # add frozen current best (self-play)
            frozen = load_ppo_model()
            deciders.append(lambda board, p: ppo_decision(board, p, frozen))
        except Exception:
            pass

    class SnakesLendersEnv(gym.Env):
        """
        Custom Gym environment for Snakes & Lenders.

        Action space (Discrete(4)): roll / cheap trap / big snake / lurk.
        The policy chooses the strategy itself — see _action_to_shop.

        Observation space:
            14-dimensional float vector (see encode_state)
        """

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(N_ACTIONS)
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
            self.opp_decide = random.choice(deciders)   # this episode's opponent
            self.turn_count = 0
            return encode_state(self.board, self.ai_player), {}

        def step(self, action):
            """
            One agent decision = one full round:
              1. Apply the policy's action on the AGENT's turn (roll, or
                 buy-best-snake then roll).
              2. Then let every OPPONENT take their own turn using their
                 own Expectimax policy, until it is the agent's turn again
                 (or the game ends).
            Reward/observation are always from the agent's perspective.
            """
            agent_id        = self.ai_player.player_id
            start_pos       = self.ai_player.position
            start_bankrupts = self.ai_player.bankrupt_count
            opps            = [o for o in self.board.players
                               if o.player_id != agent_id]
            opp_pos_before  = {o.player_id: o.position for o in opps}

            # ── 1. Agent's turn (policy picks the strategy) ──────────
            shop_decision = _action_to_shop(self.board, self.ai_player, action)
            result = do_turn(self.board, shop_decision=shop_decision)
            self.turn_count += 1
            agent_bought = bool(result.get("bought"))
            winner       = result["winner"]

            # ── 2. Opponents' turns (their own Expectimax) ───────────
            while winner is None and self.board.active_player.player_id != agent_id:
                opp          = self.board.active_player
                opp_decision = self.opp_decide(self.board, opp)
                opp_result   = do_turn(self.board, shop_decision=opp_decision)
                self.turn_count += 1
                winner = opp_result["winner"]

            # How much ground opponents lost this round (our sabotage paying
            # off — also catches them hitting board snakes, close enough).
            opp_setback = sum(max(0, opp_pos_before[o.player_id] - o.position)
                              for o in opps)

            terminated = winner is not None
            truncated  = self.turn_count >= self.max_turns

            went_bankrupt = self.ai_player.bankrupt_count > start_bankrupts
            reward = self._compute_reward(
                winner, agent_bought, start_pos, went_bankrupt, opp_setback)

            obs = encode_state(self.board, self.ai_player)
            return obs, reward, terminated, truncated, {}

        def _compute_reward(self, winner, agent_bought, start_pos,
                            went_bankrupt, opp_setback) -> float:
            """
            Win/loss dominates. Shaping is small and bounded: progress delta
            (not absolute position) plus a reward for setting opponents back,
            so the policy learns that effective sabotage is worth the spend.
            """
            if winner is not None:
                return 100.0 if winner.player_id == self.ai_player.player_id else -100.0

            reward = 0.0
            # Progress delta toward tile 100 (negative if snake-bitten back)
            reward += (self.ai_player.position - start_pos) * 0.1
            # Reward setting opponents back (sabotage paying off)
            reward += opp_setback * 0.15
            # Real bankruptcy this round (bomb wiped the wallet → reset)
            if went_bankrupt:
                reward -= 50.0
            # Tiny nudge for actually placing a trap
            if agent_bought:
                reward += 2.0
            return reward

    return SnakesLendersEnv


# ── Training Function ─────────────────────────────────────────────────────────

def train_ppo(total_timesteps: int = 100_000, opponent_pool: bool = False,
              save_path: str = MODEL_PATH):
    """
    Train the PPO agent and save the model.

    Args:
        total_timesteps: environment steps to train for.
        opponent_pool:   if True, train against {Easy, Strong, frozen best
                         PPO} instead of only the weak Easy bot.
        save_path:       where to write the model (default ai/ppo_model.zip).
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        raise ImportError(
            "PPO training requires: pip install stable-baselines3 gymnasium"
        )

    print(f"[PPO] Training {total_timesteps:,} steps "
          f"(opponent_pool={opponent_pool}) → {save_path}")

    EnvClass = build_training_env(opponent_pool=opponent_pool)

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

    # Silence the per-step game logs (board gen, AI buys, bankruptcies)
    # during training so only SB3's progress tables show.
    from game import log
    log.VERBOSE = False
    try:
        model.learn(total_timesteps=total_timesteps)
    finally:
        log.VERBOSE = True
    model.save(save_path)
    print(f"[PPO] Training complete! Model saved to {save_path}")


# ── Inference (Playing) ───────────────────────────────────────────────────────

def load_ppo_model():
    """Load the trained PPO model. Tries the main file first, then the
    backup — so the game keeps working while a training run is busy
    overwriting (or has corrupted mid-write) the main file."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("pip install stable-baselines3")

    last_err = None
    for path in (MODEL_PATH, BACKUP_PATH):
        if os.path.exists(path):
            try:
                model = PPO.load(path)
                if path == BACKUP_PATH:
                    print(f"[PPO] Loaded backup model ({path}).")
                return model
            except Exception as e:           # mid-write / corrupt → try next
                last_err = e

    raise FileNotFoundError(
        f"No loadable PPO model at {MODEL_PATH} or {BACKUP_PATH}"
        + (f" (last error: {last_err})" if last_err else "")
        + ".\nTrain first with: python main.py --train"
    )


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
    if not player.can_buy_snake:
        return None
    obs = encode_state(board, player)
    action, _ = model.predict(obs, deterministic=True)

    shop = _action_to_shop(board, player, int(action))
    if shop:
        from game.log import gprint
        gprint(f"  [{player.name}] (PPO) snake "
               f"{shop['head']}→{shop['tail']}")
    return shop