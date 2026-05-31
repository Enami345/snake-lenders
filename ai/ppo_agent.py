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
                           propose_big_snake, propose_lurk, propose_combo)

# PPO action space — the policy picks a STRATEGY each turn:
#   0 = roll only (bank points / be patient)
#   1 = cheap short trap just ahead of the leader (cheap pressure)
#   2 = save up & drop the longest affordable snake (max knockback)
#   3 = win-denial lurk snake near the goal (tiles 85-90)
#   4 = COMBO: snake whose tail is a bomb tile (knockback + bomb damage)
N_ACTIONS = 5


def _action_to_shop(board, player, action):
    """Map a PPO action to a shop decision (or None to just roll)."""
    if action == 1:
        return propose_cheap_trap(board, player)
    if action == 2:
        return propose_big_snake(board, player)
    if action == 3:
        return propose_lurk(board, player)
    if action == 4:
        return propose_combo(board, player)
    return None

# Path where the trained model is saved/loaded. BACKUP_PATH is a stable
# copy used as a fallback when the main file is missing or mid-write
# (e.g. while a training run is overwriting it).
MODEL_PATH  = "ai/ppo_model.zip"
BACKUP_PATH = "ai/ppo_model_backup.zip"


# ── State Encoder ─────────────────────────────────────────────────────────────

def encode_state(board: BoardState, player: Player) -> np.ndarray:
    """
    Convert the full board state into a flat 22-dim vector the PPO net reads.

    Layout (22 values):
    [0]      our position / 100
    [1]      our points / 2000 (cap)
    [2]      our snake count / 3
    [3-5]    opponent positions / 100 (up to 3)
    [6-8]    opponent points / 2000
    [9-11]   opponent snake counts / 3
    [12]     snakes ahead of the LEADING opponent / 10
    [13]     nearest bomb ahead of us, distance / 20 (1 if none)
    [14]     nearest ladder bottom ahead of us, distance / 20 (1 if none)
    [15]     best available snake cost / 1000
    [16]     best available snake damage / 100 (clamped -1..1)
    [17]     can we afford the best snake? (0/1)
    [18]     is a COMBO snake (tail on a bomb) available? (0/1)
    [19]     are we in post-bankruptcy immunity? (0/1)
    [20]     leading opponent's distance to goal / 100
    [21]     turn / 100
    """
    state = np.zeros(22, dtype=np.float32)
    state[0] = player.position / 100.0
    state[1] = min(player.points / 2000.0, 1.0)
    state[2] = player.snake_count / 3.0

    opponents = [p for p in board.players if p.player_id != player.player_id]
    for i, opp in enumerate(opponents[:3]):
        state[3 + i] = opp.position / 100.0
        state[6 + i] = min(opp.points / 2000.0, 1.0)
        state[9 + i] = opp.snake_count / 3.0

    if opponents:
        leader = max(opponents, key=lambda p: p.position)
        snakes_ahead = sum(1 for s in board.snakes if s.head > leader.position)
        state[12] = min(snakes_ahead / 10.0, 1.0)
        state[20] = max(0.0, (100 - leader.position) / 100.0)

    bombs_ahead = [b for b in board.bombs if b > player.position]
    state[13] = min((min(bombs_ahead) - player.position) / 20.0, 1.0) if bombs_ahead else 1.0
    ladders_ahead = [l.bottom for l in board.ladders if l.bottom > player.position]
    state[14] = min((min(ladders_ahead) - player.position) / 20.0, 1.0) if ladders_ahead else 1.0

    best = find_best_snake_to_buy(board, player)
    if best:
        head, tail, dmg = best
        cost = calculate_snake_cost(player, head, tail)
        state[15] = min(cost / 1000.0, 1.0)
        state[16] = min(max(dmg / 100.0, -1.0), 1.0)
        state[17] = 1.0 if player.points >= cost else 0.0

    state[18] = 1.0 if propose_combo(board, player) else 0.0
    state[19] = 1.0 if getattr(player, "bankrupt_immune", 0) > 0 else 0.0
    state[21] = min(board.turn_number / 100.0, 1.0)
    return state


def _obs_for_model(obs: np.ndarray, model) -> np.ndarray:
    """Adapt an observation to the size expected by a loaded PPO model.

    This keeps older saved models usable when the live encoder grows new
    trailing features.
    """
    expected_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not expected_shape:
        return obs

    expected_size = int(np.prod(expected_shape))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.size == expected_size:
        return obs
    if obs.size > expected_size:
        return obs[:expected_size]

    padded = np.zeros(expected_size, dtype=np.float32)
    padded[:obs.size] = obs
    return padded


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
                low=0.0, high=1.0, shape=(22,), dtype=np.float32
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
            placed_setback = (shop_decision["head"] - shop_decision["tail"]
                              if (agent_bought and shop_decision) else 0)
            # Combo = bought snake whose tail sits on a bomb tile.
            placed_combo = bool(
                agent_bought and shop_decision
                and shop_decision["tail"] in self.board.bombs)
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
                winner, agent_bought, start_pos, went_bankrupt,
                opp_setback, placed_setback, placed_combo)

            obs = encode_state(self.board, self.ai_player)
            return obs, reward, terminated, truncated, {}

        def _compute_reward(self, winner, agent_bought, start_pos,
                            went_bankrupt, opp_setback, placed_setback,
                            placed_combo) -> float:
            """
            Win/loss dominates, but we deliberately shape toward CUNNING,
            aggressive play so the policy actually deploys traps (the
            exact-head rule otherwise makes passivity near-optimal). We reward
            setting opponents back, reward placing snakes (scaled by their
            knockback), and give a big bonus for COMBOs (snake tail on a
            bomb tile). Anti-hoard: if we sat on plenty of points and didn't
            even try, take a small penalty.
            """
            if winner is not None:
                return 100.0 if winner.player_id == self.ai_player.player_id else -100.0

            reward = 0.0
            reward += (self.ai_player.position - start_pos) * 0.1   # progress
            reward += opp_setback * 0.4                             # sabotage paying off (heavier)
            if went_bankrupt:
                reward -= 50.0
            if agent_bought:
                reward += 4.0 + placed_setback * 0.25               # bigger trap = more
                if placed_combo:                                    # tail-on-bomb COMBO
                    reward += 8.0
            else:
                # Anti-hoard: rich and didn't even try → small penalty so the
                # policy doesn't just sit on a pile of points all game.
                if self.ai_player.points > 100:
                    reward -= 0.5
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
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.02,          # entropy bonus → keeps the policy varied (unpredictable)
        device="cpu",          # MLP PPO + Python-heavy env is typically faster on CPU.
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
    obs = _obs_for_model(encode_state(board, player), model)
    # Stochastic sampling at inference → unpredictable cunning play
    # (varies between saving, cheap pressure, big knockbacks, lurks, combos).
    action, _ = model.predict(obs, deterministic=False)

    shop = _action_to_shop(board, player, int(action))
    if shop:
        from game.log import gprint
        gprint(f"  [{player.name}] (PPO) snake "
               f"{shop['head']}→{shop['tail']}")
    return shop