"""
Snakes & Lenders — PPO Agent (Hard Mode, Autonomous Build)

The agent decides EXACTLY which snake to place (head + tail tile) directly
from the board state — no Expectimax helper functions are called at decision
time. This is a fully autonomous policy.

Action space: MultiDiscrete([72, 90])
  dim-0 → head tile index (0 = tile 19 invalid → roll only signal,
           1..71 → tiles 20..90)
  dim-1 → tail tile index (0 = roll only signal, 1..89 → tiles 1..89)

  Special case: if head_idx == 0 OR tail_idx == 0 OR the decoded
  placement is invalid (head ≤ tail, out of range, can't afford, etc.)
  the agent just rolls — no snake is bought.

  This means the agent learns BOTH "should I buy?" and "where exactly?"
  purely from reward signal, with no Expectimax guidance.

Observation space: 26-dim float32 (22 existing + 4 new board-density features)

Install dependency first:
    pip install stable-baselines3 numpy gymnasium
"""

import os
import random

import numpy as np

from game.log import gprint
from game.models import BoardState, Player
from game.engine import (calculate_snake_cost, can_place_snake,
                         MAX_SNAKE_HEAD)

# Expectimax imported ONLY for training opponents (not for placement decisions)
from ai.expectimax import expectimax_decision, strong_decision

# stable_baselines3 / gymnasium are imported lazily inside train_ppo() and
# build_training_env() — only needed during training, not at game startup.

# ── Action space encoding ─────────────────────────────────────────────────────
# Head tiles: 20..MAX_SNAKE_HEAD (90) → 71 valid values → index 1..71
#             index 0 = "roll only" signal
HEAD_MIN   = 20
HEAD_MAX   = MAX_SNAKE_HEAD          # 90
N_HEADS    = HEAD_MAX - HEAD_MIN + 1 # 71

# Tail tiles: 1..89 → 89 valid values → index 1..89
#             index 0 = "roll only" signal
TAIL_MIN   = 1
TAIL_MAX   = 89
N_TAILS    = TAIL_MAX - TAIL_MIN + 1 # 89

# MultiDiscrete dims: [head_choices, tail_choices]
# dim-0: 0 = roll, 1..71 = tiles 20..90
# dim-1: 0 = roll, 1..89 = tiles 1..89
ACT_HEAD_DIM = N_HEADS + 1   # 72  (0 = roll)
ACT_TAIL_DIM = N_TAILS + 1   # 90  (0 = roll)

OBS_DIM = 26   # 22 base + 4 board-density features

MODEL_PATH  = "ai/ppo_model.zip"
BACKUP_PATH = "ai/ppo_model_backup.zip"


def _decode_action(action) -> tuple[int | None, int | None]:
    """
    Decode a MultiDiscrete [head_idx, tail_idx] action into real tile numbers.
    Returns (None, None) if either index signals "roll only".
    """
    head_idx = int(action[0])
    tail_idx = int(action[1])
    if head_idx == 0 or tail_idx == 0:
        return None, None
    head = HEAD_MIN + (head_idx - 1)   # index 1 → tile 20, index 71 → tile 90
    tail = TAIL_MIN + (tail_idx - 1)   # index 1 → tile 1,  index 89 → tile 89
    return head, tail


# ── State Encoder ─────────────────────────────────────────────────────────────

def encode_state(board: BoardState, player: Player) -> np.ndarray:
    """
    Convert the full board state into a flat 26-dim vector the PPO net reads.

    Layout (26):
    [0]      our pos / 100
    [1]      our points / 2000 (cap)
    [2]      our snake count / 3
    [3-5]    opp positions / 100 (up to 3)
    [6-8]    opp points / 2000
    [9-11]   opp snake counts / 3
    [12]     snakes ahead of LEADING opp / 10
    [13]     distance to nearest bomb ahead / 20 (1 if none)
    [14]     distance to nearest ladder bottom ahead / 20 (1 if none)
    [15]     our points / 500 (finer resolution for affordability decisions)
    [16]     leading opp's distance to goal / 100
    [17]     turn number / 100
    [18]     in post-bankruptcy immunity (0/1)
    [19]     fraction of board tiles that are bombs / 5
    [20]     fraction of board tiles that are ladder bottoms / 7
    [21]     fraction of board tiles occupied by any snake head / 10
    [22]     leading opp pos / 100 (explicit, not derived from [3-5])
    [23]     our position relative to leader (signed) / 100
    [24]     can we buy at all (0/1)  — snake_count < 3 AND position > 0
    [25]     # valid placements we can afford right now / 20 (capped 1.0)
    """
    state = np.zeros(OBS_DIM, dtype=np.float32)
    state[0] = player.position / 100.0
    state[1] = min(player.points / 2000.0, 1.0)
    state[2] = player.snake_count / 3.0

    opponents = [p for p in board.players if p.player_id != player.player_id]
    for i, opp in enumerate(opponents[:3]):
        state[3 + i] = opp.position / 100.0
        state[6 + i] = min(opp.points / 2000.0, 1.0)
        state[9 + i] = opp.snake_count / 3.0

    leader = max(opponents, key=lambda p: p.position, default=None)
    if leader:
        snakes_ahead = sum(1 for s in board.snakes if s.head > leader.position)
        state[12] = min(snakes_ahead / 10.0, 1.0)
        state[16] = max(0.0, (100 - leader.position) / 100.0)
        state[22] = leader.position / 100.0
        state[23] = (player.position - leader.position) / 100.0

    bombs_ahead = [b for b in board.bombs if b > player.position]
    state[13] = min((min(bombs_ahead) - player.position) / 20.0, 1.0) \
                if bombs_ahead else 1.0
    ladders_ahead = [l.bottom for l in board.ladders if l.bottom > player.position]
    state[14] = min((min(ladders_ahead) - player.position) / 20.0, 1.0) \
                if ladders_ahead else 1.0

    state[15] = min(player.points / 500.0, 1.0)
    state[17] = min(board.turn_number / 100.0, 1.0)
    state[18] = 1.0 if getattr(player, "bankrupt_immune", 0) > 0 else 0.0

    # Board density features — help the agent reason about obstacle layout
    state[19] = min(len(board.bombs) / 5.0, 1.0)
    state[20] = min(len(board.ladders) / 7.0, 1.0)
    snake_heads = {s.head for s in board.snakes}
    state[21] = min(len(snake_heads) / 10.0, 1.0)

    can_buy = player.snake_count < 3 and player.position > 0
    state[24] = 1.0 if can_buy else 0.0

    # Count affordable valid placements (sampled — full search too slow per step)
    if can_buy:
        affordable = 0
        for head in range(HEAD_MIN, HEAD_MAX + 1, 3):   # stride-3 sample
            for tail in range(1, head, 5):               # stride-5 sample
                if calculate_snake_cost(player, head, tail) <= player.points:
                    ok, _ = can_place_snake(board, player, head, tail)
                    if ok:
                        affordable += 1
        state[25] = min(affordable / 20.0, 1.0)

    return state


# ── PPO Training Environment ───────────────────────────────────────────────────

def build_training_env(opponent_pool=False):
    """
    Build a Gymnasium-compatible environment for PPO training.

    Action space: MultiDiscrete([ACT_HEAD_DIM, ACT_TAIL_DIM])
    Observation:  26-dim float32 vector (see encode_state).

    The agent outputs a (head_idx, tail_idx) pair. Invalid or "roll" signals
    are masked to None (just roll). Valid placements are executed directly
    via buy_snake — NO Expectimax helpers used for placement.

    opponent_pool=True → each episode the opponent is drawn from
    {Easy, Strong heuristic, frozen current-best PPO}.
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

    def _easy(board, p):   return expectimax_decision(board, p)
    def _strong(board, p): return strong_decision(board, p)
    deciders = [_easy]
    if opponent_pool:
        deciders = [_easy, _strong]
        try:
            frozen = load_ppo_model()
            obs_ok = tuple(frozen.observation_space.shape) == (OBS_DIM,)
            act_ok = (hasattr(frozen.action_space, "nvec") and
                      list(frozen.action_space.nvec) == [ACT_HEAD_DIM, ACT_TAIL_DIM])
            if obs_ok and act_ok:
                deciders.append(lambda board, p: ppo_decision(board, p, frozen))
            else:
                print(f"[PPO] frozen model incompatible "
                      f"(obs={getattr(frozen, 'observation_space', '?')}, "
                      f"nvec={getattr(getattr(frozen,'action_space',None),'nvec','?')}) "
                      f"— skipping self-play.")
        except Exception as e:
            print(f"[PPO] no frozen model for self-play: {e}")

    class SnakesLendersEnv(gym.Env):
        """
        Autonomous PPO environment.

        Action space: MultiDiscrete([72, 90])
          dim-0: head tile index (0=roll, 1..71 → tiles 20..90)
          dim-1: tail tile index (0=roll, 1..89 → tiles 1..89)

        Invalid decoded placements (head<=tail, can't afford, placement rules)
        are silently treated as roll-only — the agent learns from reward signal
        that bad placements are wasteful.
        """

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.MultiDiscrete([ACT_HEAD_DIM, ACT_TAIL_DIM])
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
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
            self.opp_decide = random.choice(deciders)
            self.turn_count = 0
            return encode_state(self.board, self.ai_player), {}

        def step(self, action):
            agent_id       = self.ai_player.player_id
            start_pos      = self.ai_player.position
            start_bankrupts = self.ai_player.bankrupt_count
            opps           = [o for o in self.board.players
                              if o.player_id != agent_id]
            opp_pos_before = {o.player_id: o.position for o in opps}

            # ── 1. Decode action → shop decision (NO helpers) ────────
            shop_decision  = None
            agent_bought   = False
            placed_setback = 0
            placed_combo   = False

            head, tail = _decode_action(action)
            if (head is not None and tail is not None and
                    self.ai_player.can_buy_snake and
                    self.ai_player.position > 0):
                ok, _ = can_place_snake(self.board, self.ai_player, head, tail)
                if ok and calculate_snake_cost(self.ai_player, head, tail) <= self.ai_player.points:
                    shop_decision = {"head": head, "tail": tail}

            result = do_turn(self.board, shop_decision=shop_decision)
            self.turn_count += 1
            agent_bought   = bool(result.get("bought"))
            if agent_bought and shop_decision:
                placed_setback = shop_decision["head"] - shop_decision["tail"]
                placed_combo   = shop_decision["tail"] in self.board.bombs
            winner = result["winner"]

            # ── 2. Opponents' turns ───────────────────────────────────
            while winner is None and self.board.active_player.player_id != agent_id:
                opp        = self.board.active_player
                opp_dec    = self.opp_decide(self.board, opp)
                opp_result = do_turn(self.board, shop_decision=opp_dec)
                self.turn_count += 1
                winner = opp_result["winner"]

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
                            went_bankrupt, opp_setback,
                            placed_setback, placed_combo) -> float:
            if winner is not None:
                return 100.0 if winner.player_id == self.ai_player.player_id else -100.0

            reward = 0.0
            reward += (self.ai_player.position - start_pos) * 0.1
            reward += opp_setback * 0.5
            if went_bankrupt:
                reward -= 50.0
            if agent_bought:
                reward += 4.0 + placed_setback * 0.3
                if placed_combo:
                    reward += 12.0
            else:
                if self.ai_player.points > 100:
                    reward -= 1.0
            return reward

    return SnakesLendersEnv


# ── Training Function ─────────────────────────────────────────────────────────

def train_ppo(total_timesteps: int = 100_000, opponent_pool: bool = False,
              save_path: str = MODEL_PATH):
    """
    Train the autonomous PPO agent and save the model.

    The agent uses MultiDiscrete action space — it picks exact head and tail
    tile indices directly. No Expectimax helpers are used in the policy.

    For this larger action space, more steps are needed than the
    strategy-picker build. Recommended: ≥5M steps.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        raise ImportError(
            "PPO training requires: pip install stable-baselines3 gymnasium"
        )

    print(f"[PPO-AUTO] Training {total_timesteps:,} steps "
          f"(opponent_pool={opponent_pool}) → {save_path}")
    print(f"[PPO-AUTO] Action space: MultiDiscrete([{ACT_HEAD_DIM}, {ACT_TAIL_DIM}])"
          f" | Obs: {OBS_DIM}-dim")

    EnvClass = build_training_env(opponent_pool=opponent_pool)
    vec_env  = make_vec_env(EnvClass, n_envs=4)

    model = None
    if os.path.exists(save_path):
        try:
            from stable_baselines3 import PPO as _PPO
            candidate = _PPO.load(save_path, env=vec_env, device="cpu")
            obs_ok = tuple(candidate.observation_space.shape) == (OBS_DIM,)
            act_ok = (hasattr(candidate.action_space, "nvec") and
                      list(candidate.action_space.nvec) == [ACT_HEAD_DIM, ACT_TAIL_DIM])
            if obs_ok and act_ok:
                model = candidate
                print(f"[PPO-AUTO] Continuing from {save_path} "
                      f"({model.num_timesteps:,} prior steps).")
            else:
                print(f"[PPO-AUTO] {save_path} shape mismatch — starting fresh.")
        except Exception as e:
            print(f"[PPO-AUTO] cannot load {save_path}: {e} — starting fresh.")

    if model is None:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.05,    # higher entropy — larger action space needs more exploration
            device="cpu",
            verbose=1,
        )

    from game import log
    log.VERBOSE = False
    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=(model.num_timesteps == 0),
        )
    finally:
        log.VERBOSE = True
    model.save(save_path)
    print(f"[PPO-AUTO] Training complete! Model saved to {save_path} "
          f"({model.num_timesteps:,} total steps).")


# ── Inference (Playing) ───────────────────────────────────────────────────────

def load_ppo_model():
    """Load the trained PPO model (main → backup fallback)."""
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
            except Exception as e:
                last_err = e

    raise FileNotFoundError(
        f"No loadable PPO model at {MODEL_PATH} or {BACKUP_PATH}"
        + (f" (last error: {last_err})" if last_err else "")
        + ".\nTrain first: python -c \"from ai.ppo_agent import train_ppo; train_ppo(5000000, True)\""
    )


def ppo_decision(board: BoardState, player: Player, model) -> dict | None:
    """
    Autonomous PPO inference — the policy outputs exact (head, tail) tile
    indices directly. No Expectimax helper functions are called.

    Invalid decoded placements are silently treated as roll-only.
    """
    if not player.can_buy_snake:
        return None

    obs    = encode_state(board, player)
    action, _ = model.predict(obs, deterministic=False)

    head, tail = _decode_action(action)
    if head is None or tail is None:
        return None

    ok, _ = can_place_snake(board, player, head, tail)
    if not ok:
        return None
    if calculate_snake_cost(player, head, tail) > player.points:
        return None

    shop = {"head": head, "tail": tail}
    gprint(f"  [{player.name}] (PPO-AUTO) snake {head}→{tail}")
    return shop
