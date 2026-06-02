"""
Snakes & Lenders — PPO Agent (Hard Mode)

Fully autonomous placement policy — PPO picks a strategy index (0-4)
and the agent resolves an exact (head, tail) tile pair by scanning the
board itself using only engine rules. No Expectimax placement helpers
are called during inference or training.

Action space : Discrete(5)  — compatible with the shipped trained model
  0 = roll only
  1 = place the shortest affordable snake on the leading opponent
  2 = place the longest affordable snake on the leading opponent
  3 = place a snake near tile 90 (win-denial)
  4 = place a snake whose tail is a bomb tile (combo)

All placement searches are done inline with can_place_snake /
calculate_snake_cost (engine rules) — zero Expectimax calls.

Observation space: 22-dim float32 — matches the shipped 5M trained model.

Install dependency:
    pip install stable-baselines3 numpy gymnasium
"""

import os
import random

import numpy as np

from game.log import gprint
from game.models import BoardState, Player
from game.engine import (calculate_snake_cost, can_place_snake,
                         MAX_SNAKE_HEAD)

# Expectimax used ONLY as training opponents — never for placement decisions.
from ai.expectimax import expectimax_decision, strong_decision

# SB3 / gymnasium imported lazily — only needed during training.

N_ACTIONS   = 5
MODEL_PATH  = "ai/ppo_model.zip"
BACKUP_PATH = "ai/ppo_model_backup.zip"

HEAD_MIN = 20
HEAD_MAX = MAX_SNAKE_HEAD   # 90


# ── Autonomous placement helpers (no Expectimax) ──────────────────────────────

def _leading_opponent(board: BoardState, player: Player):
    opps = [p for p in board.players if p.player_id != player.player_id]
    return max(opps, key=lambda o: o.position, default=None)


def _can_afford(player, head, tail):
    return calculate_snake_cost(player, head, tail) <= player.points


def _try_place(board, player, head, tail):
    """Return {head,tail} if placement valid + affordable, else None."""
    if head <= tail or head > HEAD_MAX or tail < 1:
        return None
    ok, _ = can_place_snake(board, player, head, tail)
    if ok and _can_afford(player, head, tail):
        return {"head": head, "tail": tail}
    return None


def _action_to_shop(board: BoardState, player: Player, action: int):
    """
    Resolve a strategy index into an exact (head, tail) placement using
    only engine rules. Returns {head, tail} or None (roll only).
    No Expectimax helper functions are called.
    """
    if action == 0:
        return None   # roll only

    opp = _leading_opponent(board, player)
    if opp is None or not player.can_buy_snake or player.position < 1:
        return None

    # action 1 — SHORT pressure trap: smallest affordable snake in opp dice range
    if action == 1:
        for offset in range(1, 7):
            head = opp.position + offset
            if head > HEAD_MAX:
                continue
            for length in (5, 6, 8, 10):
                shop = _try_place(board, player, head, head - length)
                if shop:
                    return shop
        return None

    # action 2 — BIG knockback: longest affordable snake in opp dice range
    if action == 2:
        best = None
        best_len = 0
        for offset in range(1, 7):
            head = opp.position + offset
            if head > HEAD_MAX:
                continue
            for tail in (1, head - 40, head - 30, head - 20, head - 10):
                shop = _try_place(board, player, head, tail)
                if shop and (head - tail) > best_len:
                    best_len = head - tail
                    best = shop
        return best

    # action 3 — WIN-DENIAL lurk: snake near tile 90, max setback
    if action == 3:
        if opp.position < 60:
            return None
        best = None
        best_len = 0
        for offset in range(1, 7):
            head = min(opp.position + offset, HEAD_MAX)
            if head <= opp.position:
                continue
            for tail in (1, head - 40, head - 30, head - 20):
                shop = _try_place(board, player, head, tail)
                if shop and (head - tail) > best_len:
                    best_len = head - tail
                    best = shop
        return best

    # action 4 — COMBO: snake whose tail is a bomb tile
    if action == 4:
        if not board.bombs:
            return None
        best = None
        best_len = 0
        for tail in board.bombs:
            for offset in range(1, 7):
                head = opp.position + offset
                if head > HEAD_MAX or head <= tail:
                    continue
                shop = _try_place(board, player, head, tail)
                if shop and (head - tail) > best_len:
                    best_len = head - tail
                    best = shop
        return best

    return None


# ── State Encoder ─────────────────────────────────────────────────────────────

def encode_state(board: BoardState, player: Player) -> np.ndarray:
    """
    22-dim observation — matches the shipped trained model weights.

    [0]   our pos / 100
    [1]   our points / 2000
    [2]   our snake count / 3
    [3-5] opp positions / 100
    [6-8] opp points / 2000
    [9-11] opp snake counts / 3
    [12]  snakes ahead of leader / 10
    [13]  dist to nearest bomb ahead / 20
    [14]  dist to nearest ladder ahead / 20
    [15]  best snake cost / 1000
    [16]  best snake damage / 100
    [17]  can afford best snake (0/1)
    [18]  combo available (0/1)
    [19]  bankruptcy immunity (0/1)
    [20]  leader dist to goal / 100
    [21]  turn / 100
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
    state[13] = min((min(bombs_ahead) - player.position) / 20.0, 1.0) \
                if bombs_ahead else 1.0
    ladders_ahead = [l.bottom for l in board.ladders if l.bottom > player.position]
    state[14] = min((min(ladders_ahead) - player.position) / 20.0, 1.0) \
                if ladders_ahead else 1.0

    # Best-snake features — computed directly (no Expectimax call)
    best_cost = best_dmg = 0.0
    can_afford_flag = 0.0
    opp = _leading_opponent(board, player)
    if opp and player.can_buy_snake:
        for offset in range(1, 7):
            head = opp.position + offset
            if head > HEAD_MAX:
                continue
            for tail in (1, head - 20, head - 10, head - 5):
                if tail < 1 or tail >= head:
                    continue
                ok, _ = can_place_snake(board, player, head, tail)
                if not ok:
                    continue
                cost = calculate_snake_cost(player, head, tail)
                dmg  = (head - tail) * 10.0 * (1.0 + opp.position / 100.0)
                if dmg > best_dmg:
                    best_dmg  = dmg
                    best_cost = cost
                    can_afford_flag = 1.0 if player.points >= cost else 0.0

    state[15] = min(best_cost / 1000.0, 1.0)
    state[16] = min(best_dmg / 100.0, 1.0)
    state[17] = can_afford_flag

    # Combo available — check if any bomb tile can be a valid tail
    combo = 0.0
    if opp and player.can_buy_snake and board.bombs:
        for tail in board.bombs:
            for offset in range(1, 7):
                head = opp.position + offset
                if _try_place(board, player, head, tail):
                    combo = 1.0
                    break
            if combo:
                break
    state[18] = combo
    state[19] = 1.0 if getattr(player, "bankrupt_immune", 0) > 0 else 0.0
    state[21] = min(board.turn_number / 100.0, 1.0)
    return state


# ── PPO Training Environment ───────────────────────────────────────────────────

def build_training_env(opponent_pool=False):
    """
    Gymnasium env for training. Action space Discrete(5) matches the shipped
    model. Placement is resolved by _action_to_shop using engine rules only.
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
            obs_ok = tuple(frozen.observation_space.shape) == (22,)
            act_ok = int(getattr(frozen.action_space, "n", -1)) == N_ACTIONS
            if obs_ok and act_ok:
                deciders.append(lambda board, p: ppo_decision(board, p, frozen))
            else:
                print(f"[PPO] frozen model incompatible — skipping self-play.")
        except Exception as e:
            print(f"[PPO] no frozen model for self-play: {e}")

    class SnakesLendersEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space      = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(22,), dtype=np.float32)
            self.board     = None
            self.ai_player = None
            self.max_turns = 200

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            players = [
                Player(0, "PPO_Agent", is_ai=True, ai_difficulty="hard"),
                Player(1, "Opponent",  is_ai=True, ai_difficulty="easy"),
            ]
            self.board      = generate_board(players=players)
            self.ai_player  = self.board.players[0]
            self.opp_decide = random.choice(deciders)
            self.turn_count = 0
            return encode_state(self.board, self.ai_player), {}

        def step(self, action):
            agent_id        = self.ai_player.player_id
            start_pos       = self.ai_player.position
            start_bankrupts = self.ai_player.bankrupt_count
            opps            = [o for o in self.board.players
                               if o.player_id != agent_id]
            opp_pos_before  = {o.player_id: o.position for o in opps}

            shop_decision = _action_to_shop(self.board, self.ai_player, int(action))
            result        = do_turn(self.board, shop_decision=shop_decision)
            self.turn_count += 1
            agent_bought   = bool(result.get("bought"))
            placed_setback = (shop_decision["head"] - shop_decision["tail"]
                              if (agent_bought and shop_decision) else 0)
            placed_combo   = bool(
                agent_bought and shop_decision
                and shop_decision["tail"] in self.board.bombs)
            winner = result["winner"]

            while winner is None and self.board.active_player.player_id != agent_id:
                opp        = self.board.active_player
                opp_result = do_turn(self.board,
                                     shop_decision=self.opp_decide(self.board, opp))
                self.turn_count += 1
                winner = opp_result["winner"]

            opp_setback = sum(max(0, opp_pos_before[o.player_id] - o.position)
                              for o in opps)
            terminated  = winner is not None
            truncated   = self.turn_count >= self.max_turns
            went_bankrupt = self.ai_player.bankrupt_count > start_bankrupts

            reward = self._compute_reward(
                winner, agent_bought, start_pos, went_bankrupt,
                opp_setback, placed_setback, placed_combo)

            return encode_state(self.board, self.ai_player), reward, terminated, truncated, {}

        def _compute_reward(self, winner, agent_bought, start_pos,
                            went_bankrupt, opp_setback,
                            placed_setback, placed_combo) -> float:
            if winner is not None:
                return 100.0 if winner.player_id == self.ai_player.player_id else -100.0
            reward  = (self.ai_player.position - start_pos) * 0.1
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


# ── Training ──────────────────────────────────────────────────────────────────

def train_ppo(total_timesteps: int = 100_000, opponent_pool: bool = False,
              save_path: str = MODEL_PATH):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        raise ImportError("pip install stable-baselines3 gymnasium")

    print(f"[PPO] Training {total_timesteps:,} steps "
          f"(opponent_pool={opponent_pool}) → {save_path}")

    EnvClass = build_training_env(opponent_pool=opponent_pool)
    vec_env  = make_vec_env(EnvClass, n_envs=4)

    model = None
    if os.path.exists(save_path):
        try:
            from stable_baselines3 import PPO as _PPO
            candidate = _PPO.load(save_path, env=vec_env, device="cpu")
            obs_ok = tuple(candidate.observation_space.shape) == (22,)
            act_ok = int(getattr(candidate.action_space, "n", -1)) == N_ACTIONS
            if obs_ok and act_ok:
                model = candidate
                print(f"[PPO] Continuing from {save_path} "
                      f"({model.num_timesteps:,} prior steps).")
            else:
                print(f"[PPO] shape mismatch — starting fresh.")
        except Exception as e:
            print(f"[PPO] cannot load {save_path}: {e} — starting fresh.")

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
            ent_coef=0.03,
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
    print(f"[PPO] Done. Saved {save_path} ({model.num_timesteps:,} steps).")


# ── Inference ─────────────────────────────────────────────────────────────────

def load_ppo_model():
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
        + ".\nTrain: python main.py --train"
    )


def ppo_decision(board: BoardState, player: Player, model) -> dict | None:
    """
    Autonomous inference — model picks strategy index, agent resolves
    exact tiles via engine rules only. No Expectimax helpers called.
    """
    if not player.can_buy_snake:
        return None
    obs = encode_state(board, player)
    action, _ = model.predict(obs, deterministic=False)
    shop = _action_to_shop(board, player, int(action))
    if shop:
        gprint(f"  [{player.name}] (PPO) snake {shop['head']}→{shop['tail']}")
    return shop
