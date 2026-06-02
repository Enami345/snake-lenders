"""
Snakes & Lenders — PPO Agent (Hard Mode)

Full PPO implementation using PyTorch only — no stable-baselines3,
no gymnasium. Training, inference, and model save/load are all custom.

Algorithm: Proximal Policy Optimization (Schulman et al. 2017)
  - Actor-Critic MLP (shared body, separate heads)
  - GAE advantage estimation (gamma=0.99, lam=0.95)
  - Clipped surrogate objective (epsilon=0.2)
  - Entropy bonus (coef=0.03)
  - Adam optimizer (lr=3e-4)

Model saved as: ai/ppo_model.pt  (torch.save format)
"""

import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from game.log import gprint
from game.models import BoardState, Player
from game.engine import calculate_snake_cost, can_place_snake, MAX_SNAKE_HEAD

# Expectimax used ONLY as training opponents — never for placement decisions.
from ai.expectimax import expectimax_decision, strong_decision

N_ACTIONS   = 5
OBS_DIM     = 22
MODEL_PATH  = "ai/ppo_model.pt"
BACKUP_PATH = "ai/ppo_model_backup.pt"

HEAD_MIN = 20
HEAD_MAX = MAX_SNAKE_HEAD  # 90


# ── Neural Network ────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-body MLP with separate policy (actor) and value (critic) heads.

    Input:  OBS_DIM-dim float32 observation vector
    Output: policy logits over N_ACTIONS + scalar state value
    """
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs_tensor):
        """Sample action + return log_prob and value (no grad needed)."""
        with torch.no_grad():
            logits, value = self(obs_tensor)
            dist   = Categorical(logits=logits)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_tensor, actions_tensor):
        """Compute log_probs, values, entropy for a batch (with grad)."""
        logits, values = self(obs_tensor)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)
        entropy   = dist.entropy()
        return log_probs, values, entropy


# ── Autonomous placement (no Expectimax helpers) ──────────────────────────────

def _leading_opponent(board: BoardState, player: Player):
    opps = [p for p in board.players if p.player_id != player.player_id]
    return max(opps, key=lambda o: o.position, default=None)


def _try_place(board, player, head, tail):
    if head <= tail or head > HEAD_MAX or tail < 1:
        return None
    ok, _ = can_place_snake(board, player, head, tail)
    if ok and calculate_snake_cost(player, head, tail) <= player.points:
        return {"head": head, "tail": tail}
    return None


def _action_to_shop(board: BoardState, player: Player, action: int):
    """
    Resolve strategy index → exact (head, tail) using engine rules only.
    No Expectimax helper functions are called.
    """
    if action == 0:
        return None

    opp = _leading_opponent(board, player)
    if opp is None or not player.can_buy_snake or player.position < 1:
        return None

    # 1 — short pressure trap
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

    # 2 — big knockback
    if action == 2:
        best, best_len = None, 0
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

    # 3 — win-denial lurk
    if action == 3:
        if opp.position < 60:
            return None
        best, best_len = None, 0
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

    # 4 — combo (tail on bomb)
    if action == 4:
        if not board.bombs:
            return None
        best, best_len = None, 0
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
    """22-dim observation vector (pure Python/numpy — no external AI libs)."""
    state = np.zeros(OBS_DIM, dtype=np.float32)
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
        state[12] = min(sum(1 for s in board.snakes
                            if s.head > leader.position) / 10.0, 1.0)
        state[20] = max(0.0, (100 - leader.position) / 100.0)

    bombs_ahead = [b for b in board.bombs if b > player.position]
    state[13] = min((min(bombs_ahead) - player.position) / 20.0, 1.0) \
                if bombs_ahead else 1.0
    ladders_ahead = [l.bottom for l in board.ladders if l.bottom > player.position]
    state[14] = min((min(ladders_ahead) - player.position) / 20.0, 1.0) \
                if ladders_ahead else 1.0

    # Best-snake features — inline board scan (no Expectimax call)
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

    # Combo available — inline check
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


# ── PPO Rollout Buffer ────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout of experience before a PPO update."""
    def __init__(self):
        self.obs       = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """GAE (Generalized Advantage Estimation)."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = (self.rewards[t]
                     + gamma * next_val * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages


# ── Game Environment (pure Python — no gymnasium) ────────────────────────────

class SnakesLendersEnv:
    """
    Minimal game environment wrapping game/ engine.
    No gymnasium dependency — reset/step interface is the same concept
    but implemented directly in Python.
    """
    def __init__(self, opp_decide_fn):
        from game.board import generate_board
        self.generate_board = generate_board
        self.opp_decide     = opp_decide_fn
        self.board          = None
        self.ai_player      = None
        self.max_turns      = 200
        self.turn_count     = 0

    def reset(self):
        from game.board import generate_board
        players = [
            Player(0, "PPO_Agent", is_ai=True, ai_difficulty="hard"),
            Player(1, "Opponent",  is_ai=True, ai_difficulty="easy"),
        ]
        self.board      = generate_board(players=players)
        self.ai_player  = self.board.players[0]
        self.turn_count = 0
        return encode_state(self.board, self.ai_player)

    def step(self, action: int):
        from game.engine import do_turn
        agent_id        = self.ai_player.player_id
        start_pos       = self.ai_player.position
        start_bankrupts = self.ai_player.bankrupt_count
        opps            = [o for o in self.board.players
                           if o.player_id != agent_id]
        opp_pos_before  = {o.player_id: o.position for o in opps}

        shop   = _action_to_shop(self.board, self.ai_player, action)
        result = do_turn(self.board, shop_decision=shop)
        self.turn_count += 1

        agent_bought   = bool(result.get("bought"))
        placed_setback = (shop["head"] - shop["tail"]
                          if (agent_bought and shop) else 0)
        placed_combo   = bool(agent_bought and shop
                              and shop["tail"] in self.board.bombs)
        winner         = result["winner"]

        while winner is None and self.board.active_player.player_id != agent_id:
            opp        = self.board.active_player
            opp_result = do_turn(self.board,
                                 shop_decision=self.opp_decide(self.board, opp))
            self.turn_count += 1
            winner = opp_result["winner"]

        opp_setback   = sum(max(0, opp_pos_before[o.player_id] - o.position)
                            for o in opps)
        terminated    = winner is not None
        truncated     = self.turn_count >= self.max_turns
        went_bankrupt = self.ai_player.bankrupt_count > start_bankrupts

        reward = _compute_reward(
            winner, self.ai_player, agent_bought, start_pos,
            went_bankrupt, opp_setback, placed_setback, placed_combo)

        obs  = encode_state(self.board, self.ai_player)
        done = terminated or truncated
        return obs, reward, done, {}


def _compute_reward(winner, agent, agent_bought, start_pos,
                    went_bankrupt, opp_setback, placed_setback, placed_combo):
    if winner is not None:
        return 100.0 if winner.player_id == agent.player_id else -100.0
    r  = (agent.position - start_pos) * 0.1
    r += opp_setback * 0.5
    if went_bankrupt:
        r -= 50.0
    if agent_bought:
        r += 4.0 + placed_setback * 0.3
        if placed_combo:
            r += 12.0
    else:
        if agent.points > 100:
            r -= 1.0
    return r


# ── PPO Training ──────────────────────────────────────────────────────────────

def train_ppo(total_timesteps: int = 100_000, opponent_pool: bool = False,
              save_path: str = MODEL_PATH):
    """
    Train the PPO agent using pure PyTorch.
    No stable-baselines3 or gymnasium — all training logic is implemented here.

    Hyperparameters:
        n_steps    = 2048   steps per rollout (per env)
        n_epochs   = 10     PPO update epochs per rollout
        batch_size = 64     minibatch size
        gamma      = 0.99   discount factor
        lam        = 0.95   GAE lambda
        clip_eps   = 0.2    PPO clipping epsilon
        ent_coef   = 0.03   entropy bonus coefficient
        vf_coef    = 0.5    value loss coefficient
        lr         = 3e-4   Adam learning rate
    """
    from game import log as game_log

    # Hyperparameters
    n_steps    = 2048
    n_epochs   = 10
    batch_size = 64
    gamma      = 0.99
    lam        = 0.95
    clip_eps   = 0.2
    ent_coef   = 0.03
    vf_coef    = 0.5
    lr         = 3e-4
    device     = torch.device("cpu")

    print(f"[PPO] Training {total_timesteps:,} steps "
          f"(opponent_pool={opponent_pool}) → {save_path}")
    print(f"[PPO] Pure PyTorch implementation — no stable-baselines3")

    # Build opponent pool
    def _easy(board, p):   return expectimax_decision(board, p)
    def _strong(board, p): return strong_decision(board, p)
    deciders = [_easy]
    if opponent_pool:
        deciders = [_easy, _strong]
        try:
            frozen_model, frozen_net = load_ppo_model()
            deciders.append(lambda board, p: ppo_decision(board, p,
                                                          frozen_model,
                                                          frozen_net))
            print("[PPO] Frozen self-play opponent added to pool.")
        except Exception as e:
            print(f"[PPO] no frozen model for self-play: {e}")

    # Network + optimizer
    net       = ActorCritic(OBS_DIM, N_ACTIONS).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Load existing checkpoint if compatible
    if os.path.exists(save_path):
        try:
            ckpt = torch.load(save_path, map_location=device, weights_only=False)
            if (ckpt.get("obs_dim") == OBS_DIM and
                    ckpt.get("n_actions") == N_ACTIONS):
                net.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                prior_steps = ckpt.get("total_steps", 0)
                print(f"[PPO] Continuing from {save_path} "
                      f"({prior_steps:,} prior steps).")
            else:
                print(f"[PPO] {save_path} shape mismatch — starting fresh.")
                prior_steps = 0
        except Exception as e:
            print(f"[PPO] cannot load {save_path}: {e} — starting fresh.")
            prior_steps = 0
    else:
        prior_steps = 0

    # Single environment (serial — simpler, avoids multiprocessing complexity)
    env    = SnakesLendersEnv(opp_decide_fn=lambda b, p: random.choice(deciders)(b, p))
    buffer = RolloutBuffer()

    obs       = env.reset()
    steps_done = 0
    updates    = 0

    game_log.VERBOSE = False
    try:
        while steps_done < total_timesteps:
            buffer.clear()
            obs = env.reset()   # fresh episode each rollout

            # ── Collect n_steps of experience ──────────────────────────
            for _ in range(n_steps):
                obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, log_prob, value = net.act(obs_t)
                next_obs, reward, done, _ = env.step(action)

                buffer.add(obs, action, log_prob, reward, value, done)
                obs = next_obs
                steps_done += 1

                if done:
                    obs = env.reset()
                if steps_done >= total_timesteps:
                    break

            # Bootstrap last value for GAE
            with torch.no_grad():
                last_obs_t = torch.tensor(obs, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                _, last_val = net(last_obs_t)
                last_value  = last_val.item()

            returns, advantages = buffer.compute_returns_and_advantages(
                last_value, gamma, lam)

            # Normalise advantages
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            obs_arr    = np.array(buffer.obs,       dtype=np.float32)
            act_arr    = np.array(buffer.actions,   dtype=np.int64)
            lp_arr     = np.array(buffer.log_probs, dtype=np.float32)
            ret_t      = torch.tensor(returns,      dtype=torch.float32, device=device)
            obs_t_all  = torch.tensor(obs_arr,      dtype=torch.float32, device=device)
            act_t_all  = torch.tensor(act_arr,      dtype=torch.int64,   device=device)
            old_lp_all = torch.tensor(lp_arr,       dtype=torch.float32, device=device)

            # ── PPO update: n_epochs passes over the buffer ─────────────
            T = len(buffer.obs)
            for _ in range(n_epochs):
                indices = np.random.permutation(T)
                for start in range(0, T, batch_size):
                    idx = indices[start:start + batch_size]
                    if len(idx) == 0:
                        continue
                    b_obs  = obs_t_all[idx]
                    b_act  = act_t_all[idx]
                    b_adv  = adv_t[idx]
                    b_ret  = ret_t[idx]
                    b_old  = old_lp_all[idx]

                    new_lp, values, entropy = net.evaluate(b_obs, b_act)

                    ratio    = torch.exp(new_lp - b_old)
                    clip_obj = torch.min(
                        ratio * b_adv,
                        torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv)

                    policy_loss = -clip_obj.mean()
                    value_loss  = nn.functional.mse_loss(values, b_ret)
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optimizer.step()

            updates += 1
            total_so_far = prior_steps + steps_done
            ep_rew = np.mean(buffer.rewards)
            if updates % 5 == 0:
                print(f"[PPO] steps {total_so_far:,} | "
                      f"ep_rew_mean {ep_rew:.2f} | "
                      f"loss {loss.item():.4f}")

    finally:
        game_log.VERBOSE = True

    # Save checkpoint
    torch.save({
        "model_state":     net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "obs_dim":         OBS_DIM,
        "n_actions":       N_ACTIONS,
        "total_steps":     prior_steps + steps_done,
    }, save_path)
    print(f"[PPO] Done. Saved {save_path} "
          f"({prior_steps + steps_done:,} total steps).")


# ── Inference ─────────────────────────────────────────────────────────────────

def load_ppo_model():
    """
    Load trained model. Returns (checkpoint_dict, ActorCritic_net).
    Falls back to backup if main is missing/corrupt.
    """
    device = torch.device("cpu")
    last_err = None
    for path in (MODEL_PATH, BACKUP_PATH):
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                net  = ActorCritic(
                    ckpt.get("obs_dim", OBS_DIM),
                    ckpt.get("n_actions", N_ACTIONS)).to(device)
                net.load_state_dict(ckpt["model_state"])
                net.eval()
                if path == BACKUP_PATH:
                    print(f"[PPO] Loaded backup model ({path}).")
                return ckpt, net
            except Exception as e:
                last_err = e
    raise FileNotFoundError(
        f"No loadable PPO model at {MODEL_PATH} or {BACKUP_PATH}"
        + (f" (last error: {last_err})" if last_err else "")
        + "\nTrain: python -c \"from ai.ppo_agent import train_ppo; train_ppo(500000, True)\""
    )


def ppo_decision(board: BoardState, player: Player,
                 ckpt=None, net: ActorCritic = None) -> dict | None:
    """
    Inference entry point. Loads model if not provided.
    Resolves action → exact tiles via engine rules. No Expectimax helpers.
    """
    if not player.can_buy_snake:
        return None

    if net is None:
        _, net = load_ppo_model()

    device  = next(net.parameters()).device
    obs_t   = torch.tensor(encode_state(board, player),
                           dtype=torch.float32, device=device).unsqueeze(0)
    action, _, _ = net.act(obs_t)
    shop = _action_to_shop(board, player, action)
    if shop:
        gprint(f"  [{player.name}] (PPO) snake {shop['head']}→{shop['tail']}")
    return shop
