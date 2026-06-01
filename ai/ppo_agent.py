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
#   0 = roll only (patient / save points)
#   1 = cheap short trap (cheap pressure)
#   2 = save & drop the longest affordable snake (max knockback)
#   3 = win-denial lurk near the goal (tiles 85-90)
#   4 = COMBO — snake tail on a bomb tile (knockback + bomb extra damage)
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

    Layout (22):
    [0]      our pos / 100
    [1]      our points / 2000 (cap)
    [2]      our snake count / 3
    [3-5]    opp positions / 100 (up to 3)
    [6-8]    opp points / 2000
    [9-11]   opp snake counts / 3
    [12]     snakes ahead of LEADING opp / 10
    [13]     distance to nearest bomb ahead / 20 (1 if none)
    [14]     distance to nearest ladder bottom ahead / 20 (1 if none)
    [15]     best available snake cost / 1000
    [16]     best snake damage / 100 (clamped -1..1)
    [17]     can afford best snake (0/1)
    [18]     COMBO snake available (tail-on-bomb) (0/1)
    [19]     in post-bankruptcy immunity (0/1)
    [20]     leading opp's distance to goal / 100
    [21]     turn number / 100
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
    state[13] = (min(bombs_ahead) - player.position) / 20.0 if bombs_ahead else 1.0
    state[13] = min(state[13], 1.0)
    ladders_ahead = [l.bottom for l in board.ladders if l.bottom > player.position]
    state[14] = (min(ladders_ahead) - player.position) / 20.0 if ladders_ahead else 1.0
    state[14] = min(state[14], 1.0)

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
        # Self-play: add the current best PPO as a frozen opponent — but
        # ONLY if its obs/action shape matches the current env, otherwise
        # predict() crashes mid-training (e.g. when we changed obs to 22-dim).
        try:
            frozen = load_ppo_model()
            obs_ok = tuple(frozen.observation_space.shape) == (22,)
            act_ok = int(getattr(frozen.action_space, "n", -1)) == N_ACTIONS
            if obs_ok and act_ok:
                deciders.append(lambda board, p: ppo_decision(board, p, frozen))
            else:
                print(f"[PPO] frozen model incompatible (obs={frozen.observation_space.shape}, "
                      f"n_actions={getattr(frozen.action_space, 'n', '?')}) — skipping self-play.")
        except Exception as e:
            print(f"[PPO] no frozen model for self-play: {e}")

    class SnakesLendersEnv(gym.Env):
        """
        Custom Gym environment for Snakes & Lenders.

        Action space (Discrete(5)): roll / cheap / big / lurk / combo.
        Observation: 22-dim vector (see encode_state).
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
            agent_bought   = bool(result.get("bought"))
            placed_setback = (shop_decision["head"] - shop_decision["tail"]
                              if (agent_bought and shop_decision) else 0)
            placed_combo   = bool(
                agent_bought and shop_decision
                and shop_decision["tail"] in self.board.bombs)
            winner         = result["winner"]

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
                            went_bankrupt, opp_setback,
                            placed_setback, placed_combo) -> float:
            """
            Win/loss dominates. Shaping deliberately pushes toward CUNNING
            aggression so the policy actually deploys traps (exact-head
            otherwise makes hoarding near-optimal): heavy reward for
            opponent setback, scaled reward for the snake's knockback size,
            BIG bonus for tail-on-bomb combos, win-denial bonus for big
            snakes near the goal, mild anti-hoard penalty for sitting rich.
            """
            if winner is not None:
                return 100.0 if winner.player_id == self.ai_player.player_id else -100.0

            reward = 0.0
            reward += (self.ai_player.position - start_pos) * 0.1   # progress
            reward += opp_setback * 0.5                             # sabotage paying off (heavy)
            if went_bankrupt:
                reward -= 50.0
            if agent_bought:
                # Reward deploying — bigger snake = bigger reward.
                reward += 4.0 + placed_setback * 0.3
                if placed_combo:
                    reward += 12.0          # MAXIMUM annoyance: tail-on-bomb combo
            else:
                # Anti-hoard: if we sat on plenty of points and didn't even
                # try a trap, take a small penalty (kills the "1 late snake"
                # passive policy and pushes varied attack timing).
                if self.ai_player.points > 100:
                    reward -= 1.0
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

    # Continue from an existing model if its obs/action shape matches the
    # current env, otherwise start fresh. This makes multi-stage training
    # (e.g. 3M base → +2M self-play polish) actually add up instead of
    # silently overwriting the previous run with a from-scratch model.
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
                print(f"[PPO] {save_path} shape mismatch "
                      f"(obs={candidate.observation_space.shape}, "
                      f"n_actions={candidate.action_space.n}) — starting fresh.")
        except Exception as e:
            print(f"[PPO] cannot load {save_path} for continuation: {e} "
                  f"— starting fresh.")

    if model is None:
        model = PPO(
            policy="MlpPolicy",     # Multi-layer perceptron neural network
            env=vec_env,
            learning_rate=3e-4,     # Standard PPO learning rate
            n_steps=2048,           # Steps before each policy update
            batch_size=64,
            n_epochs=10,            # PPO epochs per update
            gamma=0.99,             # Discount factor (values future rewards)
            clip_range=0.2,         # The PPO clipping value — keeps updates stable
            ent_coef=0.03,         # Entropy bonus → varied, UNPREDICTABLE cunning play
            device="cpu",          # CPU is faster here than the low-utilization GPU path.
            verbose=1,
        )

    # Silence the per-step game logs (board gen, AI buys, bankruptcies)
    # during training so only SB3's progress tables show.
    from game import log
    log.VERBOSE = False
    try:
        # Preserve the prior step counter when continuing (so logs/tables
        # show cumulative steps across stages, not just this run's slice).
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=(model.num_timesteps == 0),
        )
    finally:
        log.VERBOSE = True
    model.save(save_path)
    print(f"[PPO] Training complete! Model saved to {save_path} "
          f"({model.num_timesteps:,} total steps).")


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
    # Stochastic sampling at inference → unpredictable cunning play
    # (mixes patience, cheap pressure, big knockbacks, lurks, combos).
    action, _ = model.predict(obs, deterministic=False)

    shop = _action_to_shop(board, player, int(action))
    if shop:
        from game.log import gprint
        gprint(f"  [{player.name}] (PPO) snake "
               f"{shop['head']}→{shop['tail']}")
    return shop