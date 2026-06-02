# AI Crash Course — Expectimax vs PPO

**Course:** Introduction to Artificial Intelligence — PUP BSCS 3-4, Group 10  
**Files:** `ai/expectimax.py` · `ai/ppo_agent.py`

---

## The Big Picture

Both agents answer the same question every turn:

> *"Should I buy a snake right now, and if so, which one?"*

They differ fundamentally in **how** they answer it:

| | Expectimax (Easy) | PPO (Hard) |
|---|---|---|
| **Type** | Classical search | Reinforcement learning |
| **Learns?** | No — rules written by hand | Yes — learns by playing ~75,000 games |
| **Looks ahead** | 1 dice roll (6 outcomes) | Entire game trajectory |
| **Decision basis** | Expected value formula | Neural network weights |
| **Behavior** | Fixed, predictable, weak | Stochastic, unpredictable, cunning |
| **Training needed?** | No | Yes (`train_ppo(...)`) |
| **File** | `ai/expectimax.py` | `ai/ppo_agent.py` |

---

## Part 1 — Expectimax (`ai/expectimax.py`)

### What is Expectimax?

Expectimax is a game-tree search algorithm designed for environments with
**randomness** (like dice). Unlike Minimax (which assumes a perfect adversary),
Expectimax models the opponent's moves as **chance nodes** — averaging over all
possible random outcomes instead of assuming the worst.

```
              [MAX node]  ← AI picks the best action
             /     |     \
         Buy A   Buy B   Roll
          |       |        |
    [CHANCE]  [CHANCE]  [CHANCE] ← dice: 6 outcomes each with prob 1/6
    /|||||    /|||||     /|||||
  1 2 3 4 5 6  ...       ...
  Evaluate board state at each leaf
```

### Imports & why

```python
import random                                # for EASY_SKIP_PROB coin flip
from game.log import gprint                  # gated log (silent during training)
from game.models import BoardState, Player, Snake
from game.engine import (
    calculate_snake_cost,   # compute snake price before buying
    can_place_snake,        # validate placement (7 rules)
    MAX_SNAKE_HEAD,         # tile cap = 90
    MIN_SNAKE_COST,         # floor = 12
    STRIKE_ZONE,            # 0 = exact-head only
)
```

### Key constants

| Constant | Value | Purpose |
|---|---|---|
| `TILE_VALUE` | 10.0 | 1 tile of setback ≈ 10 pts of damage (for EV calc) |
| `HIT_IN_RANGE` | 0.50 | Hit prob when head is 1–6 tiles ahead of opponent |
| `HIT_NEAR` | 0.35 | Hit prob 7–12 tiles ahead |
| `HIT_FAR` | 0.20 | Hit prob >12 tiles ahead |
| `WIN_DENIAL_TILE` | 85 | Snake head at/above = near-goal lurk |
| `WIN_DENIAL_OPP` | 72 | Opponent at/above = almost winning |
| `WIN_DENIAL_MULT` | 1.8 | Damage multiplier for win-denial plays |
| `EASY_SABOTAGE_MIN_POS` | 35 | Won't attack until opponent passes tile 35 |
| `EASY_BUFFER` | 45 | Won't buy if remaining points < 45 (over-hoards) |
| `EASY_SKIP_PROB` | 0.35 | 35% random skip chance (hesitation handicap) |

### Core algorithm — `evaluate_snake_placement`

The damage model that scores every potential snake placement:

```
damage = setback_tiles × TILE_VALUE × p_hit × progress_weight

where:
  setback_tiles    = head − tail
  p_hit            = HIT_IN_RANGE (0.50) if dist ≤ 6
                     HIT_NEAR     (0.35) if dist ≤ 12
                     HIT_FAR      (0.20) otherwise
  progress_weight  = 1.0 + (opponent.position / 100)
                     (erasing an advanced opponent = worth more)

if head ≥ WIN_DENIAL_TILE and opponent.position ≥ WIN_DENIAL_OPP:
  damage × = WIN_DENIAL_MULT  (1.8×)
```

### Core algorithm — `expected_value_after_roll`

The classic Expectimax chance node:

```
EV = (1/6) × Σ evaluate_state(board_after_roll_i)   for i in {1,2,3,4,5,6}

evaluate_state scores:
  + player.position × 10         (our progress)
  + player.points × 0.1          (wallet)
  + (player.pos − opp.pos) × 5   (relative lead, per opponent)
  + snake threat bonus            (50 pts if snake in opponent's dice range)
```

### Easy-mode handicaps (deliberate weaknesses)

`expectimax_decision` is intentionally crippled:

```python
def expectimax_decision(board, player):
    # 1. Only reacts if opponent is past tile 35
    if leader_pos < EASY_SABOTAGE_MIN_POS: return None

    # 2. Randomly skips 35% of the time
    if random.random() < EASY_SKIP_PROB: return None

    # 3. Only ever places cheap short traps (lengths 5, 6, 8)
    shop = propose_cheap_trap(board, player)

    # 4. Refuses if remaining points would drop below 45
    if player.points - cost < EASY_BUFFER: return None

    return shop
```

Never uses `propose_big_snake`, `propose_lurk`, or `propose_combo` — those are
reserved for `strong_decision` and the PPO agent.

### Placement strategy functions (shared with PPO)

All use **catch-optimal offsets**: head placed so it lands inside the opponent's
next-roll dice range (1–6 tiles ahead).

```python
_catch_offsets() = range(STRIKE_ZONE + 1, 7)   # = [1,2,3,4,5,6] when Z=0
```

| Function | Strategy | Target lengths |
|---|---|---|
| `propose_cheap_trap` | Shortest affordable, max catch chance | 5, 6, 8 tiles |
| `propose_big_snake` | Longest affordable, max setback | tail=1, 10, 20, 30, 40 tiles back |
| `propose_lurk` | Near-goal (opp pos ≥ 60), max setback | tail=1, 20, 30, 40 back |
| `propose_combo` | Tail on a bomb tile (knockback + bomb damage stacked) | Max setback with tail ∈ bombs |

### Full Expectimax agent pseudocode

```
ALGORITHM: Expectimax Agent (Easy Mode)
INPUT:  board state, player
OUTPUT: {head, tail} snake placement OR None

────────────────────────────────────────────────────────────────
FUNCTION expectimax_decision(board, player):

  // Guard: can we even buy?
  IF player.snake_count >= 3:
    RETURN None

  // Handicap 1: only react when opponent is far along
  leader_pos ← max(opponent.position for opponent in board.players)
  IF leader_pos < EASY_SABOTAGE_MIN_POS (35):
    RETURN None

  // Handicap 2: random hesitation (35% skip)
  IF random() < EASY_SKIP_PROB (0.35):
    RETURN None

  // Handicap 3: only place a cheap short trap
  shop ← propose_cheap_trap(board, player)
  IF shop is None:
    RETURN None

  // Handicap 4: over-cautious hoarding
  cost ← calculate_snake_cost(player, shop.head, shop.tail)
  IF player.points - cost < EASY_BUFFER (45):
    RETURN None

  RETURN shop

────────────────────────────────────────────────────────────────
FUNCTION propose_cheap_trap(board, player):

  opp ← opponent with highest position
  FOR offset IN [1, 2, 3, 4, 5, 6]:          // catch-optimal dice range
    head ← opp.position + offset
    FOR length IN [8, 6, 5]:                  // short traps only
      tail ← head - length
      IF is_valid(board, player, head, tail) AND affordable:
        RETURN {head, tail}
  RETURN None

────────────────────────────────────────────────────────────────
FUNCTION evaluate_snake_placement(board, player, head, tail):

  setback ← head - tail
  FOR each opponent:
    dist    ← head - opponent.position
    p_hit   ← 0.50 if dist ≤ 6
               0.35 if dist ≤ 12
               0.20 otherwise
    weight  ← 1.0 + (opponent.position / 100)
    damage  ← setback × TILE_VALUE × p_hit × weight

    IF head ≥ 85 AND opponent.position ≥ 72:  // win-denial
      damage ← damage × 1.8

  RETURN max(damage across all opponents)

────────────────────────────────────────────────────────────────
FUNCTION expected_value_after_roll(board, player):

  total ← 0
  FOR die IN [1, 2, 3, 4, 5, 6]:
    new_pos ← player.position + die
    IF new_pos > 100: new_pos ← player.position  // overshoot = stay
    apply ladder at new_pos if any
    apply snake  at new_pos if any
    total ← total + evaluate_state(board with player at new_pos)
  RETURN total / 6

FUNCTION evaluate_state(board, player):
  score ← player.position × 10
  score ← score + player.points × 0.1
  FOR each opponent:
    score ← score + (player.position - opponent.position) × 5
    FOR each snake owned by player:
      IF snake.head > opponent.position:
        dist_ahead ← snake.head - opponent.position
        score ← score + 50 if dist_ahead ≤ 6
                         20 if dist_ahead ≤ 12
                          5 otherwise
  RETURN score
```

---

## Part 2 — PPO Agent (`ai/ppo_agent.py`)

### What is PPO?

**Proximal Policy Optimization** is a reinforcement learning algorithm. Instead
of hand-coding decisions, PPO learns by **playing the game thousands of times**
and adjusting a neural network toward actions that lead to winning.

Key idea: the agent has a **policy** π(action | state) — a probability
distribution over actions given the current board. PPO updates π to increase
the probability of actions that led to high reward, but **clips the update**
so it can't change too drastically in one step (preventing catastrophic
forgetting).

```
BEFORE training                    AFTER 3M steps
Policy π:                          Policy π:
  roll      → 25%                    roll      → 28%
  cheap     → 25%                    cheap     → 20%
  big       → 25%       ──►          big       → 12%
  lurk      → 25%                    lurk      → 31%
  combo     → 25%                    combo     → 34%  ← learned combo is best
```

### Imports & why

```python
import os                           # file path checks for model load/save
import random                       # pick opponent from pool each episode
import numpy as np                  # encode_state → float32 array
import torch                        # neural network, autograd, Adam optimizer
import torch.nn as nn               # Linear, Tanh, Sequential, MSE loss
import torch.optim as optim         # Adam optimizer
from torch.distributions import Categorical   # stochastic action sampling

from game.log import gprint
from game.models import BoardState, Player
from game.engine import calculate_snake_cost, can_place_snake, MAX_SNAKE_HEAD

from ai.expectimax import (
    expectimax_decision,            # Easy bot — training opponent only
    strong_decision,                # Strong bot — training opponent only
)
# No gymnasium, no stable-baselines3.
# Training env, rollout buffer, and PPO update loop are all implemented here.
```

**Critical difference from old builds:** `ppo_agent.py` no longer imports
`propose_cheap_trap`, `propose_big_snake`, `propose_lurk`, `propose_combo`, or
`find_best_snake_to_buy` from Expectimax. PPO resolves placements by scanning
the board itself using only engine rules (`can_place_snake`, `calculate_snake_cost`).
Expectimax is imported **only as a training opponent** — never as a placement helper.

### Action space — Discrete(5)

```
N_ACTIONS = 5

Action 0 → roll only (return None)
Action 1 → short pressure trap    — scan board for shortest affordable
           snake in opponent's dice range (engine rules only)
Action 2 → big knockback          — scan board for longest affordable
           snake in opponent's dice range (engine rules only)
Action 3 → win-denial lurk        — snake near tile 90 when opp pos ≥ 60
           (engine rules only)
Action 4 → combo                  — snake whose tail is a bomb tile
           (engine rules only)
```

Each action calls `_action_to_shop(board, player, action)` which scans
the board inline. No `propose_*` functions from Expectimax are called.

### Observation space — 22-dim float32 vector

```python
encode_state(board, player) → np.ndarray shape (22,)
```

| Index | Feature | Formula |
|---|---|---|
| 0 | Our position | `pos / 100` |
| 1 | Our points | `min(pts / 2000, 1.0)` |
| 2 | Our snake count | `count / 3` |
| 3–5 | Opponent positions (up to 3) | `pos / 100` each |
| 6–8 | Opponent points | `min(pts / 2000, 1.0)` each |
| 9–11 | Opponent snake counts | `count / 3` each |
| 12 | Snakes ahead of leader | `min(count / 10, 1.0)` |
| 13 | Nearest bomb distance ahead | `min(dist / 20, 1.0)` · 1.0 if none |
| 14 | Nearest ladder distance ahead | `min(dist / 20, 1.0)` · 1.0 if none |
| 15 | Best snake cost | `min(cost / 1000, 1.0)` — computed via inline board scan |
| 16 | Best snake damage | `min(dmg / 100, 1.0)` — computed via inline board scan |
| 17 | Can afford best snake | `1.0` or `0.0` — from inline board scan |
| 18 | Combo available | `1.0` if bomb tail valid — checked inline via `_try_place` |
| 19 | Bankruptcy immunity active | `1.0` if `bankrupt_immune > 0` |
| 20 | Leader's distance to goal | `(100 − leader.pos) / 100` |
| 21 | Turn number | `min(turn / 100, 1.0)` |

### Reward function — `_compute_reward`

```python
# Terminal (dominates everything else)
if winner == agent:     reward = +100.0
if winner == opponent:  reward = −100.0

# Shaping (non-terminal only)
reward += (agent.position − start_position) × 0.1    # progress
reward += opponent_setback × 0.5                      # heavy sabotage reward
if went_bankrupt:  reward −= 50.0                     # bankruptcy penalty

if agent_bought:
    reward += 4.0 + placed_setback × 0.3              # deploy reward (bigger = more)
    if placed_combo:  reward += 12.0                   # MAX annoyance bonus
else:
    if agent.points > 100:  reward −= 1.0             # anti-hoard penalty
```

The shaping is deliberately heavy on sabotage rewards so the policy doesn't
discover that hoarding points and doing nothing is "safe."

### Neural network — `ActorCritic`

Custom MLP implemented in PyTorch (no stable-baselines3):

```python
class ActorCritic(nn.Module):
    body   = Sequential(Linear(22,64), Tanh(), Linear(64,64), Tanh())
    actor  = Linear(64, 5)    # outputs logits over 5 actions
    critic = Linear(64, 1)    # outputs scalar state value V(s)

forward(x) → (logits, value)
act(obs)   → sample action + log_prob + value  (no grad, inference)
evaluate(obs, actions) → log_probs, values, entropy  (with grad, training)
```

### PPO update algorithm (pure PyTorch, no SB3)

```
For each update iteration:
  1. Collect n_steps=2048 of experience (single env, serial)
  2. Bootstrap last value for GAE:
       last_value ← critic(last_obs)
  3. Compute GAE advantages:
       FOR t = T downto 0:
         delta_t ← reward_t + γ × V(s_{t+1}) × (1-done_t) − V(s_t)
         A_t     ← delta_t + γ × λ × A_{t+1} × (1-done_t)
       returns  ← advantages + values
  4. Normalise advantages: A ← (A − mean(A)) / (std(A) + 1e-8)
  5. For n_epochs=10, minibatches of 64:
       ratio    ← exp(log_π_new(a|s) − log_π_old(a|s))
       L_clip   ← min(ratio×A, clip(ratio, 1−0.2, 1+0.2)×A)
       L_value  ← MSE(V(s), returns)
       L_entropy← −entropy of π_new
       loss     ← −L_clip + 0.5×L_value + 0.03×L_entropy
       loss.backward(); clip_grad_norm(0.5); Adam.step()
```

### Training flow (no gymnasium, no stable-baselines3)

```
train_ppo(total_timesteps, opponent_pool=True):

  // Build opponent pool
  deciders ← [expectimax_decision, strong_decision]
  IF opponent_pool AND ai/ppo_model.pt exists AND shape matches:
    deciders.append(frozen_ppo_decision)   // self-play

  // Load or start fresh
  net ← ActorCritic(obs_dim=22, n_actions=5)
  IF ai/ppo_model.pt exists AND obs_dim==22 AND n_actions==5:
    net.load_state_dict(ckpt["model_state"])  // CONTINUE
    optimizer.load_state_dict(ckpt["optimizer_state"])
  ELSE:
    net fresh weights, fresh Adam              // FRESH

  env ← SnakesLendersEnv(random.choice(deciders))  // pure Python env
  buffer ← RolloutBuffer()

  WHILE steps < total_timesteps:
    collect n_steps via env.step()
    compute GAE returns + advantages
    PPO update (10 epochs × minibatches)

  torch.save({model_state, optimizer_state, obs_dim, n_actions,
              total_steps}, "ai/ppo_model.pt")
```

### Inference

```python
ppo_decision(board, player, ckpt, net):
    obs    = encode_state(board, player)      # 22-dim vector
    obs_t  = torch.tensor(obs).unsqueeze(0)
    action = net.act(obs_t)                   # STOCHASTIC sample (Categorical)
    shop   = _action_to_shop(board, player, action)
    return shop   # {head, tail} or None
```

Stochastic sampling via `torch.distributions.Categorical` — no
`deterministic` flag, always samples from the policy distribution.
Makes the agent unpredictable — opponents can't learn its exact timing.

### Full PPO agent pseudocode

```
ALGORITHM: PPO Agent (Hard Mode — pure PyTorch, no SB3)
INPUT:  board state, player, ActorCritic net
OUTPUT: {head, tail} snake placement OR None

────────────────────────────────────────────────────────────────
NEURAL NETWORK

CLASS ActorCritic(nn.Module):
  body   ← Linear(22,64) → Tanh → Linear(64,64) → Tanh
  actor  ← Linear(64, 5)    // policy logits
  critic ← Linear(64, 1)    // state value V(s)

  FUNCTION act(obs_tensor):        // inference, no grad
    logits, value ← forward(obs)
    dist   ← Categorical(logits=logits)
    action ← dist.sample()          // STOCHASTIC
    RETURN action, dist.log_prob(action), value

  FUNCTION evaluate(obs, actions):  // training, with grad
    logits, values ← forward(obs)
    dist      ← Categorical(logits=logits)
    log_probs ← dist.log_prob(actions)
    entropy   ← dist.entropy()
    RETURN log_probs, values, entropy

────────────────────────────────────────────────────────────────
INFERENCE (called every Hard AI turn — no Expectimax helpers)

FUNCTION ppo_decision(board, player, ckpt, net):
  IF player.snake_count >= 3: RETURN None

  obs    ← encode_state(board, player)         // 22-dim float32
  obs_t  ← torch.tensor(obs).unsqueeze(0)
  action, _, _ ← net.act(obs_t)               // stochastic sample
  shop   ← _action_to_shop(board, player, action)
  RETURN shop  // {head, tail} or None

FUNCTION _action_to_shop(board, player, action):
  // Resolves strategy → exact tiles via ENGINE RULES ONLY
  // No Expectimax helper functions called
  IF action == 0: RETURN None
  opp ← opponent with highest position
  IF action == 1:  // short trap
    FOR offset IN [1..6]:
      head ← opp.position + offset
      FOR length IN [5,6,8,10]:
        IF can_place_snake(head, head-length) AND affordable:
          RETURN {head, head-length}
  IF action == 2:  // big knockback
    best ← None
    FOR offset IN [1..6]:
      FOR tail IN [1, head-40, head-30, head-20, head-10]:
        IF valid AND longer than best: best ← {head, tail}
    RETURN best
  IF action == 3:  // win-denial (opp pos ≥ 60)
    IF opp.position < 60: RETURN None
    similar search near tile 90
  IF action == 4:  // combo (tail on bomb)
    FOR tail IN board.bombs:
      FOR offset IN [1..6]:
        IF valid AND bigger setback: best ← {head, tail}
    RETURN best

────────────────────────────────────────────────────────────────
STATE ENCODING (no Expectimax calls — all inline board scan)

FUNCTION encode_state(board, player) → float32[22]:
  state[0-2]   ← pos/100, points/2000, snake_count/3
  state[3-11]  ← opp positions, points, snake counts (up to 3 opps)
  state[12]    ← snakes_ahead_of_leader / 10
  state[13]    ← dist_to_nearest_bomb_ahead / 20  (1.0 if none)
  state[14]    ← dist_to_nearest_ladder_ahead / 20 (1.0 if none)

  // [15-17]: best-snake features — inline board scan (NOT find_best_snake)
  FOR offset IN [1..6]:
    FOR tail IN [1, head-20, head-10, head-5]:
      IF can_place_snake(head, tail):
        dmg  ← (head-tail) × 10 × (1 + opp.pos/100)
        IF dmg > best_dmg: update best_cost, best_dmg, can_afford

  state[15] ← min(best_cost / 1000, 1.0)
  state[16] ← min(best_dmg / 100, 1.0)
  state[17] ← 1.0 if affordable else 0.0

  // [18]: combo available — inline check (NOT propose_combo)
  FOR tail IN board.bombs:
    IF _try_place(board, player, opp.pos+offset, tail): combo ← 1.0

  state[18] ← combo
  state[19] ← 1.0 if bankrupt_immune > 0 else 0.0
  state[20] ← (100 - leader.pos) / 100
  state[21] ← turn_number / 100
  RETURN state

────────────────────────────────────────────────────────────────
ENVIRONMENT STEP (pure Python — no gymnasium)

CLASS SnakesLendersEnv:
  FUNCTION reset():
    board ← generate_board(2 players)
    RETURN encode_state(board, agent)

  FUNCTION step(action):
    shop   ← _action_to_shop(board, agent, action)
    result ← do_turn(board, shop_decision=shop)
    agent_bought   ← result.bought
    placed_setback ← shop.head - shop.tail  if bought else 0
    placed_combo   ← shop.tail IN board.bombs  if bought else False
    winner ← result.winner

    WHILE winner is None AND board.active_player != agent:
      opp_result ← do_turn(board, opp_decide_fn(board, opp))
      winner ← opp_result.winner

    opp_setback ← sum(max(0, opp_pos_before[o] - o.position) for each opp)
    reward ← _compute_reward(winner, agent, agent_bought, ...)
    RETURN encode_state(board, agent), reward, done

────────────────────────────────────────────────────────────────
REWARD FUNCTION (unchanged)

  IF winner == agent:    RETURN +100.0
  IF winner != agent:    RETURN -100.0
  reward ← (agent.pos - start_pos) × 0.1
  reward += opp_setback × 0.5
  IF went_bankrupt: reward -= 50.0
  IF agent_bought:
    reward += 4.0 + placed_setback × 0.3
    IF placed_combo: reward += 12.0
  ELSE IF agent.points > 100: reward -= 1.0

────────────────────────────────────────────────────────────────
PPO POLICY UPDATE (pure PyTorch)

CLASS RolloutBuffer:
  stores: obs, actions, log_probs, rewards, values, dones

  FUNCTION compute_returns_and_advantages(last_value, γ=0.99, λ=0.95):
    FOR t = T downto 0:
      delta_t ← reward_t + γ × V(t+1) × (1-done_t) − V(t)
      A_t     ← delta_t + γ × λ × A_{t+1} × (1-done_t)
    returns ← advantages + values

FOR each update iteration:
  collect 2048 steps → RolloutBuffer
  bootstrap: last_value ← critic(last_obs)
  compute GAE advantages + returns
  A ← (A - mean(A)) / (std(A) + 1e-8)   // normalise

  FOR epoch = 1 to 10:
    FOR each minibatch of 64:
      new_lp, values, entropy ← net.evaluate(obs_batch, act_batch)
      ratio    ← exp(new_lp - old_lp)
      L_clip   ← min(ratio×A, clip(ratio, 0.8, 1.2)×A)
      L_value  ← MSE(values, returns)
      L_entropy← entropy.mean()
      loss     ← -L_clip + 0.5×L_value - 0.03×L_entropy
      loss.backward()
      clip_grad_norm(net, max_norm=0.5)
      Adam.step(lr=3e-4)

────────────────────────────────────────────────────────────────
TRAINING PIPELINE

FUNCTION train_ppo(total_timesteps, opponent_pool):

  deciders ← [expectimax_decision, strong_decision]
  IF opponent_pool AND ppo_model.pt exists AND shape matches:
    deciders.append(frozen_ppo)       // self-play opponent

  net ← ActorCritic(22, 5)
  IF ppo_model.pt exists AND obs_dim==22 AND n_actions==5:
    net.load_state_dict(ckpt["model_state"])   // CONTINUE
    optimizer.load_state_dict(ckpt["optimizer_state"])
  ELSE: fresh weights                           // FRESH

  env ← SnakesLendersEnv(random.choice(deciders))
  buffer ← RolloutBuffer()

  WHILE steps < total_timesteps:
    collect 2048 steps → buffer
    compute GAE
    PPO update (10 epochs)

  torch.save({model_state, optimizer_state, obs_dim:22,
              n_actions:5, total_steps}, "ai/ppo_model.pt")
```

---

## Part 3 — Side-by-Side Comparison

### Decision process

```
EXPECTIMAX (each turn):
  1. Is opponent past tile 35? → No? Return None.
  2. Random 35% skip? → Skip? Return None.
  3. propose_cheap_trap() → valid placement?
  4. Would remaining points < 45? → Too cautious? Return None.
  5. Return trap.

PPO (each turn):
  1. encode_state() → 22-dim float vector (inline board scan, no Expectimax)
  2. net.act(obs) → Categorical.sample() → action index (stochastic)
  3. _action_to_shop(action) → inline board scan via engine rules → {head,tail}
  4. Return shop or None.
```

### Knowledge source

| | Expectimax | PPO |
|---|---|---|
| **Who encodes knowledge** | Programmer (hand-tuned rules) | The game itself (via reward signal) |
| **Placement decided by** | `propose_*` helper functions | Neural net output → engine rule scan |
| **Adapts to opponents** | No — fixed heuristic | Partly — trained vs varied pool |
| **External AI library** | None | `torch` only (no SB3, no gymnasium) |
| **Knows about combo?** | Yes — `propose_combo` is a hand-coded function | Yes — learned via `+12.0` reward signal |

### Why PPO beats Expectimax

1. **Timing** — Expectimax has a hard trigger (tile 35). PPO learns economically optimal timing from thousands of games.
2. **Strategy mix** — Expectimax only uses cheap traps. PPO mixes all 5 strategies.
3. **Anti-hoard** — The `-1.0` penalty explicitly punishes Expectimax-style hoarding.
4. **Combo exploitation** — `+12.0` reward teaches PPO to prefer tail-on-bomb snakes.
5. **Unpredictability** — Stochastic `Categorical.sample()` makes timing hard to predict.

### Why Expectimax is still useful

1. **No training needed** — instant, zero setup.
2. **Transparent** — every decision traceable through a formula.
3. **Training opponent** — `expectimax_decision` and `strong_decision` are the pool opponents PPO trains against.
4. **Beatable baseline** — validates Hard AI is actually better.

---

## Part 4 — File Structure Summary

```
ai/expectimax.py
├── evaluate_state()           → board score for a player (EV helper)
├── expected_value_after_roll() → chance node: average 6 dice outcomes
├── evaluate_snake_placement()  → expected damage formula (setback × TILE × p_hit × weight)
├── find_best_snake_to_buy()   → exhaustive search, returns best (head, tail, damage)
├── expectimax_decision()      → EASY bot decision (handicapped)
├── strong_decision()          → STRONG bot decision (training opponent)
├── _leading_opponent()        → helper: opponent furthest ahead
├── _placeable()               → helper: valid + affordable check
├── _catch_offsets()           → helper: dice-range head offsets
├── propose_cheap_trap()       → strategy: short, high catch-chance
├── propose_big_snake()        → strategy: long, max setback
├── propose_lurk()             → strategy: win-denial near goal
└── propose_combo()            → strategy: tail on bomb tile

ai/ppo_agent.py
├── ActorCritic(nn.Module)     → shared-body MLP (body + actor head + critic head)
│   ├── forward()              → (logits, value)
│   ├── act()                  → sample action + log_prob + value (inference)
│   └── evaluate()             → log_probs, values, entropy (training)
├── RolloutBuffer              → stores experience for one rollout
│   └── compute_returns_and_advantages() → GAE (γ=0.99, λ=0.95)
├── SnakesLendersEnv           → pure Python game env (no gymnasium)
│   ├── reset()                → generate_board → encode_state
│   └── step(action)           → do_turn → opp turns → reward → encode_state
├── _compute_reward()          → reward formula (win/loss + shaping)
├── _leading_opponent()        → helper: opponent furthest ahead
├── _try_place()               → helper: valid + affordable check (engine only)
├── _action_to_shop()          → strategy index → exact tiles via ENGINE RULES
├── encode_state()             → 22-dim float32 (all features inline — no Expectimax)
├── train_ppo()                → full training loop (pure PyTorch, continuation-aware)
├── load_ppo_model()           → torch.load main → backup; returns (ckpt, net)
└── ppo_decision()             → encode → net.act → _action_to_shop → shop or None
```

---

## Quick Reference — Constants Cheat Sheet

```
expectimax.py                      ppo_agent.py
─────────────────────────          ──────────────────────────────
TILE_VALUE        = 10.0           N_ACTIONS          = 5
HIT_IN_RANGE      = 0.50           OBS_DIM            = 22
HIT_NEAR          = 0.35           MODEL_PATH         = ai/ppo_model.pt
HIT_FAR           = 0.20           BACKUP_PATH        = ai/ppo_model_backup.pt
WIN_DENIAL_TILE   = 85
WIN_DENIAL_OPP    = 72             PPO hyperparameters (train_ppo):
WIN_DENIAL_MULT   = 1.8              learning_rate    = 3e-4
SABOTAGE_MIN_POS  = 10               n_steps          = 2048
MIN_DAMAGE_TO_BUY = 6.0              batch_size       = 64
SAFETY_BUFFER     = 12               n_epochs         = 10
EASY_SABOTAGE_MIN = 35               gamma            = 0.99
EASY_BUFFER       = 45               lam (GAE λ)      = 0.95
EASY_SKIP_PROB    = 0.35             clip_eps         = 0.2
                                     ent_coef         = 0.03
ActorCritic architecture:           vf_coef          = 0.5
  hidden_size   = 64                grad_clip        = 0.5
  activation    = Tanh              device           = cpu
  body layers   = 2 × Linear(64)
  actor head    = Linear(64, 5)    Libraries used:
  critic head   = Linear(64, 1)      torch (PyTorch)  ← training + inference
                                     numpy            ← obs encoding
                                     (NO stable-baselines3, NO gymnasium)
```
