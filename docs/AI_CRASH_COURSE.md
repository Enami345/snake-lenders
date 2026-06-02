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
import os                           # os.path.exists for model load/save
import numpy as np                  # encode_state → float32 array
from game.models import BoardState, Player
from game.engine import calculate_snake_cost, can_place_snake
import random                       # pick opponent from pool each episode

from ai.expectimax import (
    find_best_snake_to_buy,         # encode_state uses this for best-snake features
    expectimax_decision,            # Easy opponent in training pool
    strong_decision,                # Strong opponent in training pool
    propose_cheap_trap,             # PPO action 1
    propose_big_snake,              # PPO action 2
    propose_lurk,                   # PPO action 3
    propose_combo,                  # PPO action 4
)
# gymnasium / stable_baselines3: imported lazily inside functions
# (only needed during training — not at game startup)
```

Note: `ppo_agent.py` **imports from** `expectimax.py`. The PPO agent reuses
Expectimax's placement strategies as its action implementations. PPO decides
**which strategy** to use; Expectimax functions execute **how**.

### Action space — Discrete(5)

```
N_ACTIONS = 5

Action 0 → return None             (just roll, save points)
Action 1 → propose_cheap_trap()    (cheap pressure)
Action 2 → propose_big_snake()     (max knockback)
Action 3 → propose_lurk()          (win-denial near finish)
Action 4 → propose_combo()         (tail on bomb = knockback + bomb damage)
```

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
| 15 | Best snake cost | `min(cost / 1000, 1.0)` |
| 16 | Best snake damage | `clamp(dmg / 100, −1, 1)` |
| 17 | Can afford best snake | `1.0` or `0.0` |
| 18 | Combo available | `1.0` if `propose_combo()` returns non-None |
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

### PPO update algorithm

```
For each update iteration:
  1. Collect rollouts with 4 parallel envs × 2048 steps each
  2. Compute advantages using GAE (Generalized Advantage Estimation):
       A_t = r_t + γ × V(s_{t+1}) − V(s_t)
       (γ = 0.99 — values future rewards)
  3. For 10 mini-batch epochs:
       Compute ratio r = π_new(a|s) / π_old(a|s)
       Clipped objective: min(r × A, clip(r, 1−ε, 1+ε) × A)
                          where ε = clip_range = 0.2
       Add entropy bonus: +ent_coef × H(π)  (forces exploration)
       Update policy net + value net via Adam (lr=3e-4)
```

The **clipping** `clip(r, 0.8, 1.2)` prevents the update from changing the
policy too much in one step — if the new policy is too different from the old
one, the gradient is cut off. This is what makes PPO stable.

### Training flow

```
train_ppo(total_timesteps, opponent_pool=True):

  build_training_env():
    deciders = [easy_bot, strong_bot]
    if frozen model exists AND shape matches (22,)/5:
        deciders.append(frozen_ppo)    # self-play
    else:
        print "skipping self-play (shape mismatch)"

  if existing model has matching shape:
    model = PPO.load(save_path, env)   # CONTINUE from prior steps
  else:
    model = PPO(MlpPolicy, ...)        # fresh start

  model.learn(total_timesteps, reset_num_timesteps=(steps==0))
  model.save(save_path)
```

**Two-stage training:**
```
Stage 1: train_ppo(3_000_000, opponent_pool=True)
         → pool = {Easy, Strong}  (frozen self-play skipped, old model is 14-dim)
         → saves ai/ppo_model.zip at 3M steps

Stage 2: train_ppo(2_000_000, opponent_pool=True)
         → pool = {Easy, Strong, frozen stage-1 PPO}  (shape now matches)
         → continues from 3M, saves at 5M steps
```

### Inference

```python
ppo_decision(board, player, model):
    obs = encode_state(board, player)    # 22-dim vector
    action, _ = model.predict(obs, deterministic=False)  # STOCHASTIC
    shop = _action_to_shop(board, player, int(action))
    return shop  # dict {head, tail} or None
```

`deterministic=False` → samples from the probability distribution instead of
always picking the single highest-probability action. This makes the agent
**unpredictable** — opponents can't learn its exact timing.

### Full PPO agent pseudocode

```
ALGORITHM: PPO Agent (Hard Mode)
INPUT:  board state, player, trained model
OUTPUT: {head, tail} snake placement OR None

────────────────────────────────────────────────────────────────
INFERENCE (runtime — called every Hard AI turn)

FUNCTION ppo_decision(board, player, model):
  IF player.snake_count >= 3:
    RETURN None

  obs    ← encode_state(board, player)       // 22-dim float32 vector
  action ← model.predict(obs,
              deterministic=False)            // STOCHASTIC sample
  shop   ← _action_to_shop(board, player, action)
  RETURN shop   // {head, tail} or None

FUNCTION _action_to_shop(board, player, action):
  IF action == 0: RETURN None                // roll only
  IF action == 1: RETURN propose_cheap_trap(board, player)
  IF action == 2: RETURN propose_big_snake(board, player)
  IF action == 3: RETURN propose_lurk(board, player)
  IF action == 4: RETURN propose_combo(board, player)

────────────────────────────────────────────────────────────────
STATE ENCODING (called before every prediction)

FUNCTION encode_state(board, player) → float32[22]:
  state[0]    ← player.position / 100
  state[1]    ← min(player.points / 2000, 1.0)
  state[2]    ← player.snake_count / 3

  FOR i, opp IN enumerate(opponents[:3]):
    state[3+i] ← opp.position / 100
    state[6+i] ← min(opp.points / 2000, 1.0)
    state[9+i] ← opp.snake_count / 3

  leader      ← opponent with max position
  state[12]   ← min(snakes_ahead_of_leader / 10, 1.0)
  state[20]   ← (100 - leader.position) / 100

  state[13]   ← min(dist_to_nearest_bomb_ahead / 20, 1.0)   // 1.0 if none
  state[14]   ← min(dist_to_nearest_ladder_ahead / 20, 1.0) // 1.0 if none

  best ← find_best_snake_to_buy(board, player)
  IF best:
    state[15] ← min(cost / 1000, 1.0)
    state[16] ← clamp(damage / 100, -1, 1)
    state[17] ← 1.0 if player.points >= cost else 0.0

  state[18] ← 1.0 if propose_combo(board, player) else 0.0
  state[19] ← 1.0 if player.bankrupt_immune > 0 else 0.0
  state[21] ← min(turn_number / 100, 1.0)
  RETURN state

────────────────────────────────────────────────────────────────
ENVIRONMENT STEP (one full round during training)

FUNCTION step(action):
  start_pos     ← agent.position
  opp_pos_before ← {opp.id: opp.position for each opp}

  // 1. Agent's turn
  shop   ← _action_to_shop(board, agent, action)
  result ← do_turn(board, shop_decision=shop)
  agent_bought   ← result.bought
  placed_setback ← shop.head - shop.tail  if bought else 0
  placed_combo   ← shop.tail IN board.bombs  if bought else False
  winner ← result.winner

  // 2. Opponents' turns (loop until back to agent)
  WHILE winner is None AND board.active_player != agent:
    opp_decision ← opp_decide_fn(board, active_player)  // random from pool
    opp_result   ← do_turn(board, shop_decision=opp_decision)
    winner ← opp_result.winner

  // 3. Measure opponent setback (our sabotage paying off)
  opp_setback ← sum(max(0, opp_pos_before[o.id] - o.position) for each opp)

  // 4. Compute reward
  reward ← _compute_reward(winner, agent_bought, start_pos,
                            went_bankrupt, opp_setback,
                            placed_setback, placed_combo)

  // 5. Next observation
  obs ← encode_state(board, agent)
  RETURN obs, reward, terminated, truncated

────────────────────────────────────────────────────────────────
REWARD FUNCTION

FUNCTION _compute_reward(...):
  // Terminal: win/loss dominates everything
  IF winner == agent:    RETURN +100.0
  IF winner != agent:    RETURN -100.0

  reward ← 0.0
  reward ← reward + (agent.position - start_pos) × 0.1   // progress
  reward ← reward + opp_setback × 0.5                    // sabotage

  IF went_bankrupt:
    reward ← reward - 50.0

  IF agent_bought:
    reward ← reward + 4.0 + placed_setback × 0.3         // deploy
    IF placed_combo:
      reward ← reward + 12.0                              // COMBO BONUS

  ELSE IF agent.points > 100:
    reward ← reward - 1.0                                 // anti-hoard

  RETURN reward

────────────────────────────────────────────────────────────────
PPO POLICY UPDATE (happens after collecting 4 × 2048 steps)

FOR each update iteration:

  // Collect experience
  FOR t = 1 to n_steps (2048) × n_envs (4):
    obs_t   ← encode_state(board, agent)
    action  ← π_old.sample(obs_t)              // stochastic from current policy
    obs_t+1, reward_t, done_t ← step(action)

  // Compute advantages (GAE)
  FOR t = T downto 0:
    delta_t ← reward_t + γ × V(obs_t+1) - V(obs_t)
    A_t     ← delta_t + (γ × λ) × A_t+1       // γ=0.99, λ=0.95

  // Update (10 epochs, batches of 64)
  FOR epoch = 1 to 10:
    FOR each minibatch:
      ratio ← π_new(action|obs) / π_old(action|obs)
      L_clip ← min(ratio × A,
                   clip(ratio, 1-ε, 1+ε) × A)   // ε=0.2
      L_entropy ← ent_coef × H(π_new)           // = 0.03 × entropy
      L_value ← (V(obs) - target)²

      loss ← -L_clip - L_entropy + 0.5 × L_value
      gradient_step(loss, lr=3e-4)               // Adam optimizer

────────────────────────────────────────────────────────────────
TRAINING PIPELINE

FUNCTION train_ppo(total_timesteps, opponent_pool):

  // Build opponent pool
  deciders ← [expectimax_decision, strong_decision]
  IF opponent_pool AND frozen_model_exists AND shape_matches:
    deciders.append(frozen_ppo_decision)         // self-play

  // Load or start fresh
  IF existing_model AND shape_matches (22-dim / 5-action):
    model ← PPO.load(save_path)                  // CONTINUE
  ELSE:
    model ← PPO(MlpPolicy, lr=3e-4, clip=0.2,
                ent_coef=0.03, device=cpu, ...)  // FRESH

  // Silence game logs, train, restore logs
  log.VERBOSE ← False
  model.learn(total_timesteps,
              reset_num_timesteps=(model.steps == 0))
  log.VERBOSE ← True

  model.save(save_path)
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
  1. encode_state() → 22-dim float vector
  2. model.predict(obs) → sample action from neural net distribution
  3. _action_to_shop(action) → propose_cheap/big/lurk/combo or None
  4. Return shop or None.
```

### Knowledge source

| | Expectimax | PPO |
|---|---|---|
| **Who encodes knowledge** | Programmer (hand-tuned rules) | The game itself (via reward signal) |
| **Adapts to opponents** | No — fixed heuristic | Partly — trained vs varied pool |
| **Exploits board patterns** | Only what programmer coded | Any pattern that helps win |
| **Knows about combo?** | Yes — `propose_combo` is a function | Yes — learned combo = biggest reward signal |

### Why PPO beats Expectimax

1. **Timing** — Expectimax has a hard trigger (tile 35). PPO learns the
   economically optimal timing from thousands of games.
2. **Strategy mix** — Expectimax only uses cheap traps. PPO mixes all 5
   strategies based on what the board actually calls for.
3. **Anti-hoard** — The `-1.0` penalty in the reward function explicitly
   punishes Expectimax-style hoarding. PPO learns to spend.
4. **Combo exploitation** — `+12.0` reward makes PPO obsessively place
   tail-on-bomb snakes. Expectimax never plays this.
5. **Unpredictability** — Stochastic inference makes PPO hard to play around.

### Why Expectimax is still useful

1. **No training needed** — instant, zero setup.
2. **Transparent** — every decision is traceable through a formula.
3. **Training opponent** — its placement helpers (`propose_*`) are reused
   directly by PPO as action implementations.
4. **Beatable baseline** — validates that Hard AI is actually better.

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
├── N_ACTIONS = 5              → action space size
├── _action_to_shop()          → maps action index → placement function
├── encode_state()             → board state → 22-dim float32 vector
├── _obs_for_model()           → legacy obs adapter (shape mismatch handling)
├── build_training_env()       → Gymnasium env class factory + opponent pool
│   └── SnakesLendersEnv
│       ├── reset()            → new game, pick random opponent from pool
│       ├── step()             → agent turn + opponent turns + reward
│       └── _compute_reward()  → reward formula (win + shaping)
├── train_ppo()                → full training pipeline (continuation-aware)
├── load_ppo_model()           → load main → backup (resilient)
└── ppo_decision()             → inference: encode → predict → shop or None
```

---

## Quick Reference — Constants Cheat Sheet

```
expectimax.py                      ppo_agent.py
─────────────────────────          ──────────────────────────────
TILE_VALUE        = 10.0           N_ACTIONS          = 5
HIT_IN_RANGE      = 0.50           MODEL_PATH         = ai/ppo_model.zip
HIT_NEAR          = 0.35           BACKUP_PATH        = ai/ppo_model_backup.zip
HIT_FAR           = 0.20
WIN_DENIAL_TILE   = 85             PPO hyperparameters (train_ppo):
WIN_DENIAL_OPP    = 72               learning_rate    = 3e-4
WIN_DENIAL_MULT   = 1.8              n_steps          = 2048
SABOTAGE_MIN_POS  = 10               batch_size       = 64
MIN_DAMAGE_TO_BUY = 6.0              n_epochs         = 10
SAFETY_BUFFER     = 12               gamma            = 0.99
EASY_SABOTAGE_MIN = 35               clip_range       = 0.2
EASY_BUFFER       = 45               ent_coef         = 0.03
EASY_SKIP_PROB    = 0.35             device           = cpu
                                     n_envs           = 4
```
