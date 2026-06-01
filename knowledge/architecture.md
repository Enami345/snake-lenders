# Snakes & Lenders — Architecture & Code Map

---

## Data Models (`game/models.py`)

**Purpose:** Pure dataclasses — no game logic. Single source of truth for all
state. Every other module imports from here; nothing else owns state.

```python
Snake(head, tail, owner_id)   # owner_id = -1 board terrain, >=0 player trap
Ladder(bottom, top)
Player(player_id, name, position=0, points=0, snakes_owned=[], is_ai,
       ai_difficulty, bankrupt_count, bankrupt_immune)
BoardState(tiles, ladders, snakes, bombs, players, current_turn, turn_number)
```

**Key constants / logic:**

| Symbol | Value | Description |
|---|---|---|
| `BANKRUPT_IMMUNITY_TURNS` | 6 | Turns of post-bankruptcy immunity |

**Bankruptcy flow:**
1. `Player.deduct_points(amount)` — subtracts from wallet.
   - If wallet < 0 **and** `bankrupt_immune > 0`: clamp to 0, log 🛡️ (no reset).
   - If wallet < 0 **and** not immune: call `go_bankrupt()`.
2. `Player.go_bankrupt()`:
   ```
   position      ← 0       (back to off-board start)
   points        ← 0
   bankrupt_count ← +1
   bankrupt_immune ← BANKRUPT_IMMUNITY_TURNS   (starts immunity window)
   ```
3. `bankrupt_immune` ticked down once per player's own turn in `engine.do_turn`.

`BoardState.active_player`, `next_turn()` — generic over N players.
`get_snake_at` / `get_ladder_at` = exact-tile lookups (head / bottom).

---

## Board Generation (`game/board.py`)

**Algorithm: Randomized BFS-validated placement**

```
generate_board(seed, players):
  rng ← Random(seed)
  attempt ≤ 200 times:
    tiles   ← generate_tile_values(rng)     # scarce income distribution
    ladders ← _place_ladders(rng)           # 7 ladders, 5-20 tile jump
    snakes  ← _place_snakes(rng, forbidden) # 4 board terrain snakes
    bombs   ← _place_bombs(rng, forbidden)  # 5 bomb tiles
    if bfs_expected_turns(ladders, snakes) >= MIN_AVG_TURNS:
      return BoardState(...)
  raise RuntimeError("no valid board in 200 attempts")
```

**BFS validation** — `bfs_expected_turns(ladders, snakes)`:
- Runs BFS over the 100-tile Markov chain.
- Each state = tile; transitions = uniform dice 1-6, then ladder/snake rules.
- Computes expected turns to reach tile 100 from tile 0.
- Rejects boards where expected turns < `MIN_AVG_TURNS=10` (too trivial).

**Board constants:**

| Constant | Value | Description |
|---|---|---|
| `NUM_LADDERS` | 7 | Ladders per board |
| `NUM_INIT_SNAKES` | 4 | Board terrain snakes (permanent) |
| `NUM_BOMBS` | 5 | Bomb tiles |
| `MIN_AVG_TURNS` | 10 | BFS min expected turns (solvability gate) |
| `MIN_LADDER_GAP` | 6 | Min tile spacing between any two ladder endpoints |
| `BASE_TILE_VALUE` | 3 | Base income per tile (scarce by design) |

**Tile income formula:**
```
tile_value(tile) = BASE_TILE_VALUE + depth_bonus(tile)
                 ≈ 4–14 pts/turn (deeper tiles pay more)
```

**Ladder placement rules:**
- Jump range: 5–20 tiles (no absurd shortcuts, no trivial hops).
- All endpoints spaced ≥ `MIN_LADDER_GAP=6` from each other (anti-clutter).
- Ladder top must not be a snake head (loop check → re-seed if invalid).
- Note: no explicit tile 1–10 exclusion in code (diagram is inaccurate here).

---

## Game Engine (`game/engine.py`)

**Purpose:** All game rules. Single source of truth. No rule logic anywhere else.

### Snake pricing formula

```
cost = max(BASE_SNAKE_PRICE × purchase_count × length^PRICE_ALPHA, MIN_SNAKE_COST)
```

| Constant | Value | Description |
|---|---|---|
| `BASE_SNAKE_PRICE` | 2 | Multiplier base |
| `PRICE_ALPHA` | 0.9 | Sub-linear exponent (long snakes stay affordable) |
| `MIN_SNAKE_COST` | 12 | Floor — shortest snakes always cost ≥12 |
| `MAX_SNAKE_HEAD` | 90 | Max tile for player snake heads |
| `STRIKE_ZONE` | 0 | Exact-head only (0 = no zone, only direct landing) |
| `MAX_HEAD_RUN` | 2 | Max adjacent snake heads before placement blocked |
| `BOMB_BASE` | 18 | Base bomb damage |
| `BOMB_DEPTH` | 6 | Damage per tile of depth (deeper bombs hit harder) |
| `STEAL_FLAT` | 15 | Flat points stolen per bite |
| `STEAL_PCT` | 0.30 | Fraction of victim's remaining wallet also stolen |

**Bomb damage formula:**
```
bomb_damage(position) = BOMB_BASE + (position // BOMB_DEPTH)
```
Deeper tiles = heavier bomb penalty. Can bankrupt (wallet goes negative → `go_bankrupt()`).

### Point-stealing formula (player-owned snake bites only)

```
amount = STEAL_FLAT + int(victim.points × STEAL_PCT)
taken  = min(amount, victim.points)   # can't steal more than victim has
victim.points -= taken
owner.points  += taken
```
Board terrain snakes (`owner_id < 0`) do **not** steal.

### Placement validation (`can_place_snake`)

All must pass:
1. `tail < head` (direction valid)
2. `head ≤ MAX_SNAKE_HEAD` (cap)
3. Neither tile occupied by a player
4. Neither tile is a ladder endpoint
5. No chain: head not sitting on another snake's tail
6. No wall: `_head_run_length(head) ≤ MAX_HEAD_RUN` (max 2 adjacent heads)
7. Player can afford the cost

### Turn loop (`do_turn`)

```
do_turn(board, shop_decision):
  player ← board.active_player
  tick bankrupt_immune down (if > 0)
  if shop_decision and can_buy_snake and position > 0:
    buy_snake(head, tail)               # deduct cost, add to board
  roll ← random.randint(1, 6)
  new_pos ← position + roll
  if new_pos > 100: stay put            # exact roll to win
  else:
    move to new_pos
    collect tile points
    apply ladders (climb if on bottom)
    apply snakes  (slide + steal if on head)
    apply bomb    (deduct scaled damage, may bankrupt)
  check winner (position == 100)
  board.next_turn()
  return {logs, winner, bought, move}
```

---

## Easy AI (`ai/expectimax.py`)

**Algorithm: Expectimax (deliberately weakened)**

Expectimax evaluates actions by building a probability tree:
- **Decision nodes** (AI's choice): pick the action with max expected value.
- **Chance nodes** (dice outcomes): weight each outcome 1/6, sum expected values.

```
CalculateBestMove(board_state):
  for each action in [roll, buy_snake_A, buy_snake_B, ...]:
    EV ← expected_value_after_roll(board_state_after_action)
    where:
      expected_value_after_roll = Σ (1/6) × evaluate_state(result_of_roll_i)
                                   i=1..6
  return action with highest EV
```

**Deliberate weaknesses (beatable baseline):**

| Parameter | Value | Effect |
|---|---|---|
| `EASY_SABOTAGE_MIN_POS` | 35 | Won't attack until opponent passes tile 35 |
| `EASY_SKIP_PROB` | 0.35 | 35% random chance to skip buying even when it should |
| `EASY_BUFFER` | 45 | Won't buy if remaining points < 45 (over-hoards) |

**Placement strategies (shared with PPO):**

| Function | Description | Algorithm |
|---|---|---|
| `propose_cheap_trap` | Cheapest affordable trap on leading opponent | Catch-optimal: head in dice range offset `STRIKE_ZONE+1..6` from opponent |
| `propose_big_snake` | Longest affordable snake with max setback | Tries tail=1 first for maximum knockback |
| `propose_lurk` | Win-denial: head near finish line (tiles 85–90) | Catch-optimal relative to opponent position |
| `propose_combo` | Snake whose **tail lands on a bomb tile** | Knockback + bomb damage stacked; highest-setback combo wins |

**Catch-optimal offset:**
```
_catch_offsets() = range(STRIKE_ZONE+1, 7)  # e.g. STRIKE_ZONE=0 → 1,2,3,4,5,6
head = opponent.position + offset
```
Places head exactly within the opponent's next-roll dice range.

**Damage model (`evaluate_snake_placement`):**
```
value = setback × TILE_VALUE × p_hit × progress_weight + win_denial_bonus
where:
  p_hit            = 1/6 per tile in strike zone
  progress_weight  = opponent.position / 100
  win_denial_bonus = large constant if head near tile 100
```

---

## Hard AI (`ai/ppo_agent.py`)

**Algorithm: PPO (Proximal Policy Optimization) — stable-baselines3**

PPO is an on-policy actor-critic RL algorithm. Key properties:
- **Clipping** (`clip_range=0.2`): limits policy update size each step → stable training, no catastrophic forgetting.
- **MlpPolicy**: fully connected neural network (not CNN) — appropriate for flat vector obs.
- **Entropy bonus** (`ent_coef=0.03`): adds a small penalty for being too deterministic → forces the policy to explore and mix strategies.

**Training loop:**
```
For N total_timesteps:
  collect rollouts (4 parallel envs × n_steps=2048)
  compute advantages using GAE(γ=0.99)
  for 10 epochs:
    update policy + value net via clipped surrogate objective
    KL divergence monitored (~0.01 target)
```

**Observation space — 22-dim vector:**

| Index | Feature | Normalization |
|---|---|---|
| 0 | Agent position | ÷ 100 |
| 1 | Agent points | ÷ 2000 (capped) |
| 2 | Agent snake count | ÷ 3 |
| 3–5 | Opponent positions (up to 3) | ÷ 100 each |
| 6–8 | Opponent points | ÷ 2000 each |
| 9–11 | Opponent snake counts | ÷ 3 each |
| 12 | Snakes ahead of leading opponent | ÷ 10 |
| 13 | Distance to nearest bomb ahead | ÷ 20 (1.0 if none) |
| 14 | Distance to nearest ladder ahead | ÷ 20 (1.0 if none) |
| 15 | Best available snake cost | ÷ 1000 |
| 16 | Best available snake damage | ÷ 100 (clamped ±1) |
| 17 | Can afford best snake | 0 or 1 |
| 18 | Combo snake available (tail-on-bomb) | 0 or 1 |
| 19 | In post-bankruptcy immunity | 0 or 1 |
| 20 | Leading opponent's distance to goal | ÷ 100 |
| 21 | Turn number | ÷ 100 |

**Action space — Discrete(5):**

| Action | Strategy | Implementation |
|---|---|---|
| 0 | Roll only (save points, be patient) | `None` |
| 1 | Cheap pressure trap | `propose_cheap_trap` |
| 2 | Big knockback snake (save up) | `propose_big_snake` |
| 3 | Win-denial lurk near finish | `propose_lurk` |
| 4 | Combo (tail-on-bomb) | `propose_combo` |

**Reward function:**
```
R = 0

# Terminal
if winner == agent:   R ← +100
if winner == opponent: R ← −100

# Shaping (non-terminal)
R += (agent.position − start_position) × 0.1      # progress
R += opponent_setback × 0.5                         # heavy sabotage reward
if went_bankrupt:  R −= 50.0                        # bankruptcy penalty
if agent_bought:
    R += 4.0 + placed_setback × 0.3               # deploy reward
    if placed_combo:  R += 12.0                    # MAXIMUM annoyance bonus
else:
    if agent.points > 100:  R −= 1.0              # anti-hoard penalty
```

**PPO hyperparameters:**

| Param | Value | Reason |
|---|---|---|
| `learning_rate` | 3e-4 | Standard PPO LR |
| `n_steps` | 2048 | Steps per rollout per env |
| `batch_size` | 64 | Mini-batch for gradient update |
| `n_epochs` | 10 | PPO epochs per update |
| `gamma` | 0.99 | Discount factor (future rewards valued) |
| `clip_range` | 0.2 | PPO clipping — keeps updates stable |
| `ent_coef` | 0.03 | Entropy → unpredictable, varied policy |
| `device` | cpu | Faster than GPU for small MLP |
| `n_envs` | 4 | Parallel environments |

**Training stages:**
```
Stage 1 (base, 3M steps):
  opponent_pool = {Easy, Strong}
  frozen self-play: skipped if existing model shape ≠ (22, 5)

Stage 2 (self-play polish, 2M steps, true continuation):
  load existing model via PPO.load(..., env=vec_env)
  learn(reset_num_timesteps=False)   # step counter accumulates
  opponent_pool = {Easy, Strong, frozen stage-1 PPO}
```

**Inference:** `deterministic=False` → stochastic sampling → unpredictable timing.

---

## Web Server (`server.py`)

**Architecture: Stateful Flask + engine-driven thin client**

```
Browser (web/)                     Server (server.py)           Engine (game/)
    |                                     |                           |
    |── POST /api/new ──────────────────→ |── generate_board ────────→|
    |← state ─────────────────────────── |                           |
    |── POST /api/turn ────────────────→ |── do_turn ───────────────→|
    |                                     |   (AI: ppo_decision or    |
    |                                     |    expectimax_decision)   |
    |← {logs, move, state} ──────────── |                           |
    |── POST /api/buy ─────────────────→ |── buy_snake ─────────────→|
    |── GET  /api/shop-options ────────→ |── can_place_snake ────────→|
```

No game logic in JavaScript. All rules, costs, validity, movement run server-side.
PPO inference wrapped in try/except per turn — falls back to Expectimax if it raises.

**REST API endpoints:**

| Method | Path | Action |
|---|---|---|
| POST | `/api/new` | Start game: players list + optional seed |
| GET | `/api/state` | Full board state snapshot |
| GET | `/api/shop-options` | Valid+affordable placements for active human |
| POST | `/api/buy` | `{head, tail}` → `buy_snake` |
| POST | `/api/turn` | Roll + move (AI auto-decides shop) |
| POST | `/api/quit` | Reset session |
| POST | `/api/generate-board` | Lobby board preview (stateless) |

---

## Session Management (`web/app.js` + `server.py`)

```
gameSession = {
  players: [...],           # current positions, points, snake counts
  current_turn: int,        # whose turn
  winner: str | null,
  stats: {                  # per-player counters (tracked in JS from engine logs)
    snakesPlaced, bittenCount, bankruptCount,
    laddersClimbed, bombsHit, turnsTaken, pointsStolen
  }
}
```

Session held in memory on the server (`Session` class). One game at a time.
Stats parsed from engine log strings each turn (🛒 🐍 ☠️ 🪜 💣 💸 patterns).

---

## Logging (`game/log.py`)

`VERBOSE` flag + `gprint(msg)`. All gameplay prints (board gen, AI buys,
bankruptcy, steal, ladder, bomb) route through `gprint`. `train_ppo` flips
`VERBOSE=False` during `model.learn` so only SB3 progress tables print.
Console-game UI prints use plain `print` (always visible).

---

## Tests (`tests/test_core.py`)

14 unittest cases covering:
- Board generation (counts, BFS solvability)
- Snake cost (monotonic, floor, rises per purchase)
- Movement (overshoot stay-put, exact-roll win, entry ladder climb)
- Strike zone (exact-head bite, clean-jump safe, owner immunity, board snake)
- Placement rules (head cap, no-wall, no-chain)
- Bomb bankruptcy (resets to tile 0)

Run: `python -m unittest tests.test_core`
