# Snakes & Lenders — Architecture & Code Map

## Data Models (`game/models.py`)

```python
Snake(head, tail, owner_id)      # head > tail always; owner_id=-1 means board snake
Ladder(bottom, top)              # bottom < top always
Player(player_id, name, position=0, points=0, snakes_owned=[], is_ai, ai_difficulty)
BoardState(tiles, ladders, snakes, bombs, players, current_turn, turn_number)
```

Key Player methods:
- `add_points(amount)` — adds points
- `deduct_points(amount)` — deducts; calls `go_bankrupt()` if below 0
- `go_bankrupt()` — resets position=0, points=0, increments bankrupt_count
- `can_buy_snake` — property, True if snake_count < 3

Key BoardState methods:
- `get_snake_at(tile)` — returns Snake if tile is snake head
- `get_ladder_at(tile)` — returns Ladder if tile is ladder bottom
- `occupied_tiles()` — set of all player positions (excluding tile 0)
- `active_player` — player whose turn it is
- `next_turn()` — advances current_turn, increments turn_number at wrap

---

## Board Generator (`game/board.py`)

Constants:
```python
NUM_TILES = 100, NUM_LADDERS = 7, NUM_INIT_SNAKES = 4, NUM_BOMBS = 5
MIN_AVG_TURNS = 10, MAX_RETRIES = 200, BOMB_DEDUCTION = 30, BASE_TILE_VALUE = 10
```

Key functions:
- `generate_board(seed, players)` → BoardState — randomizes with seed, retries up to 200x
- `bfs_expected_turns(ladders, snakes)` → float — BFS/value-iteration to compute expected turns from tile 1
- `_board_is_valid(ladders, snakes)` — checks: no chained ladders, no chained snakes, no ladder top = snake head, expected >= 10
- `generate_tile_values(seed)` — tile values = base + (row * 5) + variance; tile 100 = 200
- `print_board(board)` — ASCII grid output

---

## Game Engine (`game/engine.py`)

Key functions:
- `roll_dice()` → 1–6
- `calculate_snake_cost(player, head, tail)` → int — exponential: `max(BASE(5) * purchase_count * length^ALPHA(1.3), 30)`
- `can_place_snake(board, buyer, head, tail)` → (bool, reason) — validates all constraints
- `move_player(board, player, roll)` → logs list — overshoot=stay put (exact roll to win), ladders, snakes, bomb
- `buy_snake(board, buyer, head, tail)` → (bool, msg) — validates, deducts points, adds snake
- `check_winner(board)` → Player | None
- `do_turn(board, shop_decision)` → dict{logs, winner, bought} — full turn: shop → roll → move → win check

**Movement order**: overshoot=stay put → land new_pos → ladder (exact) → snake (exact) → tile points → bomb check

**Triggers:** ladders = exact landing. Snakes:
- Player-placed (`owner_id >= 0`) = **strike range** `_striking_snake`: bite on head or `STRIKE_ZONE`(2) tiles below; **single-use** (`_consume_snake` frees a slot for re-placement). ~50% catch on a passing opponent.
- Board snakes (`owner_id == -1`) = exact-head terrain, persistent.
- Jumping clean over a head is safe; slides always move down (loop terminates). No softlock (pass-over was tried + reverted — long snakes walled the board).

**Economy constants** (`engine.py`): `BASE_SNAKE_PRICE=2`, `PRICE_ALPHA=0.9` (sub-linear so long snakes are affordable to save for), `MIN_SNAKE_COST=12`, `MAX_SNAKE_HEAD=90`, `MAX_HEAD_RUN=2` (anti-wall), `BOMB_BASE=18`+`pos//BOMB_DEPTH(6)` (depth-scaled, can bankrupt). Tile income low (`board.py` `BASE_TILE_VALUE=3`, ~4-14/turn).

---

## AI — Expectimax (`ai/expectimax.py`)

Easy AI. No training needed. Plays a **cunning saboteur** strategy.

- `evaluate_snake_placement(board, player, head, tail)` → float — expected DAMAGE in points: `setback × TILE_VALUE × p_hit × progress_weight`. `p_hit` by distance ahead (`HIT_IN_RANGE`/`HIT_NEAR`/`HIT_FAR`); `progress_weight` favors hitting advanced opponents.
- `find_best_snake_to_buy(board, player)` → (head, tail, damage) | None — searches heads 1-15 ahead of opponents, trying max-setback tails (incl. tail=1) for brutal knockback.
- `expectimax_decision(board, player)` → dict | None — entry point. Holds budget until the leading opponent passes `SABOTAGE_MIN_POS` (15), keeps a bomb-survival `SAFETY_BUFFER` (20), then spends surplus on the highest-damage affordable snake (`MIN_DAMAGE_TO_BUY` 10). Saves for long, devastating, well-placed traps rather than spamming.
- Strike-hit estimates: `HIT_IN_RANGE`(0.5)/`HIT_NEAR`(0.35)/`HIT_FAR`(0.2).
- `evaluate_state` / `expected_value_after_roll` — legacy depth-1 EV helpers, no longer used by the decision.

Annoyance layer: snake bites **steal points** (`_steal_points`, engine) to the owner; AI emphasizes **win-denial** snakes near the goal (`WIN_DENIAL_*`) and plays aggressively (low `SAFETY_BUFFER`/thresholds, self-funded by steals).

Validated: strategic AI beats roll-only **57-59%** over 200 games; AI-vs-AI ~50% (balanced); ~0.78 bankruptcies/game; ~60-turn games. Economy load-bearing + annoying.

⚠️ **PPO is NOT yet stronger than Expectimax** — its action space is Discrete(2) [roll / buy-best] and it delegates placement to `find_best_snake_to_buy` (the Expectimax heuristic). It only learns buy-timing. To make "Hard" a real skill gap, expand the action space (choose snake type/aggression) + retrain vs the cunning Expectimax.

---

## AI — PPO Agent (`ai/ppo_agent.py`)

Hard AI. Requires trained model at `ai/ppo_model.zip`.

State vector (14 floats, all normalized 0–1):
```
[0]    our position / 100
[1]    our points / 2000 (capped)
[2]    our snake count / 3
[3-5]  opponent positions / 100 (up to 3)
[6-8]  opponent points / 2000
[9]    snakes ahead of nearest opponent / 10
[10]   best snake cost / 1000
[11]   best snake ROI / 100 (clamped -1..1)
[12]   turn number / 100
[13]   can afford best snake? (0 or 1)
```

Action space: Discrete(2) — 0=just roll, 1=buy best snake

Reward: +100 win, -100 lose, +0.1×position progress, -50 bankruptcy, +10 snake placed

Training: `train_ppo(total_timesteps=100_000)` — uses 4 parallel envs, MlpPolicy, lr=3e-4

Key functions:
- `encode_state(board, player)` → np.ndarray(14)
- `train_ppo(total_timesteps)` — trains and saves to `ai/ppo_model.zip`
- `load_ppo_model()` → PPO model — raises FileNotFoundError if not trained
- `ppo_decision(board, player, model)` → dict | None — deterministic inference

**Fallback**: If ppo_model.zip not found, Hard AI falls back to Expectimax automatically.

---

## UI Renderer (`ui/renderer.py`)

Pygame window: 1000×700px. Board = 700×700 (10×10, 70px/tile). Side panel = 300px.

Key class: `GameRenderer`
- `run(board, mode, ppo_model)` — main game loop
- `_human_shop(board, player)` — interactive shop: B to open, type head/tail, B to confirm
- `_get_number_input(board, prompt, min_val, max_val)` — keyboard number input with ESC cancel
- `wait_for_space(board, message)` — blocks until SPACE or Q

Board color coding:
- Yellow/blue checkerboard (base)
- Red = snake head, Brown = snake tail
- Green = ladder bottom, Dark green = ladder top
- Dark grey = bomb, Gold = tile 100

Tile-to-screen conversion: `tile_to_screen(tile)` — handles snake-pattern rows (even rows L→R, odd rows R→L), bottom-to-top display.

---

## Entry Point (`main.py`)

Argparse flags: `--phase`, `--mode [hvh|hvai|aivai]`, `--hard`, `--train`, `--steps`, `--seed`, `--players [2|3|4]`, `--console`

Flow:
1. `--train` → `train_ppo()` and exit
2. `--phase 1` → generate board, print, show BFS expected turns, exit
3. Build player list based on mode
4. Load PPO model if any hard AI player
5. Generate board
6. `--console` → `play_game()`, else `GameRenderer().run()`
