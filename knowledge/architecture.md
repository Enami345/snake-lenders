# Snakes & Lenders — Architecture & Code Map

## Data Models (`game/models.py`)

```python
Snake(head, tail, owner_id)   # owner_id = -1 board terrain, >=0 player trap
Ladder(bottom, top)
Player(player_id, name, position=0, points=0, snakes_owned=[], is_ai, ai_difficulty, bankrupt_count)
BoardState(tiles, ladders, snakes, bombs, players, current_turn, turn_number)
```
- `Player.go_bankrupt()` → position=0, points=0, bankrupt_count+1 (reset to start; logs via `gprint`).
- `BoardState.active_player`, `next_turn()` — generic over N players.
- `get_snake_at`/`get_ladder_at` = exact-tile lookups (head / bottom).

## Board + Economy (`game/board.py`)

- `generate_board(seed, players)` — randomized, BFS-validated (`bfs_expected_turns >= 10`), retries ≤200.
- `generate_tile_values` — **scarce** income (`BASE_TILE_VALUE=3`, ~4-14/turn).
- `_place_ladders` — jump 5–20 tiles, endpoints spaced ≥ `MIN_LADDER_GAP=6`
  (anti-clutter, well distributed across the board).
- Constants: 7 ladders, 4 board snakes, 5 bombs.
- Uses `gprint` (silent during training).

## Engine (`game/engine.py`) — the rules

Economy / snake constants:
`BASE_SNAKE_PRICE=2`, `PRICE_ALPHA=0.9` (sub-linear), `MIN_SNAKE_COST=12`,
`MAX_SNAKE_HEAD=90`, `STRIKE_ZONE=0` (exact-head only), `MAX_HEAD_RUN=2`,
`BOMB_BASE=18`/`BOMB_DEPTH=6`. (Point-stealing was removed.)

Key functions:
- `calculate_snake_cost` — `max(BASE * purchase_count * length^ALPHA, MIN)`.
- `can_place_snake` — validates: head>tail, head ≤ MAX_SNAKE_HEAD, not occupied/
  ladder, no chain, **no wall** (`_head_run_length` ≤ MAX_HEAD_RUN), affordable.
- `move_player` — overshoot=stay put; entry + normal share one path (entry
  **does** apply ladders/snakes); collect points; bomb (depth-scaled, can bankrupt);
  returns a `move` breakdown {from, landing, final, roll} for UI animation.
- `_striking_snake(board, tile, mover_id)` — bite if landed within `STRIKE_ZONE`
  tiles below head (0 = exact head only); **owner immune**; highest head wins.
- `_apply_snakes` — slides to tail, `_consume_snake` (player snakes single-use).
  No point theft. Loop terminates (always moves down).
- `_apply_ladders` — exact climb.
- `buy_snake`, `do_turn(board, shop_decision)` → {logs, winner, bought, move}.

## Easy AI (`ai/expectimax.py`) — deliberately WEAK

- `expectimax_decision` = Easy bot: reacts only past `EASY_SABOTAGE_MIN_POS=35`,
  `EASY_SKIP_PROB=0.35` hesitation, only cheap short traps, over-hoards
  (`EASY_BUFFER=45`). Beatable baseline.
- Placement strategies (also used by PPO): `propose_cheap_trap`, `propose_big_snake`,
  `propose_lurk` — all **catch-optimal** (`_catch_offsets` = STRIKE_ZONE+1..6, so the
  strike zone lands inside the target's dice range).
- `evaluate_snake_placement`, `find_best_snake_to_buy` — damage model
  (`setback × TILE_VALUE × p_hit × progress_weight`, win-denial bonus).
- `evaluate_state` / `expected_value_after_roll` — EV/chance-node helpers kept to
  document the Expectimax concept (not used by the live decision).

## Hard AI (`ai/ppo_agent.py`) — PPO

- `encode_state` → 14-dim obs.
- Env: **Discrete(4)** actions (`_action_to_shop`): roll / cheap trap / big snake /
  win-denial lurk. `step()` = agent turn then opponents play their own Expectimax;
  reward = ±100 win/loss + bounded progress delta + opponent-setback + bankruptcy
  penalty. Trained vs the weak Easy bot.
- `train_ppo(total_timesteps, opponent_pool, save_path)` — 4 parallel envs;
  `opponent_pool=True` trains vs {Easy, Strong heuristic, frozen best PPO}.
- `load_ppo_model` — tries `MODEL_PATH` then `BACKUP_PATH` (survives mid-write
  training); `ppo_decision` deterministic inference.
- **Shipped model: exact-head-trained, 2.5M, self-play pool.** Proven **> Easy
  (~54-56%)**; doesn't dominate (exact-head snakes fire ~1/6 → dice cap ~57%).
  Skill shows in behavior: Hard avg snake length ~53 (+ win-denial lurks) vs
  Easy ~7.5. (Historical: an interim strike-range rule hit ~89-94%.) See `training.md`.

## Setup + Entry (`main.py`)

- Flags: `--players --humans --hard-ais --difficulty --web --console --train --steps --seed --phase`.
- `resolve_setup` → `(n_players, n_humans, n_hard)`; `build_players` makes humans +
  `n_hard` Hard AIs + rest Easy (**mixed**), **shuffled**, ids 0..N-1 →
  `load_ppo_if_needed` (graceful fallback) → board → run.
- `--hard-ais N` sets how many AIs are Hard; `--difficulty` is the all/none shortcut.
- `--web` → `ui.web.server.run_server` (threaded); else Pygame; `--console` → terminal.
- Forces UTF-8 stdout (Windows emoji/arrow logs).

## Web UI (`ui/web/`) — PRIMARY

- `server.py` — stdlib **`ThreadingHTTPServer`** (single-thread deadlocks browser
  keep-alive connections). In-memory `Session`. Endpoints: `GET /`, `/app.js`,
  `/style.css`; `GET /api/state`; `POST /api/new` (players/humans/**hard_ais**),
  `/api/buy`, `/api/turn` (advance; AI auto-decides; returns `move` for animation),
  `/api/cost`, `/api/quit` (reset session). All POSTs wrapped in try/except →
  500 + traceback (no silent hangs).
- `index.html` — **title page → setup → loading overlay → board**.
- `app.js` — screen flow, staged loading bar, board grid (boustrophedon), tokens,
  **per-step token animation** (walk dice then ladder/snake jump), shop, log;
  **resumes** an in-progress game on refresh; New game calls `/api/quit`.
- `style.css` — minimal placeholder (teammate redesigns), incl. overlay + bar.

## Logging (`game/log.py`)

`VERBOSE` + `gprint`. Gameplay prints (board gen, AI buys, bankruptcy) route
through it; `train_ppo` sets `VERBOSE=False` during `model.learn`. Console-game UI
prints stay on plain `print`.

## Tests (`tests/test_core.py`)

14 unittest cases: board gen, snake cost, overshoot/exact-win, entry ladder climb,
strike range, clean-jump-safe, owner immunity, board-snake exact-head, placement
rules (cap/wall/chain), bomb bankruptcy. Run: `python -m unittest tests.test_core`.
