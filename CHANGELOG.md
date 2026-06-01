# Changes — `refactor/economy-ai-overhaul` branch (for the team)

Heads up team — this branch ships a major AI rebuild + web UI polish on top of
the earlier gameplay/web overhaul. Read this before you pull.

## Newest — Web UI polish + bug fixes

- **CSS root bug fixed** — an unclosed `@media (max-width: 480px)` was silently
  trapping every CSS rule after line 1446 (confirm-snake overrides, vstat-card
  styles, replay rules, animations) so they never applied on desktop. Closed it.
- **Confirm Snake modal** — floats at bottom-center, transparent backdrop (board
  stays visible), compact 320px card with thin divider rows and stacked buttons.
  `✦ Confirm Snake` / `✓ Buy & Place` / `↺ Pick again` icon labels.
- **Replay bar** — removed duplicate rule block overriding the full-width bar
  with a narrow 420px card. Now `position:fixed bottom:24px` centered — no
  more layout bleed / "black half-panel."
- **Victory stats** — all stats (bittenCount, bankruptCount, laddersClimbed,
  bombsHit, turnsTaken, pointsStolen) now tracked by parsing engine turn logs
  each turn. Previously only human snakesPlaced incremented; all others were 0.
- **No-cache server** — `server.py` now sends `Cache-Control: no-store` on all
  responses so CSS/JS edits land on a normal browser reload.
- **In-game rules updated** — steal mechanic, 6-turn immunity, Hard AI 5-strategy
  description added to Quick Rules (lobby) and How to Play modal.

## Previous — Cunning PPO rebuild

The 14-dim PPO was "too reserved." Rebuilt it as a **22-dim / 5-action cunning
agent** with restored point-stealing and a bankruptcy-immunity cooldown.

- **PPO observation 14 → 22 dim**: added bomb/ladder lookahead, leader's
  distance to goal, combo availability flag, bankruptcy-immunity flag.
- **PPO actions 4 → 5**: added **combo** — places a snake whose tail lands on a
  bomb tile so the victim eats knockback + bomb damage. Action mix is
  dominated by combo + lurk; agent uses all 5 strategies.
- **Stochastic inference** (`deterministic=False`) + **`ent_coef=0.03`** —
  unpredictable, varied policy instead of one repeated move.
- **Sabotage-heavy reward shaping**: heavy opponent-setback reward, big combo
  bonus, **anti-hoard penalty** if the agent sits on >100 points without
  buying. Kills the "one late snake" passive policy.
- **Point-stealing restored** (`STEAL_FLAT=15`, `STEAL_PCT=0.30`) — only on
  player-owned snake bites (board terrain still doesn't steal).
- **Bankruptcy-immunity cooldown** (`BANKRUPT_IMMUNITY_TURNS=6`) — after a
  bankruptcy, further losses clamp the wallet to 0 instead of resetting again
  for 6 of the player's own turns. Anti steal/bomb death-loop.
- **Frozen self-play shape guard** — training only adds the existing PPO model
  as a frozen opponent if its obs/action shape matches; legacy 14-dim/4-action
  snapshots are skipped with a warning instead of crashing `predict()`.
- **Forced `device="cpu"`** — CPU is faster than the low-utilization GPU path
  for this small MLP (SB3 even warns about it). The "PPO on GPU" warning that
  still prints is spurious; the next line confirms `Using cpu device`.
- **Server PPO try/except per turn** — if `ppo_decision` raises (shape
  mismatch while a retrain is mid-write), the Hard AI falls back to Expectimax
  for the turn so the game doesn't crash.
- **Two-stage training recipe** documented in `knowledge/training.md`:
  - Stage 1: `train_ppo(3000000, opponent_pool=True)` (~80 min CPU, frozen
    self-play auto-skipped if shape mismatches).
  - **Back up before polishing:** `copy ai\ppo_model.zip ai\ppo_model_backup.zip`.
  - Stage 2: `train_ppo(2000000, opponent_pool=True)` (~55 min CPU, true
    continuation — stage-1 weights are loaded AND stage-1 PPO joins the
    frozen opponent pool; final file ≈ 5M total steps).
- **`train_ppo` continuation fix** — original `train_ppo` always built a
  fresh PPO and overwrote the model on save, so a "stage 2" run was really
  2M-from-scratch (and silently regressed 64%→52% WR on the first attempt).
  Now `train_ppo` loads the existing model if its shape matches
  (`PPO.load(..., env=vec_env)`) and calls `learn(reset_num_timesteps=False)`
  so the step counter and weights accumulate. Falls back to fresh PPO with a
  logged reason if the file is missing/incompatible.
- **Eval (200 games each, stochastic):** WR vs Easy **64%** (4.18 snakes/game,
  3.98 combos/game), WR vs Strong **62%** (4.12 snakes/game, 4.00 combos/game).
  PPO self-bankrupt 0.30-0.45/game (immunity working).

## Earlier (gameplay feel + web polish)
- **Snakes are exact-head only** — you only slide if you land exactly on a
  snake head (no "near miss" bites).
- **Mixed difficulty:** pick how many AIs are Hard (rest Easy) — menu has a
  "Hard AIs" count; CLI `--hard-ais N`.
- **Ladders** capped at 5–20 climb and spread out (no clustered/overlapping ladders).
- **Web UI:** title page → config → loading screen → board with **per-step token
  animation**; New game fully resets; refresh resumes; server is threaded (fixes
  the browser hang).
- **Engine-driven web** — the Flask server runs the Python engine as the single
  source of truth; `web/app.js` is a thin client (canvas render + animations
  only, no JS rules).

## Earlier additions (web UI, multiplayer, opponent-pool training)
- **Web UI** is now Flask-served on `http://localhost:5000` (`python main.py
  --web`), engine-driven — the Python engine in `game/` is the single source
  of truth; `web/app.js` is a thin client (canvas render + animations only).
  Pygame still works.
- **2-4 players, any mix of humans and AI**, shuffled turn order. Setup is
  asked on launch (or `--players/--humans/--hard-ais/--difficulty`).
- **Opponent-pool training** (`train_ppo(opponent_pool=True)`): each episode
  draws an opponent from {Easy, Strong heuristic, frozen self-PPO}. The frozen
  self-PPO is added only if its shape matches the current env (the cunning
  rebuild's guard).
- `--mode hvai/aivai` flags are gone — use `--players/--humans/--hard-ais`.

## TL;DR
- The **economy is now the core of the game.** Points are scarce; you spend them
  on snakes to sabotage opponents. Just rolling and ignoring the shop will lose.
- **Two real difficulties:** Easy (a weak, beatable bot) and Hard (the cunning
  22-dim / 5-action PPO agent — stochastic, sabotage-heavy, combo-spamming;
  WR vs Easy ~64%, vs Strong heuristic ~62%).
- A **trained PPO model is included** (`ai/ppo_model.zip` — cunning rebuild) —
  Hard mode works out of the box, no training step required.
- **Point-stealing on player-snake bites is back.** Bankruptcy gets a 6-turn
  immunity cooldown so death-loops can't happen.

## What to do after pulling
```bash
git checkout <this-branch>
python -m venv venv && venv\Scripts\activate     # if you don't have one yet
pip install -r requirements.txt
python main.py --web                             # play in the browser
# or: python main.py --players 2 --humans 1 --difficulty hard
```
You do **not** need to retrain — the model is committed. (Optional: `python
main.py --train` to regenerate it.)

## Current rules (after both passes)
- **Exact roll to win:** overshooting tile 100 = invalid move, you stay put.
- **Snakes:**
  - **Exact-head only** — you slide only when you land exactly on a head.
    Landing below or jumping clean over is safe.
  - A bite from a **player-owned snake** slides you back AND **steals points**
    (`STEAL_FLAT=15 + 30%` of your remaining points) to the snake's owner.
    Board terrain bites don't steal.
  - **Your own snakes don't bite you** (owner immunity).
  - Player snakes are **single-use** (consumed when they fire — then re-place).
  - You can't build a **wall** of adjacent snake heads.
- **Bankruptcy:** points below zero → back to tile 0, wallet zeroed, and you
  get a **6-turn immunity cooldown** during which further losses clamp the
  wallet to 0 instead of resetting again. Anti death-loop.
- **Economy:** tile income is low; snake pricing is sub-linear (save up for a
  big one). **Bombs scale with depth and can bankrupt you.**

## Code changes (where to look)
- `game/engine.py` — exact-head snakes, owner immunity, **point theft restored**
  (`_steal_points`, `STEAL_FLAT=15`, `STEAL_PCT=0.30`), scaled bombs, snake
  pricing, anti-wall rule, **`bankrupt_immune` decrement** in `do_turn`.
- `game/board.py` — scarce tile income.
- `game/models.py` — **`BANKRUPT_IMMUNITY_TURNS=6`**, `Player.bankrupt_immune`,
  `deduct_points` clamps to 0 during immunity.
- `ai/expectimax.py` — Easy bot (weakened) + catch-optimal placement strategies,
  **`propose_combo` (tail-on-bomb placement)**, `strong_decision` heuristic.
- `ai/ppo_agent.py` — **22-dim `encode_state`**, **`N_ACTIONS=5`** (combo
  action), cunning `_compute_reward` (heavy opp-setback, combo bonus,
  anti-hoard), `ent_coef=0.03`, `device="cpu"`, **stochastic** `ppo_decision`,
  frozen self-play shape guard in `build_training_env`.
- `server.py` — PPO inference wrapped in try/except per turn (graceful
  Expectimax fallback on shape mismatch / inference error).
- `main.py` — UTF-8 console output (fixes a Windows crash with the emoji logs).
- `knowledge/` — full design notes and the decision log behind all of this.

## Known notes
- Hard mode is intentionally tough (that's the point). Easy is the gentle one.
- If `ai/ppo_model.zip` ever fails to load, Hard mode auto-falls back to the
  Expectimax bot so the game still runs.
