# Snakes & Lenders — Current Status & Notes

## Status: working, on branch `web-app`

Big AI overhaul shipped: PPO rebuilt as a **22-dim / 5-action cunning agent**
with restored point-stealing, a bankruptcy-immunity cooldown, and a stochastic
inference policy. Web UI is engine-driven (Flask thin client). Two-stage
training recipe documented.

---

## What's Included / Working

- **Web UI** (`server.py` Flask + `web/`, **engine-driven thin client**) —
  premium canvas board, audio, **chess-style snake placement** (glow heads/tails
  + confirm dialog + grow animation), per-step token animation, bomb/bankruptcy
  animation, tile point values. All rules run server-side via `game/`.
  (`python main.py --web` → http://localhost:5000.)
  PPO inference is wrapped in try/except per turn → Hard AI falls back to
  Expectimax for the turn if PPO can't decide.
- Console + legacy Pygame UI still present.
- **Multiplayer** 2-4 players, 0-N humans, **mixed Easy/Hard** AIs (pick how
  many Hard), shuffled turns.
- **Easy AI** = weak Expectimax baseline.
  **Hard AI** = PPO **22-dim obs / 5-action** cunning agent — actions: roll /
  cheap trap / big snake / win-denial lurk / **combo (tail-on-bomb)**;
  **stochastic** inference for unpredictable play.
- Economy: scarce points, **single-use exact-head sabotage snakes**;
  **point-stealing restored** (`STEAL_FLAT=15`, `STEAL_PCT=0.30`) so successful
  sabotage self-funds the saboteur and pressures the victim;
  depth-scaled bombs; bankruptcy → tile 0 with a **6-turn immunity cooldown**
  that clamps further losses to 0 (anti death-loop). Ladders capped 5-20 +
  spread.
- Exact-roll-to-win, anti-wall placement, owner immunity.
- 14 passing unit tests (`tests/test_core.py`).
- PPO model + legacy backup committed; auto-fallback if main is mid-write or
  raises.

---

## Verified

- All P0 + P1 done; P2 done bar 2 intentional skips (manuscript=PDF, full PPO
  per-tile decoupling).
- Tests pass (14/14); all modules import; web API smoke-tested.
- Cunning PPO eval: WR 64% vs Easy, 62% vs Strong; ~4 snakes/game; ~4
  combos/game; PPO self-bankrupt 0.30-0.45/game (immunity working).
- CSS root bug fixed (unclosed `@media` trapped all rules after line 1446).
- Confirm-snake modal, replay bar UI fixed; victory stats now tracked from
  engine logs; in-game rules text updated (steal, immunity, Hard AI desc).
- No-cache Flask headers so CSS/JS edits show on normal reload.
- Graphify skill installed (`graphifyy`); set `ANTHROPIC_API_KEY` then run
  `graphify . --backend claude` for interactive codebase graph.

---

## Known Notes / Caveats

1. **Shipped Hard AI:** 22-dim / 5-action cunning rebuild (~3M base + 2M
   self-play stage-2). Stochastic, sabotage-heavy, combo-spamming. Beats Easy
   ~64% and Strong ~62% (200 games each). Behavior is dramatically more
   aggressive than the reserved 14-dim build (~4 snakes/game vs ~1-2).
   See `training.md`.
2. **`ai/ppo_model_backup.zip` is the last known-good cunning model** (22-dim
   / 5-action, 3M-step stage-1). It's the manual restore point before risky
   polish runs and is also served by `load_ppo_model` as a last-resort
   inference path. If a future polish run regresses, restore with
   `copy ai\ppo_model_backup.zip ai\ppo_model.zip`. (If you ever see
   "frozen model incompatible (obs=…)" during training, that's the shape
   guard skipping an incompatible snapshot — train a current-shape model
   and the next stage will pick it up.)
3. **Spurious GPU warning** during training ("PPO on the GPU…") prints before
   `device="cpu"` is applied. The next line shows `Using cpu device`. Training
   is on CPU as intended.
4. **Web UI not yet tested in a real browser session** (multi-human shop flow)
   for the cunning build — logic + API smoke-tested only.
5. **4×Hard FFA drags** (long games) — intentional nightmare, left as-is.
6. **Hard while training** — `load_ppo_model` tries main → backup; the server
   then wraps inference in try/except so a shape-mismatched in-progress model
   doesn't crash the running game (it falls back to Expectimax that turn).
   Refresh backup after a successful train + play-test:
   `copy ai\ppo_model.zip ai\ppo_model_backup.zip`.
7. **Manuscript (docs PDF) diverged** from code — see `manuscript.md`.
8. Bankruptcy resetting to tile 0 is intentional; the 6-turn immunity prevents
   chains.

---

## Setup

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt      # pygame, stable-baselines3, gymnasium, numpy, flask, flask-cors
```
Python 3.10+ (uses `tuple[bool,str]` / `dict | None` hints).

### Retraining the cunning PPO

`train_ppo` now **continues from `ai/ppo_model.zip` if the shape matches**, so
runs accumulate instead of overwriting. Two-stage recipe (full rationale in
`training.md`):

```bash
venv\Scripts\activate

# Stage 1 — base (3M).  Either remove ai/ppo_model.zip first OR rely on the
# shape-mismatch fallback (e.g. against a legacy 14-dim file) so this run
# isn't an accidental continuation of an unrelated checkpoint.
python -c "from ai.ppo_agent import train_ppo; train_ppo(3000000, opponent_pool=True)"

# **Always back up before polishing** so a regression is recoverable.
copy ai\ppo_model.zip ai\ppo_model_backup.zip

# Stage 2 — self-play polish (2M true continuation; stage-1 PPO also joins
# the frozen opponent pool → final file ≈ 5M total steps).
python -c "from ai.ppo_agent import train_ppo; train_ppo(2000000, opponent_pool=True)"
```

If stage-2 eval regresses, restore:
```
copy ai\ppo_model_backup.zip ai\ppo_model.zip
```
After a successful play-test, refresh the backup again.
