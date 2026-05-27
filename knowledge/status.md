# Snakes & Lenders — Current Status & Notes

## Status: working, on branch `refactor/economy-ai-overhaul`

Big overhaul done (economy, AI, multiplayer, web UI). 2 branch commits ahead of
`main`:
```
e325280  feat: local multiplayer (2-4 players) + P1/P2 fixes, logging, tests
4372636  refactor: make the economy load-bearing and the AI difficulties real
```
Uncommitted (this session): entry-ladder fix, gprint logging, UI shop cap, dead
code, tests, **web UI (`ui/web/`)**, model backup-fallback, log relabel,
refresh-resume. (Pending a commit.)

---

## What's Included / Working

- **Web UI** (`ui/web/`, stdlib, threaded) — title page → config → loading
  overlay → board with **per-step token animation**, shop + log; New game resets,
  refresh resumes. Primary UI. (`python main.py --web`)
- Console + legacy Pygame UI still present.
- **Multiplayer** 2-4 players, 0-N humans, **mixed Easy/Hard** AIs (pick how many
  Hard), shuffled turns.
- **Easy AI** = weak Expectimax baseline. **Hard AI** = PPO (4-action).
- Economy: scarce points, **single-use exact-head sabotage snakes (no steal)**,
  depth-scaled bombs, bankruptcy → tile 0 (rare now). Ladders capped 5–20 + spread.
- Exact-roll-to-win, anti-wall placement, owner immunity.
- 14 passing unit tests (`tests/test_core.py`).
- PPO model + backup committed; auto-fallback if main is mid-write/corrupt.

---

## Verified

- All P0 + P1 done; P2 done bar 2 intentional skips (manuscript=PDF, full PPO
  per-tile decoupling).
- Tests pass; all modules import; web API smoke-tested (new/turn/state/quit).
- Game is playable + fun: passive-human bankruptcies ~0.21/game under exact-head.

---

## Known Notes / Caveats

1. **AI difficulty under exact-head:** snakes fire ~1/6, so the snake economy is
   weak and the shipped PPO (trained on the old strike-range rule) is ~level with
   Easy in **bot-vs-bot** sims (~42%). It still beats human players. Retraining
   on exact-head is optional (caps ~55%, dice ceiling). See `training.md`.
2. **Web UI not yet tested in a real browser session** (multi-human shop flow) —
   logic + API smoke-tested only. Needs a live pass.
3. **4×Hard FFA drags** (long games) — intentional nightmare, left as-is.
4. **Hard while training** — loads fall back to `ppo_model_backup.zip` if the main
   file is mid-write. Refresh backup after training:
   `copy ai\ppo_model.zip ai\ppo_model_backup.zip`.
5. **Manuscript (docs PDF) diverged** from code — see `manuscript.md`.
6. Bankruptcy resetting to tile 0 is intentional (now rare).

---

## Setup

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt      # pygame, stable-baselines3, gymnasium, numpy
```
Python 3.10+ (uses `tuple[bool,str]` / `dict | None` hints).
