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

- **Web UI** (`ui/web/`, stdlib) — setup screen + board + snake shop + log;
  resumes on refresh. Primary UI. (`python main.py --web`)
- Console + legacy Pygame UI still present.
- **Multiplayer** 2-4 players, 0-N humans, AI fills the rest, shuffled turns.
- **Easy AI** = weak Expectimax baseline. **Hard AI** = PPO, beats Easy ~94%.
- Load-bearing economy: scarce points, single-use strike-range sabotage snakes
  with point theft, depth-scaled bombs, bankruptcy → tile 0.
- Exact-roll-to-win, anti-wall placement, owner immunity, catch-optimal AI.
- 14 passing unit tests (`tests/test_core.py`).
- PPO model + backup committed; auto-fallback if main is mid-write/corrupt.

---

## Verified

- All P0 + P1 done; P2 done bar 2 intentional skips (manuscript=PDF, full PPO
  per-tile decoupling).
- Sims: Hard vs Easy ~94%; strategy vs roll-only ~57%; AI-vs-AI ~50%.
- Tests pass; all modules import; web API smoke-tested (new/turn/state).

---

## Known Notes / Caveats

1. **Web UI not yet tested in a real browser session** (multi-human shop flow) —
   logic + API smoke-tested only. Needs a live pass.
2. **4×Hard FFA drags** (300+ turns) — intentional nightmare, left as-is.
3. **Hard while training** — if a training run is overwriting `ppo_model.zip`,
   loads fall back to `ppo_model_backup.zip` (stable). Refresh the backup after
   training: `copy ai\ppo_model.zip ai\ppo_model_backup.zip`.
4. **Manuscript (docs PDF) diverged** from code — see `manuscript.md`.
5. Bankruptcy resetting to tile 0 is intentional (brutal by design).

---

## Setup

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt      # pygame, stable-baselines3, gymnasium, numpy
```
Python 3.10+ (uses `tuple[bool,str]` / `dict | None` hints).
