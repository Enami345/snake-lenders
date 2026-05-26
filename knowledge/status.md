# Snakes & Lenders — Current Status & Known Notes

## Completion Status: DONE

All phases complete. Game is fully playable in both Pygame UI and console mode.

**Git history:**
```
6946a39  Include trained PPO model so groupmates skip training
f0dfb67  Add requirements.txt and readme
d91f65b  Move README and gitignore to root folder
27afcea  Initial commit — Snakes & Lenders complete
```

The PPO model is intentionally committed (`ai/ppo_model.zip` and `ai/ppo_model_100k.zip`) so teammates don't need to re-train. README now outdated on this point (still says model not included).

---

## What's Included

- Full Pygame UI with snake shop interaction
- Console/terminal mode (same gameplay)
- Expectimax AI (Easy) — no setup needed
- PPO AI (Hard) — pre-trained model already in repo
- Randomized board with BFS validation (ensures avg ≥ 10 turns)
- Snake shop with ROI-based AI decision making
- Bankruptcy mechanic
- Exact-roll-to-win: overshoot tile 100 = invalid, stay put

---

## Untracked Folder

- `docs/` is listed as untracked in git status — contains `G10_SnakesLenders.pdf` (project report)

---

## Potential Issues / Notes

1. **README says PPO model not included** — outdated. Model IS included now.
2. **Console shop** asks `press Enter to roll` before shop — AI prompt says same thing for both human and AI turns (cosmetic)
3. **PPO fallback** — Hard AI silently falls back to Expectimax if model missing; this is intentional
4. **3-4 player support** exists in argparse (`--players`) but `hvh` and `hvai` modes hardcode 2 players; only `aivai` and `--phase 1` use `--players`
5. **Board regeneration** — generates new random board each run; no way to replay exact same board without `--seed`
6. **`__pycache__`** folders present in repo (not gitignored per module, just root `.gitignore`)

---

## Dependencies — Install Command

```bash
pip install pygame stable-baselines3 gymnasium numpy
```

Or via requirements.txt:
```bash
pip install -r requirements.txt
```

---

## Python Version

Requires Python 3.10+ (uses `match`-style type hints like `tuple[bool, str]` and `dict | None`).
