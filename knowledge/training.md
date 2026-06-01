# Snakes & Lenders — AI Training & Quality

How the Hard (PPO) AI is trained, how strong it is, and what we measured.

> ⚠️ **CURRENT shipped model = "cunning rebuild" — 22-dim obs / 5 actions,
> ~3M base steps + 2M self-play stage-2.**
> Rules: **exact-head snakes** (`STRIKE_ZONE=0`), **point-stealing on bite
> restored** (`STEAL_FLAT=15`, `STEAL_PCT=0.30`), **bankruptcy-immunity
> cooldown** (`BANKRUPT_IMMUNITY_TURNS=6`) prevents steal/bomb death-loops.
> Measured vs Easy ~**64%**, vs Strong heuristic ~**62%** (200 games each,
> stochastic inference). Behavior: ~4 snakes/game, ~4 combos/game,
> action-mix spread across all 5 strategies (no single move dominates).

## What changed from the old build

| | Old (reserved) | New (cunning) |
|---|---|---|
| Observation | 14-dim | **22-dim** (+ bomb/ladder lookahead, leader's distance to goal, combo availability, bankruptcy-immunity flag) |
| Actions | Discrete(4): roll / cheap / big / lurk | **Discrete(5)** — adds **combo** (snake tail on bomb tile) |
| Inference | `deterministic=True` | **`deterministic=False`** — stochastic, unpredictable mix |
| Entropy | `ent_coef` default (0.0) | **`ent_coef=0.03`** — varied policy |
| Device | auto (GPU warning fires) | **`device="cpu"`** — faster for small MLP |
| Reward shaping | progress + bounded setback | **Heavy sabotage:** `+0.5 × opp_setback`, `+12.0` combo bonus, `+4.0 + 0.3 × placed_setback` per buy, **`-1.0` anti-hoard** if sitting >100pts without buying |
| Stealing | none | **`STEAL_FLAT=15 + 30%`** of victim points → owner |
| Death-loop | n/a (no steal) | **6-turn immunity** clamps wallet to 0 after a bankruptcy |
| Snakes/game (vs Easy) | ~1.23 | **~4.18** |
| Combos/game | 0 | **~4.0** |
| WR vs Easy | ~54-56% | **~64%** |
| WR vs Strong | ~89% (old strict shaping) → ~52% (reserved variant) | **~62%** |

The cunning rebuild trades a tiny bit of raw vs-Easy dominance for:
1. **Unpredictability** — no single action accounts for >35% of decisions.
2. **Aggression** — actually deploys snakes (~4/game), maxes combo annoyance.
3. **Self-funding** — point theft restored, so successful sabotage compounds.
4. **No death-loops** — immunity cooldown protects against repeat bankruptcies.

## Timesteps vs games
- PPO trains in **environment steps**; 1 step = one full round (agent turn +
  opponents). ~**30-50 steps per game**.
- So ~40 steps/game → 200k steps ≈ 5,000 games; 3M steps ≈ ~75,000 games.

## Convergence / diminishing returns
- Bigger obs (22 vs 14) + bigger action space (5 vs 4) is slightly more
  sample-hungry → returns plateau **~2-4M steps** for this MLP.
- Stage-2 self-play polish saturates fast (frozen opponent doesn't evolve);
  **1.5-2M is the safe ceiling** — past that risks overfitting that one
  snapshot's quirks instead of gaining general skill.
- Past ~5M total = waste or mild regression.

## Two-stage training (recommended)

`train_ppo` now **continues from `ai/ppo_model.zip` if the shape matches**
(otherwise it starts fresh and logs why). Step counts accumulate across runs
via `learn(reset_num_timesteps=False)`.

Stage 1 — base (~3M steps, ~80 min on CPU, 4 envs):
```
# Before stage 1, make sure ai/ppo_model.zip is either the legacy 14-dim file
# (will be detected as shape-mismatch and started fresh) OR has been removed,
# so this run isn't accidentally a continuation.
python -c "from ai.ppo_agent import train_ppo; train_ppo(3000000, opponent_pool=True)"
```
The frozen self-play opponent is **automatically skipped** if the existing
`ai/ppo_model.zip` shape doesn't match. Pool falls back to {Easy, Strong}.

**Save the stage-1 model before stage 2** (so a bad polish run doesn't
overwrite it):
```
copy ai\ppo_model.zip ai\ppo_model_backup.zip
```

Stage 2 — self-play polish (~1.5-2M steps, ~40-55 min on CPU):
```
python -c "from ai.ppo_agent import train_ppo; train_ppo(2000000, opponent_pool=True)"
```
This is now a **true continuation**: `train_ppo` loads the stage-1 weights,
keeps the step counter, and adds the stage-1 PPO as a frozen self-play
opponent (pool = {Easy, Strong, frozen stage-1 PPO}). Final file ≈ 5M total
steps.

If stage-2 eval regresses (the polish overfit the frozen snapshot's quirks),
restore stage 1:
```
copy ai\ppo_model_backup.zip ai\ppo_model.zip
```

Optional stage-3 — multi-snapshot pool: stash the stage-2 model under a
second name and modify `build_training_env` to load both as frozen opponents.

## Training setup
- Reward: `_compute_reward` in `ai/ppo_agent.py` (see Architecture for the
  shaping table).
- 4 parallel envs (`make_vec_env(n_envs=4)`).
- `MlpPolicy`, lr `3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`,
  `gamma=0.99`, `clip_range=0.2`, `ent_coef=0.03`, `device="cpu"`.
- Opponent pool decider is randomly drawn per `reset()` so each episode faces
  a different opponent.
- `game.log.VERBOSE` is flipped off during `model.learn` so only SB3's
  progress tables print.

## Model lineage
| Model | Steps | Obs / Actions | Notes |
|-------|-------|---------------|-------|
| (legacy 100k) | 100k | 14 / 4 | removed |
| 200k | 204,800 | 14 / 4 | first valid model |
| 2.5M | 2,506,752 | 14 / 4 | vs weak Easy |
| 2.5M pool/self-play | 2,506,752 | 14 / 4 | reserved play (~52% vs Strong); **`ai/ppo_model_backup.zip`** |
| **cunning rebuild — 3M base + 2M stage-2** | ~5M | **22 / 5** | **shipped — `ai/ppo_model.zip`** |

The legacy backup is kept for emergency fallback only; it has an incompatible
shape so it cannot be used as a frozen self-play opponent for the new env (the
shape guard skips it).

## Measured performance (shipped cunning model)

200 games each, seats alternated, **stochastic** inference:

| Matchup | WR | Snakes/game | Combos/game | Avg setback | Steals → PPO/game | PPO self-bankrupt/game |
|---|---|---|---|---|---|---|
| Hard vs Easy | **64%** (127-73) | 4.18 | 3.98 | 11.4 | 1.6 | 0.30 |
| Hard vs Strong | **62%** (125-75) | 4.12 | 4.00 | 10.9 | 1.6 | 0.45 |

Action mix (vs Strong, 200 games):
- combo: 1755 ← top pick
- lurk:  1598 ← second (patient ambush)
- roll:  1403
- cheap: 729
- big:   622

No single action dominates → stochastic policy is working. Combo fires roughly
every chance it can. Anti-hoard penalty is biting: agent spends instead of
sitting. Immunity cooldown is biting: self-bankruptcy stays under 0.5/game.

## Model files
- `ai/ppo_model.zip` — shipped cunning model (22 / 5).
- `ai/ppo_model_backup.zip` — the **last known-good** cunning model. Used as a
  manual restore point before risky polish runs (see two-stage section). Also
  served by `load_ppo_model` as a last-resort inference path if the main file
  is missing/mid-write.
- Refresh the backup **after a successful train and play-test**:
  `copy ai\ppo_model.zip ai\ppo_model_backup.zip`.
- If a polish run regresses, restore with the reverse copy
  (`copy ai\ppo_model_backup.zip ai\ppo_model.zip`).

## Notes
- The SB3 "PPO on GPU" warning that prints during training is **spurious** —
  the next line confirms `Using cpu device`. Our `device="cpu"` override does
  take effect; the warning fires in PPO's `__init__` before the override is
  applied.
- `train_ppo` continues from the existing model when the shape matches; a
  "Continuing from ai/ppo_model.zip (N prior steps)." line confirms it. If you
  see "starting fresh" instead, the file was missing/incompatible and the next
  run is from scratch — back up your good model first.
- "frozen model incompatible (obs=…, n_actions=…) — skipping self-play" is the
  shape guard working: a legacy snapshot can't safely be added to the opponent
  pool. Train a current-shape model and the next stage will pick it up.
- Server PPO inference is wrapped in try/except — if a shape-mismatched model
  gets loaded by mistake, the Hard AI falls back to Expectimax per turn rather
  than crashing the game.

## Stage-2 retrain verdict (2026-06-01)

Ran 2M true continuation (patch working; counter started at 3M → ended at 5M).
Eval showed **marginal / wash result**:

| | Stage-1 (3M backup) | Stage-2 (5M) |
|---|---|---|
| WR vs Easy | 54.5% | **58.0%** (+3.5pp) |
| WR vs Strong | 60.0% | **61.5%** (+1.5pp) |
| H2H 300g | **53.7%** | 46.3% (−7.4pp) |
| Avg setback | **11.0 tiles** | 8.7 tiles |
| Top actions | roll / cheap / big | **lurk / combo** (spammer shift) |

Stage-2 shifted toward lurk+combo spam, away from big snakes. Loses H2H to
stage-1 and places shorter snakes. "Spammer vs sniper" behavioral tradeoff.
Decision: **ship stage-1 (3M)** as primary — better H2H + longer setbacks.
Stage-2 vs-bot gain within noise (±3-4pp at 200g). Stage-2 stored as new
`ppo_model_backup.zip` for reference.

## Lessons learned (2026-05/06)

- **Always back up before a polish run.** The first stage-2 attempt was a
  2M-from-scratch run (when `train_ppo` didn't yet continue from the existing
  weights), which silently overwrote the 3M stage-1 model with a weaker 2M
  one (WR vs Easy 64% → 52%). Recovery worked because the 3M was saved as
  `ppo_model_backup.zip` first. The `train_ppo` patch + the backup discipline
  prevent this from happening again.
