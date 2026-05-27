# Snakes & Lenders — AI Training & Quality

How the Hard (PPO) AI is trained, how strong it is, and what we measured.

> ⚠️ **IMPORTANT — rule change after these numbers were measured.** The win-rate
> battery below was measured under the **strike-range** snake rule (snakes bit
> within a few tiles below the head, ~50% catch). The game has since switched to
> **exact-head only** (`STRIKE_ZONE=0`) and **point-stealing was removed**, for a
> fairer, less-bankruptcy-heavy game. Under exact-head, snakes fire ~1/6, the
> snake economy is weak, and the **shipped PPO (trained on strike-range) is
> rule-mismatched → ~42% vs Easy in bot-vs-bot sims** (it still beats human
> players). Retraining on exact-head is optional and caps around ~55% (dice
> ceiling). The numbers below document what the strike-range training achieved.

## Timesteps vs games
- PPO trains in **environment steps**; 1 step = one full round (agent turn +
  opponents). ~**30-50 steps per game**.
- So ~40 steps/game → 200k steps ≈ 5,000 games; 2.5M steps ≈ ~62,000 games;
  100k games ≈ ~4M steps.

## Convergence / diminishing returns
- Small env (14-dim obs, Discrete(4) actions) → returns **plateau ~1-3M steps**.
- Past that: near-flat or mild overfit; `clip_fraction → 0`, `approx_kl → ~5e-5`
  signal the policy has essentially stopped changing.
- **More games = better, but with diminishing returns, then a wall.** Not linear,
  not unbounded. ~2.5M is the practical sweet spot here.

## Training setup
- Reward: ±100 win/loss (dominant) + bounded progress-delta + opponent-setback +
  bankruptcy penalty. 4 parallel envs, MlpPolicy, lr 3e-4.
- **Opponent pool / self-play** (`train_ppo(opponent_pool=True)`): each episode the
  opponent is drawn from {Easy bot, Strong heuristic, frozen current-best PPO}.
  Improves robustness vs the old "train only vs weak Easy" setup.
- Commands:
  ```
  python main.py --train --steps 2500000            # vs weak Easy
  python -c "from ai.ppo_agent import train_ppo; train_ppo(2500000, opponent_pool=True)"
  ```

## Model lineage
| Model | Steps | Notes |
|-------|-------|-------|
| (old 100k) | 100k | removed (ai/ppo_model_100k.zip deleted) |
| 200k | 204,800 | first valid model on fixed env; ~89% vs Easy |
| 2.5M | 2,506,752 | vs weak Easy; ~91% vs Easy |
| **2.5M pool/self-play** | 2,506,752 | **shipped** (`ai/ppo_model.zip`); trained vs the pool |

## Measured performance (shipped model)
Sims of ~300 games each, seats shuffled:

| Matchup | Win rate |
|---------|----------|
| Hard vs Easy | ~89-91% |
| Hard vs **Strong heuristic** | **89%** (pool model; 2.5M-vs-Easy was 83%) |
| Hard vs old 200k (head-to-head) | 78% |
| 2.5M-pool vs 2.5M-Easy (head-to-head) | ~52% (tie within noise) |
| 4p — 1 Hard vs 3 Easy | ~86% (random = 25%) |
| 4p — 1 Hard vs 3 Strong | ~80% (random = 25%) |
| Mirror (vs identical copy) | ~50% |

Key results:
- Hard **generalizes** — beats the Strong hand-tuned bot ~86-89% despite training
  against weaker opponents.
- The **self-play/pool** model matched the plain-2.5M head-to-head (~52%) and was
  **+6pp more robust vs Strong** (83%→89%) — small but real, so it was promoted.
- Dice variance caps everything below 100%; even strongest-vs-passive ≈ 57% in a
  straight 2-player race. The big multiplayer numbers come from sabotage compounding.

## Model files
- `ai/ppo_model.zip` — shipped model (loaded first).
- `ai/ppo_model_backup.zip` — stable fallback; `load_ppo_model` uses it if the main
  file is missing/mid-write/corrupt (e.g. during a training run).
- Refresh the backup after a new train: `copy ai\ppo_model.zip ai\ppo_model_backup.zip`.

## Possible further improvements (not done — diminishing returns)
- Richer observation (encode snake gauntlets, bombs ahead, opponents' economy).
- Finer action space (net picks placement tiles, not 4 canned strategies).
- Reward tuning / exploration (`ent_coef`), longer self-play league.
For a class project the shipped model is already strong (80-91% everywhere).
