# Snakes & Lenders — Project Overview

## What It Is

Strategic twist on Snakes & Ladders. Players earn **scarce** points and spend
them on **single-use sabotage snakes** that knock opponents back to the snake's
tail. Plays in the browser (web UI) with up to 4 players, any mix of humans/AI.

**Course:** Introduction to Artificial Intelligence — PUP, BSCS 3-4, Group 10
**Members:** Cabral · Caparas · Exconde · Rivera (repo owner: Geuel John Rivera)
**Status:** Working. On branch `web-app`.

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.10+ (tested on 3.11) |
| Easy AI | Expectimax (deliberately weak heuristic) |
| Hard AI | PPO via stable-baselines3 + Gymnasium |
| UI | **Web (Flask + canvas HTML/CSS/JS)** — primary, engine-driven; Pygame renderer legacy |
| Deps | pygame, stable-baselines3, gymnasium, numpy, **flask, flask-cors** (`requirements.txt`) |

Web UI is a **thin client** over the Python engine (single source of truth) — it
renders state and sends actions; all rules run server-side in `game/`.

---

## Project Structure

```
snake-lenders/
├── main.py              # CLI + setup flow (players/humans/difficulty) + dispatch
├── requirements.txt
├── CHANGELOG.md         # teammate-facing change brief
├── game/
│   ├── models.py        # Snake, Ladder, Player, BoardState
│   ├── board.py         # randomized board (BFS-validated) + scarce tile economy
│   ├── engine.py        # turn loop, movement, snake/bomb/economy rules
│   ├── console_game.py  # terminal loop (play_game(board, ppo_model))
│   └── log.py           # gated gprint() — silenced during training
├── ai/
│   ├── expectimax.py    # Easy bot (weak) + catch-optimal placement helpers
│   ├── ppo_agent.py     # Hard bot: 4-action PPO env, training, inference
│   ├── ppo_model.zip    # trained model (included)
│   └── ppo_model_backup.zip  # stable fallback (used if main is mid-write)
├── server.py            # PRIMARY UI backend: Flask, engine-driven (--web)
├── web/                 # PRIMARY UI frontend: index.html + app.js + style.css (thin client)
├── ui/
│   ├── web/             # legacy stdlib web UI (unused)
│   └── renderer.py      # legacy Pygame UI (kept)
├── tests/test_core.py   # 14 unittest cases
├── docs/                # case study manuscript (PDF)
└── knowledge/           # this folder — design notes + change log
```

---

## How to Run

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

python main.py --web        # Web UI → http://localhost:5000  (recommended)
python main.py --console    # terminal
python main.py              # legacy Pygame UI

# setup via flags (skip prompts):
python main.py --web        # browser handles the setup screen
python main.py --players 4 --humans 1 --difficulty hard      # pygame/console
python main.py --train --steps 3000000                       # (re)train PPO base
python main.py --phase 1                                     # board test

# Two-stage cunning training (see knowledge/training.md):
python -c "from ai.ppo_agent import train_ppo; train_ppo(3000000, opponent_pool=True)"  # stage 1
python -c "from ai.ppo_agent import train_ppo; train_ppo(2000000, opponent_pool=True)"  # stage 2
```

---

## Player Setup (any mix of human + AI)

Web flow: **title page → config → loading screen → board.** Config (or flags
`--players/--humans/--hard-ais`, `--difficulty` as all-easy/all-hard shortcut):
1. Players total **2-4**
2. Human players **0-N** (rest are AI)
3. **How many Hard AIs** (0..AI count) — the rest are Easy. Mixed allowed.

- Human(s) first, then AI; **turn order is shuffled** (no first-mover bias).
- AI is **free-for-all**: each bot targets the current leader (human or AI).
- 0 humans = AI watch mode. 4×Hard = intentional nightmare (long games).

---

## Game Rules (current)

- 10×10 board, randomized each game (7 ladders, 4 board snakes, 5 bombs).
- **Exact roll to win:** overshooting 100 = invalid, stay put.
- **Ladders:** climb on exact landing (also on board entry). Jump range capped
  at 5–20 tiles (no absurd 42→90 leaps).
- **Snakes — exact-head only** (`STRIKE_ZONE=0`): you slide ONLY when you land
  exactly on a snake head. Landing below it or jumping clean over = safe.
  - *Player snakes* = single-use traps; consumed on fire (re-placeable).
    **Owner immune to own snakes.**
  - **Point-stealing on bite** (player-owned snakes): victim loses
    `STEAL_FLAT=15 + 30%` of remaining points → to the snake's owner.
    Restored to make the game stressful and self-fund the saboteur.
- **Bankruptcy immunity:** after a bankruptcy, the player gets
  `BANKRUPT_IMMUNITY_TURNS=6` turns where further losses **clamp the wallet to
  0** instead of resetting again. Prevents steal/bomb death-loops.
- **Economy:** scarce tile income (~4-14/turn); snake cost sub-linear
  (`2 × purchase_count × length^0.9`, min 12).
- **Bombs** scale with board depth and can **bankrupt** you → reset to tile 0.
- Max 3 active player snakes; head tiles 20-90; no occupied/ladder tiles; no
  chaining; no wall (run of adjacent heads ≤ 2).

---

## AI

- **Easy = Expectimax**, deliberately weak (late, hesitant, hoards, cheap traps
  only). Beatable baseline.
- **Hard = PPO**, **5-action** strategy space (roll / cheap trap / save-for-big /
  win-denial lurk / **combo = tail-on-bomb**). **22-dim** observation including
  bomb/ladder lookahead, leader's distance to goal, combo availability, and the
  agent's bankruptcy-immunity flag. **Stochastic inference** for unpredictable,
  cunning play.
- **You can mix difficulties** — choose how many AIs are Hard (rest Easy).
- Shipped Hard model = **cunning rebuild, 22-dim / 5-action, ~3M base + 2M
  self-play stage-2**. Trained against an opponent pool {Easy, Strong heuristic,
  frozen self-PPO snapshot} with sabotage-heavy reward shaping (heavy opponent-
  setback reward, large combo bonus, anti-hoard penalty). Measured WR vs Easy
  ~64%, vs Strong ~62%, with ~4 snakes/game and ~4 combos/game vs the prior
  reserved ~1-2 snakes / 0 combos. See [training.md](training.md). Server falls
  back to Expectimax per turn if PPO inference fails (e.g. shape mismatch).

See [architecture.md](architecture.md) for the code map, [status.md](status.md)
for current state, [training.md](training.md) for AI training + measured strength,
and [revisions.md](revisions.md) for the full change log.
