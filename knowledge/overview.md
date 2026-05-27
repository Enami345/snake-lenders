# Snakes & Lenders — Project Overview

## What It Is

Strategic twist on Snakes & Ladders. Players earn **scarce** points and spend
them on **single-use sabotage snakes** that knock opponents back to the snake's
tail. Plays in the browser (web UI) with up to 4 players, any mix of humans/AI.

**Course:** Introduction to Artificial Intelligence — PUP, BSCS 3-4, Group 10
**Members:** Cabral · Caparas · Exconde · Rivera (repo owner: Geuel John Rivera)
**Status:** Working. On branch `refactor/economy-ai-overhaul`.

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.10+ (tested on 3.11) |
| Easy AI | Expectimax (deliberately weak heuristic) |
| Hard AI | PPO via stable-baselines3 + Gymnasium |
| UI | **Web (stdlib http.server + HTML/CSS/JS)** — primary; Pygame renderer still present |
| Deps | pygame, stable-baselines3, gymnasium, numpy (`requirements.txt`) |

Web UI uses **no extra dependency** (Python stdlib only).

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
├── ui/
│   ├── web/             # PRIMARY UI: server.py + index.html + style.css + app.js
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

python main.py --web        # Web UI → http://localhost:8000  (recommended)
python main.py --console    # terminal
python main.py              # legacy Pygame UI

# setup via flags (skip prompts):
python main.py --web        # browser handles the setup screen
python main.py --players 4 --humans 1 --difficulty hard      # pygame/console
python main.py --train --steps 300000                        # (re)train PPO
python main.py --phase 1                                     # board test
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
  - A bite just **slides you to the tail** — no point-stealing (removed for
    fairness; was causing bankruptcy death-spirals).
- **Economy:** scarce tile income (~4-14/turn); snake cost sub-linear
  (`2 × purchase_count × length^0.9`, min 12).
- **Bombs** scale with board depth and can **bankrupt** you → reset to tile 0
  (rare now, ~1 per 5 games for a passive player).
- Max 3 active player snakes; head tiles 20-90; no occupied/ladder tiles; no
  chaining; no wall (run of adjacent heads ≤ 2).

---

## AI

- **Easy = Expectimax**, deliberately weak (late, hesitant, hoards, cheap traps
  only). Beatable baseline.
- **Hard = PPO**, 4-action strategy space (roll / cheap trap / save-for-big /
  win-denial lurk). Beats human players in practice.
- **You can mix difficulties** — choose how many AIs are Hard (rest Easy).
- ⚠️ Honest note: under the current **exact-head** rule, snakes fire ~1/6, so
  the snake economy is weak and the shipped PPO (trained on an older
  stronger-snake rule) is **~level with Easy in bot-vs-bot sims** (it doesn't
  dominate). It still plays a sensible game and beats humans. See
  [training.md](training.md). Falls back to Expectimax if the model can't load.

See [architecture.md](architecture.md) for the code map, [status.md](status.md)
for current state, [training.md](training.md) for AI training + measured strength,
and [revisions.md](revisions.md) for the full change log.
