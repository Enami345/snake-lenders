# 🐍 Snakes & Lenders

A strategic twist on the classic Snakes & Ladders board game.
Players earn points and **buy snakes** to sabotage opponents.
Built with Python and Pygame, featuring two AI difficulty levels.

**Group 10 — BSCS 3-4 | Introduction to Artificial Intelligence | PUP**
Cabral · Caparas · Exconde · Rivera

---

## What is Snakes & Lenders?

Unlike classic Snakes & Ladders, which is pure luck, Snakes & Lenders adds a
**resource-management layer**: you earn points from tiles and spend them to
place snakes that knock opponents backward (and rob their points). Points are
**scarce** and bombs can **bankrupt** you, so winning depends on smart timing —
when to save, when to strike, and where to trap an opponent — not just the dice.

> The core principle: **using the economy well is your leverage.** A player who
> just rolls and ignores the shop will lose to one who plays the economy.

---

## Requirements

### Python
- Python **3.10 or higher** — https://www.python.org/downloads/
- During install, check **"Add Python to PATH"**

### Setup (virtual environment recommended)
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

`requirements.txt` pulls in: `pygame`, `stable-baselines3`, `gymnasium`, `numpy`,
`flask`, `flask-cors` (the web UI runs on Flask).

---

## Project Structure

```
snake-lenders/
├── main.py              # Entry point / CLI (modes, training, UTF-8 console fix)
├── requirements.txt
├── game/
│   ├── models.py        # Snake, Ladder, Player, BoardState dataclasses
│   ├── board.py         # Randomized board generator (BFS-validated) + tile economy
│   ├── engine.py        # Turn loop, movement, snake/bomb/economy rules
│   └── console_game.py  # Terminal game loop
├── ai/
│   ├── expectimax.py    # Easy AI — cunning saboteur + placement strategies
│   ├── ppo_agent.py     # Hard AI — PPO env, training, inference
│   └── ppo_model.zip    # Trained PPO model (included)
├── server.py            # Web UI backend — Flask, engine-driven (python main.py --web)
├── web/                 # Web UI frontend — index.html + app.js + style.css (thin client)
├── ui/
│   └── renderer.py      # legacy Pygame board + side panel + snake shop
├── docs/                # Case study manuscript (PDF)
└── knowledge/           # Design notes / change log for the refactor
```

---

## How to Run

```bash
# from the project root, with the venv activated
python main.py --web        # Web UI → http://localhost:5000  (recommended)
python main.py --console    # play in the terminal
python main.py              # legacy Pygame UI

# Setup is asked interactively (players 2-4 / humans 0-N / how many Hard AIs),
# or skip the prompts with flags:
python main.py --players 4 --humans 1 --hard-ais 1     # you + 1 Hard + 2 Easy
python main.py --players 2 --humans 2                  # local 2-human
python main.py --players 4 --humans 0 --difficulty easy --web   # AI watch mode

python main.py --phase 1                # board-generation test
python main.py --train --steps 2500000  # (re)train the PPO Hard AI
```

A trained `ai/ppo_model.zip` is **included** (a 2.5M-step self-play model), so
Hard mode works out of the box. A stable `ai/ppo_model_backup.zip` is used if the
main file is missing/mid-write; if neither loads, Hard falls back to Expectimax.

---

## Controls (Pygame UI)

| Key | Action |
|-----|--------|
| **SPACE** | Roll dice / continue |
| **B** | Open snake shop (human players) |
| **ENTER** | Confirm number input in shop |
| **ESC** | Cancel shop input |
| **Q** | Quit |

---

## Game Rules

### Board
- 10×10 board, 100 tiles, **randomized every game** (BFS-validated as solvable)
- 7 ladders, 4 initial board snakes, 5 bomb tiles
- Players start off the board at tile 0

### Movement
- Roll a 6-sided die each turn
- Land on a ladder bottom → climb to the top. Ladder climbs are capped at 5–20
  tiles and spread out across the board (no clustered/overlapping ladders).
- **Exact roll to win:** overshooting tile 100 is an invalid move — you **stay
  put** (e.g. tile 97 + roll 5 → stays at 97). You must land on 100 exactly.

### Snakes — exact-head only
- You slide **only when you land exactly on a snake head.** Landing on a tile
  *below* the head, or jumping clean *over* it, is safe.
- A bite just **slides you down to the tail** (no point-stealing).
- **Owner immunity:** your own snakes never bite you.
- Player snakes are **single-use**: once a snake fires it's consumed, freeing a
  slot so you can place another. Board snakes are permanent terrain.

### Economy (the heart of the game)
- **Tile income is scarce** (~4–14 points/turn) — you must manage points, not
  hoard mindlessly.
- **Bombs scale with board depth** (deeper = nastier) and can push you below zero
  → **bankruptcy**: sent **back to tile 0** with zero points (rare now, ~1 per 5
  games for a passive player).
- Tile 100 gives a small finish bonus.

### Snake Shop
- **Cost (sub-linear):** `2 × purchase_count × length^0.9` (min 12). Short snakes
  are cheap; long, devastating snakes are affordable if you **save up** for them.
- Maximum **3 active snakes** per player (single-use; place more as they fire)
- Snake head must be between tiles **20–90**
- Cannot place on occupied tiles or ladder tiles
- Cannot chain snakes (head can't sit on another snake's tail)
- **Cannot build a wall:** a snake can't extend a run of adjacent heads past 2,
  so a region is always passable

### Winning
- First player to land **exactly** on tile 100 wins.

---

## AI Opponents

Pick how many AIs are **Hard** (rest are **Easy**) — you can mix them.

### Easy — Expectimax (deliberately weak)
A simple, beatable baseline: it reacts late, hesitates, hoards too many points,
and only ever places cheap short traps. Good for learning the game. No training.

### Hard — PPO (Proximal Policy Optimization)
A neural-network agent (`stable-baselines3`) that chooses a **strategy** each turn
from a 4-action space (roll / cheap trap / save-for-big / win-denial lurk),
trained by self-play against an opponent pool. A trained model ships with the repo.

> Note: the shipped model is **exact-head-trained (2.5M, self-play)** and is
> **proven stronger than Easy (~54-56%)** — it just doesn't dominate, because
> exact-head snakes fire only ~1/6 and the dice cap the gap. The skill difference
> shows in *behavior*: Hard saves up for ~50-tile knockbacks + finish-line traps,
> Easy only sprinkles short cheap snakes (avg length 53 vs 7.5). Full detail in
> `knowledge/training.md`.

---

## What changed in this refactor

This branch reworks the game (economy, AIs, a web UI, local multiplayer).

**Latest tuning (current rules)**
- Snakes are **exact-head only** and **don't steal points** — far fewer
  bankruptcies, fairer play. (An earlier strike-range + steal version made the AI
  strong but caused bankruptcy spirals.)
- Ladder climbs capped at 5–20 tiles and **spread out** (no clustered ladders).
- **Mixed difficulty:** choose how many AIs are Hard (rest Easy).
- **Web UI** with a title page → config → loading screen → animated board.

**Movement & win condition**
- Overshooting tile 100 means **stay put** (exact roll to win).
- Fixed a bug where entering the board could skip a ladder.

**Snakes**
- Single-use traps, owner-immune, exact-head trigger; board snakes are terrain.
- **Anti-wall placement rule** so heads can't form an impassable cluster.
  (Pass-over triggering was prototyped and reverted — long snakes became walls.)

**Economy & bankruptcy**
- **Slashed tile income** and switched snake pricing to **sub-linear** so points
  are scarce and you save up for big plays.
- **Depth-scaled bombs** that actually cause **bankruptcy** (reset to tile 0).

**AI**
- Rewrote the **Expectimax** AI into a cunning, aggressive, win-denying saboteur
  with proper points-based valuation (the old ROI math treated cost as ~free).
- Rebuilt the **PPO** Hard AI: fixed a broken training environment (turns didn't
  alternate; the opponent never played its own policy), fixed runaway reward
  shaping, expanded the action space to 4 strategies, and retrained.

**Infra**
- Forced **UTF-8 console output** so emoji/arrow game logs don't crash on Windows.
- Added design notes and a full change log under `knowledge/`.
