# Snakes & Lenders — Project Overview

## What It Is

Strategic twist on Snakes & Ladders. Players earn points from tiles and **spend them to place snakes** that send opponents backward. Built with Python + Pygame.

**Course:** Introduction to Artificial Intelligence  
**Group:** 10 — BSCS 3-4, PUP  
**Members:** Cabral · Caparas · Exconde · Rivera (Geuel John Rivera = repo owner)  
**Status:** COMPLETE (4 commits, all phases done)

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.10+ |
| UI | Pygame |
| Easy AI | Expectimax (custom) |
| Hard AI | PPO via stable-baselines3 + Gymnasium |
| Data | dataclasses (no DB) |

**Dependencies** (`requirements.txt`):
```
pygame>=2.5.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
```

---

## Project Structure

```
snake-lenders/
├── main.py                  # Entry point, argparse CLI
├── requirements.txt
├── game/
│   ├── models.py            # Snake, Ladder, Player, BoardState dataclasses
│   ├── board.py             # Board generator (BFS validation)
│   ├── engine.py            # Turn loop, movement, shop, win check
│   └── console_game.py      # Terminal UI game loop
├── ai/
│   ├── expectimax.py        # Easy AI — Expectimax algorithm
│   ├── ppo_agent.py         # Hard AI — PPO training + inference
│   ├── ppo_model.zip        # Pre-trained PPO model (included in repo)
│   └── ppo_model_100k.zip   # Alternate pre-trained model
├── ui/
│   └── renderer.py          # Pygame renderer (board, panel, shop UI)
├── docs/
│   └── G10_SnakesLenders.pdf
└── knowledge/               # This folder — Claude session memory
    ├── overview.md          # This file
    ├── architecture.md      # Full code map
    ├── status.md            # Completion status & known issues
    └── manuscript.md        # Case study doc summary (from docs/G10_SnakesLenders.pdf)
```

---

## How to Run

```bash
python main.py                          # Human vs Human (Pygame UI)
python main.py --mode hvai              # Human vs Easy AI
python main.py --mode hvai --hard       # Human vs Hard AI
python main.py --mode aivai             # Easy AI vs Hard AI
python main.py --console                # Terminal mode
python main.py --phase 1                # Board generation test
python main.py --train                  # Train PPO (20-40 min)
python main.py --train --steps 500000   # Train more (better quality)
```

---

## Game Rules Summary

- 10×10 board, 100 tiles, **randomized every game**
- 7 ladders, 4 initial snakes, 5 bomb tiles (-30 pts)
- Players start at tile 0; first to reach tile 100 wins
- **Exact roll to win**: overshooting tile 100 is an invalid move — player stays put (e.g. 97+5 → stays at 97)
- **Bankruptcy**: going below 0 points → reset to tile 0
- **Snakes**:
  - **Player-placed snakes** = active single-use traps with a **strike range**: bite if you land on the head OR the 2 tiles just below it (~50% catch); consumed after firing (frees a slot to re-place). Jumping clean over is safe. A bite also **robs points** from the victim to the snake's owner (`STEAL_FLAT + 30%`), which can bankrupt and self-funds more traps.
  - **Board snakes** = fixed terrain, exact-head only.
- **Scarce economy** (this is the strategic core — using it well beats just rolling):
  - Low tile income (~4-14/turn) — points must be managed, not hoarded blindly.
  - Snake cost (sub-linear) = `2 × purchase_count × length^0.9`, min 12 — save up for a long, devastating snake.
  - Bombs scale with depth (`18 + tile//6`) and **can bankrupt** you (→ reset to tile 0) when low on points.
  - Max 3 active snakes per player; head tile 20–90; can't build a wall (`MAX_HEAD_RUN`).
  - Tile 100 = +40 finish bonus.

---

## Pygame Controls

| Key | Action |
|-----|--------|
| SPACE | Roll dice / continue |
| B | Open snake shop |
| ENTER | Confirm number input |
| ESC | Cancel |
| Q | Quit |
