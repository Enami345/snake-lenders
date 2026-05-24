# 🐍 Snakes & Lenders

A strategic twist on the classic Snakes & Ladders board game.
Players earn points and **buy snakes** to hinder opponents.
Built with Python and Pygame, featuring two AI difficulty levels.

**Group 10 — BSCS 3-4 | Introduction to Artificial Intelligence | PUP**
Cabral · Caparas · Exconde · Rivera

---

## What is Snakes & Lenders?

Unlike classic Snakes & Ladders which relies purely on luck, Snakes & Lenders
adds a strategic layer where players earn points from tiles and spend them to
**place snakes** on the board to send opponents backward. Victory depends on
smart resource management, not just dice rolls.

---

## Requirements

Make sure you have the following installed before running the game.

### Python
- Python **3.10 or higher**
- Download from: https://www.python.org/downloads/
- During installation, check **"Add Python to PATH"**

### Python Libraries
Install all required libraries by running these commands in your terminal:

```bash
pip install pygame
pip install stable-baselines3
pip install gymnasium
pip install numpy
```

Or install all at once:
```bash
pip install pygame stable-baselines3 gymnasium numpy
```

To verify everything is installed correctly:
```bash
pip show pygame stable-baselines3 gymnasium numpy
```

---

## Project Structure

---

## How to Run

### 1. Clone or download the project
```bash
git clone https://github.com/YOUR_USERNAME/snakes-lenders.git
cd snakes-lenders
```

### 2. Install libraries
```bash
pip install pygame stable-baselines3 gymnasium numpy
```

### 3. Train the Hard AI (PPO) — do this once before playing Hard mode
```bash
python main.py --train
```
This takes **20–40 minutes**. The trained model is saved to `ai/ppo_model.zip`.

For a smarter AI (takes longer):
```bash
python main.py --train --steps 500000
```

### 4. Run the game
```bash
python main.py
```

---

## Game Modes

| Command | Mode |
|---------|------|
| `python main.py` | Human vs Human (Pygame UI) |
| `python main.py --mode hvai` | Human vs Easy AI |
| `python main.py --mode hvai --hard` | Human vs Hard AI |
| `python main.py --mode aivai` | Easy AI vs Hard AI |
| `python main.py --console` | Play in terminal instead of UI |
| `python main.py --phase 1` | Test board generation only |
| `python main.py --train` | Train the PPO Hard AI |
| `python main.py --train --steps 500000` | Train with more steps |

---

## Controls (Pygame UI)

| Key | Action |
|-----|--------|
| **SPACE** | Roll dice / continue |
| **B** | Open snake shop (human players only) |
| **ENTER** | Confirm number input in shop |
| **ESC** | Cancel shop input |
| **Q** | Quit the game |

---

## Game Rules

### Board
- 10×10 board, 100 tiles, randomized every game
- 7 ladders (move up), 4 initial snakes (move down), 5 bomb tiles (lose points)
- Players start off the board at tile 0

### Movement
- Players roll a 6-sided die each turn
- Landing on a ladder bottom → climb to the top
- Landing on a snake head → slide to the tail
- If a roll overshoots tile 100 → bounce back (e.g. tile 98 + roll 4 = tile 96)

### Points
- Each tile has a point value — earned when you land on it
- Bomb tiles deduct 30 points
- Tile 100 gives a 200 point bonus
- Going below 0 points → bankrupt → return to tile 0

### Snake Shop
- Spend points to place snakes on the board to hinder opponents
- Snake cost: `length × 10 × multiplier`
  - 1st snake: 1.0× multiplier
  - 2nd snake: 1.5× multiplier
  - 3rd snake: 2.0× multiplier
- Maximum 3 snakes per player
- Snake head must be between tiles 20–80
- Cannot place snakes on occupied tiles or ladder tiles
- Cannot chain snakes (head cannot be at another snake's tail)

### Winning
- First player to reach tile 100 wins

---

## AI Algorithms

### Easy Mode — Expectimax
- Makes decisions by calculating **expected value** across all 6 dice outcomes
- Evaluates whether buying a snake is profitable using ROI calculation
- No training needed — works immediately
- Consistent and predictable behavior

### Hard Mode — PPO (Proximal Policy Optimization)
- Uses a **neural network** trained via reinforcement learning
- Learns strategies by playing thousands of simulated games
- Requires training before use (`python main.py --train`)
- More adaptive and less predictable than Expectimax

---

## Note

The trained PPO model file (`ai/ppo_model.zip`) is **not included** in this
repository because it is specific to each machine's training run.

Each person needs to train their own model by running:
```bash
python main.py --train
```

This takes 20–40 minutes. For a better model:
```bash
python main.py --train --steps 500000
```