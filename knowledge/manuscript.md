# Snakes & Lenders — Manuscript / Case Study Documentation

**File:** `docs/G10_SnakesLenders.pdf`  
**Submitted:** April 14, 2026  
**To:** Prof. Ria Sagum  
**Course:** BSCS 3-4 | Introduction to Artificial Intelligence | PUP

---

## I. Introduction

Snakes & Lenders is inspired by classic Snakes & Ladders but adds a strategic layer: players earn points from tiles and spend them to **buy snakes** to hinder opponents. Victory depends on smart planning and resource management, not just dice luck.

---

## II. Game Rules (from manuscript)

1. **Board Setup** — 10×10 board, randomized every game, 7 ladders, 4 initial snakes
2. **Movement** — dice roll; ladders = move up, snakes = move down
3. **Points** — earned from tiles, spent on snakes, deducted by bombs
4. **Snake Shop** — cost based on length + base value; max 3 per player; cost rises each purchase; cannot place on occupied tiles
5. **Bankruptcy** — points below zero → return to starting tile
6. **Win** — first to reach tile 100

---

## III. AI Algorithms (from manuscript)

### PPO (Hard AI)
- Reinforcement learning via self-play (thousands of simulations)
- "Clipping" mechanism = stable incremental updates, prevents strategy forgetting
- Chosen over other RL algos because of stability in high-randomness (dice) environments
- Rewards: win (+), collect money (+) | Penalties: bankruptcy (-)

**PPO pseudocode (from doc):**
```
Start with random AI.
For thousands of simulations:
  Step 1: Play — observe board, pick action (roll OR buy snake), save result
  Step 2: Evaluate — which actions led to good/bad outcomes
  Step 3: Update — adjust logic slightly toward good actions, away from bad
           (PPO clipping: limit how drastically logic can change at once)
```

### Expectimax (Easy AI)
- Designed for chance/randomness environments (unlike Minimax which assumes perfect prediction)
- Builds probability tree with "chance nodes" for dice outcomes
- Calculates "Expected Value" of each possible action
- Key question: "If I spend 100 pts on a snake now, what's the probability opponent lands on it (1/6 dice)?"

**Expectimax pseudocode (from doc):**
```
Function CalculateBestMove(board_state):
  AI decision node:
    Test all possible moves (Buy Snake A, Buy Snake B, just Roll)
    Simulate resulting board for each
    Pick move with highest average score

  Dice chance node:
    6 possible outcomes (1-6), each with 1/6 probability
    Simulate all 6 futures
    Return average outcome score
```

---

## IV. Software Architecture Diagram (from manuscript)

The manuscript includes a full architecture flowchart. Key layers:

```
Session Management
  → New game trigger · Player count · Difficulty (Easy=Expectimax / Hard=PPO)

Board Generation — BFS
  → Random seed → Ladders + Snakes + Bombs
  → BFS path check (min avg 10 turns)
  → FIX 1: Ladder skip guard (no tile 1-10 ladder jumping to 90+)
  → Loop check (ladder top not a snake head) → re-seed if invalid

Turn Loop
  → Roll dice → Move → Collect tile pts → Shop phase → Win check
  → Human: UI prompt | AI: policy decision

Trap Shop
  → FIX 4: Placement cap (head max tile 80)
  → Placement rules: no occupied tiles, head > tail, max 3/player
  → Purchase counter: max 3, price rises each purchase

Economy Engine — Exponential Pricing (manuscript design)
  → Cost = Base_Price × Length^α  (α starts at 1.3, tunable)
  → FIX 2: Alpha tuning guard (monitor PPO buy rate)
  → NOTE: Actual code uses linear formula: length × 10 × multiplier (1.0/1.5/2.0×)

AI Systems
  → Expectimax: Dice EV tree (1/6 per outcome) + Trap ROI (FIX 3: EV pre-computed, injected into PPO state vec)
  → PPO Agent: 14-dim state input (board + wallets + pos + Expectimax EV vec) → Policy (Roll or Buy)

Win Condition: First to tile 100
Bankruptcy Reset (was TBD in design): Points < 0 → return to tile 1 [NOW IMPLEMENTED]
```

### FIX labels from architecture doc:
- **FIX 1** — Ladder skip guard (no ladder jumping to tile 90+) ✓ implemented
- **FIX 2** — Alpha tuning guard for PPO economy
- **FIX 3** — Expectimax EV pre-computed and injected into PPO state vector ✓ implemented (see `encode_state` in `ppo_agent.py`)
- **FIX 4** — Snake head placement cap at tile 80 ✓ implemented

---

## Notes: Manuscript vs. Actual Code Differences

| Manuscript Design | Actual Implementation |
|---|---|
| Cost = Base × Length^α (exponential) | Cost = length × 10 × multiplier (linear, 1.0/1.5/2.0×) |
| Bankruptcy reset TBD | Implemented: position=0, points=0 |
| Bombs TBD | Implemented: 5 bombs, -30 pts |
| "Millions of simulations" for PPO | Actual: 100k steps default (4 parallel envs) |
| α = 1.3 tunable parameter | Not in code; simplified to fixed multipliers |
