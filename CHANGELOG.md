# Changes — `refactor` branch (for the team)

Heads up team — this branch is a big gameplay + AI overhaul. Read this before
you pull so the new rules don't surprise you.

## Latest additions (web UI, multiplayer, stronger AI)
- **Web UI** (`python main.py --web` → http://localhost:8000). Plain
  HTML/CSS/JS, no extra deps — this is the new primary UI (teammate will
  restyle `ui/web/style.css`). Pygame still works.
- **2-4 players, any mix of humans and AI**, shuffled turn order. Setup is
  asked on launch (or `--players/--humans/--difficulty`).
- **Stronger Hard AI:** retrained 2.5M steps with **self-play / opponent pool**.
  Wins ~89-91% vs Easy, ~89% vs a strong heuristic, dominates 4-player FFAs.
  Model shipped + a backup fallback. Full numbers in `knowledge/training.md`.
- Snakes are **single-use traps with a strike range** and **steal points**; your
  own snakes don't bite you. (See rules below.)
- `--mode hvai/aivai` flags are gone — use `--players/--humans/--difficulty`.

## TL;DR
- The **economy is now the core of the game.** Points are scarce; you spend them
  on snakes to sabotage opponents. Just rolling and ignoring the shop will lose.
- **Two real difficulties:** Easy (a weak, beatable bot) and Hard (a trained PPO
  agent that beats Easy ~90% of the time and a strong heuristic ~89%).
- A **trained PPO model is included** (`ai/ppo_model.zip`) — Hard mode works out
  of the box, no training step required.

## What to do after pulling
```bash
git checkout <this-branch>
python -m venv venv && venv\Scripts\activate     # if you don't have one yet
pip install -r requirements.txt
python main.py --web                             # play in the browser
# or: python main.py --players 2 --humans 1 --difficulty hard
```
You do **not** need to retrain — the model is committed. (Optional: `python
main.py --train` to regenerate it.)

## New / changed rules
- **Exact roll to win:** overshooting tile 100 = invalid move, you stay put.
- **Snakes:**
  - Player-placed snakes have a **strike range** (head + a few tiles below), so a
    well-placed trap reliably catches a passing opponent. Jumping clean over is
    safe. **Your own snakes don't bite you.**
  - A bite **slides you back AND steals your points** to the snake's owner.
  - Player snakes are **single-use** (consumed when they fire — then re-place).
  - Board snakes are unchanged terrain (exact-head only).
  - You can't build a **wall** of adjacent snake heads.
- **Economy:** tile income is low; snake pricing is sub-linear (save up for a big
  one). **Bombs scale with depth and can bankrupt you → back to tile 0.**

## Code changes (where to look)
- `game/engine.py` — movement, strike-range snakes, owner immunity, point theft,
  scaled bombs, snake pricing, anti-wall rule.
- `game/board.py` — scarce tile income.
- `game/models.py` — bankruptcy behavior.
- `ai/expectimax.py` — Easy bot (weakened) + catch-optimal placement strategies.
- `ai/ppo_agent.py` — fixed/expanded PPO env (4-action strategy space), reward,
  inference; retrained model.
- `main.py` — UTF-8 console output (fixes a Windows crash with the emoji logs).
- `knowledge/` — full design notes and the decision log behind all of this.

## Known notes
- Hard mode is intentionally tough (that's the point). Easy is the gentle one.
- If `ai/ppo_model.zip` ever fails to load, Hard mode auto-falls back to the
  Expectimax bot so the game still runs.
