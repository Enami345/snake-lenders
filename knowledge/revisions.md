# Snakes & Lenders — Revision Checklist

Ranked by importance. P0 = breaks core premise or grader claims. P1 = real bugs. P2 = quality/polish.

---

## P0 — Critical (fix or the project's claims are false)

- [x] **Make economy load-bearing — ignoring it must be a LOSING strategy.** Root design flaw: roll-only player did fine, strategic layer optional. DONE — strategy beats rolling (acceptance passed).
  - TRIED **pass-over** (land on OR jump past head) → made snakes reliably fire BUT softlocks: any snake length>6 is an impassable wall (from the tail you can't out-roll the head). Reverted.
  - TRIED **one-shot pass-over** (immune after first bite) → no softlock, reliably load-bearing, but rejected by user (wanted exact-head + no immunity).
  - FINAL RULES (user choice): **exact-head landing only**, no immunity, anti-wall placement rule (`MAX_HEAD_RUN=2`, `_head_run_length`).
  - [x] Exponential pricing (manuscript): `base × purchase_count × length^1.3` (`engine.py`).
  - [x] Raised player snake head cap 80→90 (`MAX_SNAKE_HEAD`).
  - [x] Anti-softlock placement rule: a player snake can't extend a run of adjacent heads past 2.
  - Naive shop AI failed acceptance (46%). FIX = make the AI **cunning** (user direction), not change the rules:
    - [x] Hold the 3-snake budget until leading opponent reaches `SABOTAGE_MIN_POS` (25) — late-game knockbacks erase more progress.
    - [x] Value placements by: head in opponent's immediate dice range (`HIT_IN_RANGE`), max setback (try tail=1), and targeting advanced opponents (`progress_weight`). Forces re-climbs through lower snakes.
    - [x] Search tries max-setback tails (incl. tail=1) so a hit is brutal.
  - Intermediate: high-income version passed at 59% but economy wasn't a real constraint (points free).
  - ✅ **FINAL DESIGN (economy made binding for both players):**
    - Scarce tile income (~4-14/turn, `BASE_TILE_VALUE=3`) — points must be managed.
    - Depth-scaled bombs (`BOMB_BASE=18 + pos//6`) that **cause real bankruptcy** (~0.32/game) when low.
    - Player snakes = **single-use traps with a strike range** (`STRIKE_ZONE=2`, ~50% catch), consumed on fire → re-placeable. Board snakes stay exact-head terrain (no game-drag).
    - **Sub-linear pricing** (`PRICE_ALPHA=0.9`) so the AI/player can SAVE UP for a long, devastating, well-placed snake — strategy, not spam.
    - Cunning AI keeps a bomb-survival `SAFETY_BUFFER`, targets advanced opponents, places ~1.8 high-value snakes/game.
  - ✅ **ACCEPTANCE PASSED:** strategic-AI vs roll-only = **59%** over 200 games; AI-vs-AI ~50% (balanced); ~52-turn games; bankruptcy visible. Strategy clearly beats rolling without brute-force spam.
- [x] **Fix ROI cost weighting** — `ai/expectimax.py` `evaluate_snake_placement` rewritten as expected DAMAGE (points): `setback × TILE_VALUE × p_hit × progress_weight`. Removed broken `cost*0.01`. Buy gate lowered 100→MIN_SNAKE_COST; removed dead `ev_roll`; cap uses MAX_SNAKE_HEAD. (Final form is the cunning model under exact-head — see #1.)
- [x] **Fix PPO training env turn handling** — `ai/ppo_agent.py` `step()` rewritten. One step = agent's turn (policy action) THEN opponents take their own `expectimax_decision` turns until it's the agent's turn again. Reward/obs always agent-perspective. Verified turns alternate, opponent plays independently. Added UTF-8 stdout in `main.py` so emoji/arrow logs don't crash Windows training.
- [x] **Rebalance PPO reward** — `ai/ppo_agent.py` `_compute_reward`. Win/loss ±100 dominates; shaping now = **progress delta** `(pos-start_pos)*0.1` (bounded over a game, was absolute position each step → 577 cumulative, now ~2-10). Bankruptcy penalty fixed to fire on real bankruptcy (`bankrupt_count` delta, was dead `points<0`). Snake-buy nudge +2.
- [x] **PPO upgrade + retrain** — PPO was only Discrete(2) [roll/buy-best] and delegated placement to Expectimax → no real skill edge. Expanded to **Discrete(4)**: roll / cheap trap / save-for-big-snake / win-denial lurk (`_action_to_shop`, `propose_*`, catch-optimal). Reward credits opponent setback. Owner-immunity + strike-zone=4 + difficulty split (Easy deliberately weak). Retrained vs weak Easy. **DONE: PPO Hard beats Easy ~94% over 300 games** (target was ≥75%).
- [x] **Annoyance layer** — (1) snake bites **steal points** to owner (`_steal_points`, drives bankruptcy + self-funds traps), (2) AI **win-denial lurk** snakes near goal (`WIN_DENIAL_*`), (3) **aggressive** AI (low `SAFETY_BUFFER=12`, `SABOTAGE_MIN_POS=10`). Sim: AI 57% vs roll-only, bankruptcies up 0.32→0.78/game, snakes 1.8→4.3/game.

## P1 — Significant bugs

- [x] **Fix first-move ladder skip** — `game/engine.py` `move_player`. Merged entry + normal paths so a player entering onto a ladder bottom now climbs. Verified (entry roll onto ladder bottom 3 → climbs to 40).
- [x] **Remove dead bankruptcy penalty** — fixed during #4: now detects real bankruptcy via `bankrupt_count` delta instead of the never-true `points < 0`.
- [x] **Exact-roll-to-win** — `game/engine.py` `move_player` now stays put on overshoot (was bounce-back). Exact roll required to land on 100. Easy-AI EV sim + README/knowledge updated to match. NOTE: endgame stall remains (no snakes above tile 80) — separate item: consider raising snake head cap to make near-100 strategic.
- [WONTFIX / by design] **Bomb/bankruptcy swinginess** — user chose bankruptcy = reset to tile 0 ("bankruptcy must start at beginning"). Brutal variance is intended. Agency comes from keeping a bomb-survival buffer (the AI does this).

## P2 — Quality / polish

- [x] **Add tests** — `tests/test_core.py` (14 unittest cases): board counts+solvability, cost monotonic/floor/rises, overshoot stay-put, exact-roll win, entry ladder climb, strike range, clean-jump safe, owner immunity, board-snake exact-head, head cap, no-wall, no-chain, bomb bankruptcy. `python -m unittest tests.test_core`.
- [~] **Decouple PPO action from Expectimax** — partial: PPO now Discrete(4) strategy space (roll/cheap/big/lurk) with catch-optimal placement helpers. Net picks strategy, not exact tiles. Good enough; full per-tile control left.
- [x] **3-4 player support** — done: `main.py` setup (players 2-4 / humans 0-N / difficulty) + shuffled turn order; `console_game.play_game` takes prebuilt board; renderer N-player generic. (Caveat: 4×Hard FFA drags — left by design.)
- [x] **Silence library stdout** — `game/log.py` `gprint` gates gameplay prints (board gen, AI buys, bankruptcy); `train_ppo` flips `VERBOSE=False` during `model.learn`.
- [x] **Remove dead code** — removed unreachable `pygame.quit()` in `renderer.run`. KEPT `evaluate_state`/`expected_value_after_roll` deliberately (document the Expectimax EV/chance-node concept the manuscript describes).
- [x] **Console unicode crash (Windows)** — UTF-8 stdout in `main.py`.
- [x] **UI shop head cap** — was hardcoded `20-80`, now uses `MAX_SNAKE_HEAD` (90).
- [x] **Fix README** — model now included; README rewritten.
- [WONTFIX] **Align manuscript vs code** — manuscript is a PDF (can't edit here); code diverged further (sub-linear pricing, strike range, point steal). Divergence is documented in `knowledge/manuscript.md`.

## Beyond P0-P2 (this session, post-push)

- [x] **Web UI** (`ui/web/`) — stdlib http.server + HTML/CSS/JS; setup screen → board; resumes on refresh. `python main.py --web`. Removed old `--mode` flags. Pygame kept.
- [x] **PPO improvements (opponent pool / self-play)** — `train_ppo(opponent_pool=True)` trains vs {Easy, Strong heuristic, frozen best PPO}; added `expectimax.strong_decision`. Trained 2.5M; promoted the self-play model to `ai/ppo_model.zip` (more robust vs Strong: 83%→89%; ~tie head-to-head vs plain 2.5M). Removed stale `ppo_model_100k.zip`; refreshed backup.
- [x] **Model load resilience** — `load_ppo_model` tries main → backup (survives mid-write training); graceful Expectimax fallback. AI logs use real player names + `(PPO)` tag.
- [x] **AI training + quality fully documented** — see [training.md](training.md) (timesteps↔games, plateau, model lineage, full win-rate battery).
