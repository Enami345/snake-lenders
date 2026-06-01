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

## Gameplay-feel pass (earlier)

- [x] **Snakes → exact-head only** (`STRIKE_ZONE=0`) — bite only on landing exactly on a head (user's chosen rule). Removes "near-miss" bites.
- [x] **Removed point-stealing** — bite just slides to tail. Killed the bankruptcy death-spiral; passive-human bankruptcies ~0.21/game. **(Later REVERTED — see Cunning rebuild below.)**
- [x] **Mixed Easy/Hard AIs** — `build_players(n_players, n_humans, n_hard)`; menu "Hard AIs" count; CLI `--hard-ais N` (`--difficulty` = all/none shortcut).
- [x] **Ladder anti-clutter** — jump capped 5–20; endpoints spaced ≥ `MIN_LADDER_GAP=6` so ladders distribute across the board (no overlapping pile).
- [x] **Web polish** — title page → config → **loading overlay (progress bar)** → board; **per-step token animation** (`do_turn` returns a `move` breakdown); **threaded server** (fixes browser hang); `/api/quit` reset so New game / refresh behave; POST error-handling (no silent hangs).
- ⚠️ **Tradeoff (then):** exact-head made snakes weak → 14-dim PPO ~level with Easy in bot-vs-bot sims. Triggered the cunning rebuild below.

## Engine-driven web (no duplicated game logic)

- [x] **Web is now a thin client over the Python engine** — a teammate's premium Flask UI (`server.py` + `web/`) had **reimplemented all rules in JS** (with stale strike-range + steal). Removed that duplication; the engine in `game/` is the single source of truth.
- [x] **`server.py` → stateful, engine-driven.** Endpoints: `/api/new`, `/api/state`, `/api/shop-options` (valid placements from `can_place_snake`+`calculate_snake_cost`), `/api/buy` (`buy_snake`), `/api/turn` (`do_turn`; AI auto), `/api/quit`, `/api/generate-board` (lobby preview).
- [x] **`web/app.js` gutted of logic** — deleted JS movement/cascade/`getStrikingSnake`/steal/bankruptcy + cost/canPlace mocks + replay re-sim. Turns call `/api/turn` and animate the reported `move`; buys via `/api/buy`. Kept canvas render + audio + animations.
- [x] **Chess-style snake placement** — 🎯 button → click glowing HEAD → glowing affordable TAILs (from `/api/shop-options`) → confirm dialog (cost + projected points) → head→tail grow in player color → `/api/buy`.
- [x] **Web bug fixes:** shop now actually opens (players weren't given `snake_count`); dice face shows the real rolled number; tile point values drawn on cells; **bomb + bankruptcy animation** (blast + spin to tile 0); robust `api()` error (no JSON-parse crash on stale server). Web port is **5000** (Flask).
- Note: `ui/web/` (old stdlib UI) is now unused/legacy; `requirements.txt` includes `flask`, `flask-cors`.

## Cunning PPO rebuild (latest)

Triggered by user feedback that the 14-dim PPO was "too reserved — just waits
and drops one snake late." User direction: "extra ruthless, master cunning
expert that has sole purpose to horribly sabotage its enemies — very
unpredictable, very calculating, restored stealing OK but no death-loop."

- [x] **PPO observation 14 → 22 dim** (`encode_state`) — added bomb/ladder
  lookahead, leader's distance to goal, combo availability flag, and the
  agent's bankruptcy-immunity flag.
- [x] **PPO actions 4 → 5** (`N_ACTIONS=5`) — added **`combo`** action that
  places a snake whose **tail lands on a bomb tile** so the victim eats the
  knockback AND the bomb damage.
- [x] **`propose_combo`** added in `ai/expectimax.py` — picks the highest-
  setback placement whose tail is a bomb, catch-optimal offset from the
  leading opponent.
- [x] **Cunning reward shaping** in `_compute_reward`:
  - `+0.5 × opp_setback` (heavy sabotage reward)
  - `+12.0` on a combo placement (MAXIMUM annoyance)
  - `+4.0 + 0.3 × placed_setback` on any snake buy
  - **`-1.0` anti-hoard** if the agent passes on buying while sitting on
    >100 pts (kills the "one late snake" passive policy)
  - `-50.0` real-bankruptcy penalty (unchanged)
- [x] **Stochastic inference** (`ppo_decision`: `deterministic=False`) +
  **`ent_coef=0.03`** in training → varied, unpredictable policy that mixes
  patience, cheap pressure, big knockbacks, lurks, and combos.
- [x] **`device="cpu"`** forced in PPO config — CPU is faster than the
  low-utilization GPU path for this small MLP (SB3 even warns about it).
- [x] **Point-stealing restored** (`game/engine.py` `_steal_points`,
  `STEAL_FLAT=15`, `STEAL_PCT=0.30`) — only fires on player-owned snake
  bites; bites by board terrain still don't steal.
- [x] **Anti death-loop: bankruptcy-immunity cooldown**
  (`game/models.py` `BANKRUPT_IMMUNITY_TURNS=6`, `Player.bankrupt_immune`).
  After a bankruptcy, further losses clamp the wallet to 0 instead of
  resetting position again for 6 of the player's own turns. Ticked down in
  `engine.do_turn`.
- [x] **Frozen self-play shape guard** in `build_training_env` — only adds
  the existing `ai/ppo_model.zip` as a frozen opponent if its
  `observation_space.shape == (22,)` and `action_space.n == N_ACTIONS`,
  otherwise prints a warning and skips. Prevents `predict()` from crashing
  mid-training on a legacy 14-dim/4-action backup.
- [x] **Server PPO try/except per turn** (`server.py` `_ai_decision`) — if
  `ppo_decision` raises (e.g. shape mismatch while a retrain is mid-write),
  the Hard AI falls back to `expectimax_decision` for that turn so the game
  doesn't crash.
- [x] **Two-stage training recipe** documented in `training.md`: stage 1 = 3M
  base (frozen self-play auto-skipped if shape mismatches), stage 2 = 1.5-2M
  self-play polish (stage-1 model now joins the frozen pool).
- [x] **`train_ppo` continuation fix** — original `train_ppo` always built a
  fresh `PPO(...)` and overwrote `ai/ppo_model.zip` on save, so a "stage 2"
  run was really a 2M-from-scratch run. First stage-2 attempt regressed
  64%→52% WR vs Easy for exactly this reason (recovered from the manual
  `ppo_model_backup.zip`). Patched `train_ppo` to **load and continue** from
  the existing model when its shape matches (`PPO.load(..., env=vec_env)`,
  `learn(reset_num_timesteps=False)`); falls back to a fresh PPO with a logged
  reason otherwise. Multi-stage runs now actually accumulate steps.
- [x] **Backup discipline documented** — always
  `copy ai\ppo_model.zip ai\ppo_model_backup.zip` before a polish run so a
  regression is recoverable with the reverse copy.
- [x] **Eval (cunning model, 200 games each, stochastic, stage-1 3M base):**
  - vs Easy: **64%** WR, 4.18 snakes/game, 3.98 combos/game, 11.4 avg
    setback, 1.6 steals → PPO/game, 0.30 PPO self-bankrupt/game.
  - vs Strong: **62%** WR, 4.12 snakes/game, 4.00 combos/game, 10.9 avg
    setback, 1.6 steals → PPO/game, 0.45 PPO self-bankrupt/game.
  - Action mix vs Strong (5108 buys): combo 1755 > lurk 1598 > cheap 729 >
    big 622 > (roll 1403 non-buy). No single action >35% → stochastic
    policy working.

## Web UI Polish + Bug Fixes (post-cunning rebuild)

- [x] **CSS root cause fixed** — unclosed `@media (max-width: 480px)` at line
  1446 silently trapped every rule below it (confirm-snake overrides, vstat
  styles, animations, replay rules). Closed the media query after `.board-frame`.
  No rule changes — just un-trapped everything that was already written.
- [x] **Confirm Snake modal** — rebuilt as floating 320px card at bottom-center
  with transparent backdrop (board visible/clickable behind). `#modal-snake-confirm`
  ID-scoped CSS so nothing overrides it. Thin divider rows, vertical button stack,
  `✦ Confirm Snake` / `✓ Buy & Place` / `↺ Pick again` icons.
- [x] **Replay bar** — removed duplicate `.replay-panel` block (was overriding
  the proper full-width bar with a narrow 420px card). Rewritten as
  `position:fixed bottom:24px` centered bar so it never bleeds into the board
  layout and causes the "black half-panel" dead space.
- [x] **Victory stats now tracked** — parse engine turn logs each
  `executeTurn()` to count `bittenCount`, `bankruptCount`, `laddersClimbed`,
  `bombsHit`, `turnsTaken`, `pointsStolen` (AI-placed `snakesPlaced` also from
  `🛒` log). Previously only human `snakesPlaced` incremented; all others
  stayed 0.
- [x] **No-cache Flask headers** — `SEND_FILE_MAX_AGE_DEFAULT=0` +
  `after_request` hook sends `Cache-Control: no-store` so CSS/JS edits land on
  a normal browser reload without needing DevTools cache bypass.
- [x] **In-game rules text updated** — Quick Rules (lobby) + How to Play modal
  updated: steal mechanic explained (flat 15 + 30%, owner gets points), 6-turn
  immunity cooldown added, Hard AI description updated (5 strategies, combo
  strikes, stochastic timing, point theft on bite). Removed outdated
  "no point-stealing" line.
- [x] **Stage-2 retrain (5M total continuation)** — `train_ppo` continuation
  patch working; stage-2 ran 2M on top of 3M base. Eval showed marginal
  improvement (WR +3.5pp vs Easy, +1.5pp vs Strong) but H2H lost to stage-1
  (-7.4pp, shorter avg setback 8.7 vs 11.0). Stage-1 backup restored as
  primary; stage-2 noted as "spammer vs sniper" behavioral tradeoff.
- [x] **knowledge/graph.md** — 6 Mermaid knowledge graphs added (branch
  lineage, code modules, feature→implementation, training pipeline, doc topic
  map, contributors). Graphify skill installed for Windows (`graphifyy`);
  requires `ANTHROPIC_API_KEY` env var to build interactive `graphify-out/graph.html`.
