# Snakes & Lenders — Knowledge Graph

Visual index of how branches, commits, code modules, game features, training
pipeline, and docs connect. Built from this branch's commit history,
`knowledge/*`, and current source. Renders as Mermaid on GitHub.

---

## 1. Branch & commit lineage

How `refactor/economy-ai-overhaul` arrived at its current tip (`5364af0`).

```mermaid
flowchart LR
  classDef main fill:#1a3a52,stroke:#5ab3ff,color:#e8f4ff
  classDef refactor fill:#3a1a52,stroke:#b35aff,color:#f4e8ff
  classDef webapp fill:#1a523a,stroke:#5affb3,color:#e8fff4
  classDef feat fill:#523a1a,stroke:#ffb35a,color:#fff4e8

  m["6946a39<br/>main: ship PPO model"]:::main

  m --> bm["460d563<br/>feat: PPO beast mode<br/>(22/5 code, 14/4 model)"]:::refactor

  m --> wa1["06d1390<br/>Merge PR #1 web-app2"]:::webapp
  wa1 --> wa2["29105e9<br/>refactor(web):<br/>engine-driven thin client"]:::webapp
  wa2 --> wa3["2201ca9 / 2420dd9<br/>gitignore .venv"]:::webapp
  wa3 --> wa4["31d9472<br/>bgm/responsiveness/emote"]:::webapp
  wa4 --> wa5["676d742<br/>JJEEYYSSEE: bgm proc<br/>autoplay-unlock + replay"]:::webapp

  wa4 --> ft["cc5163a<br/>feat/cunning-ppo-rebuild<br/>cunning 22/5 model + continuation fix"]:::feat

  wa4 --> mwa["55fb1a6<br/>Merge PR #2:<br/>web-app → refactor"]:::refactor
  bm --> mwa
  mwa --> mft["69bc1ff<br/>Merge PR #3:<br/>feat → refactor"]:::refactor
  ft --> mft
  mft --> cp["5364af0<br/>cherry-pick 676d742<br/>(refactor HEAD)"]:::refactor
  wa5 -. cherry-picked .-> cp
```

**Color key:** blue = `main`, purple = `refactor/economy-ai-overhaul`,
green = `web-app`, orange = `feat/cunning-ppo-rebuild` (disposable).

---

## 2. Code module graph (Python + JS)

Solid arrow = Python import. Dashed arrow = HTTP call.

```mermaid
flowchart TD
  classDef entry fill:#1a3a52,stroke:#5ab3ff,color:#e8f4ff
  classDef game fill:#1a523a,stroke:#5affb3,color:#e8fff4
  classDef ai fill:#523a1a,stroke:#ffb35a,color:#fff4e8
  classDef web fill:#3a1a52,stroke:#b35aff,color:#f4e8ff
  classDef test fill:#523a3a,stroke:#ff5a5a,color:#ffe8e8

  main["main.py<br/>CLI + setup"]:::entry
  server["server.py<br/>Flask, stateful Session<br/>PPO try/except per turn"]:::entry
  console["game/console_game.py"]:::entry

  models["game/models.py<br/>Player, Snake, Ladder, BoardState<br/>BANKRUPT_IMMUNITY_TURNS=6"]:::game
  engine["game/engine.py<br/>do_turn, move_player,<br/>_steal_points, _apply_snakes"]:::game
  board["game/board.py<br/>generate_board, BFS validation"]:::game
  glog["game/log.py<br/>VERBOSE + gprint"]:::game

  exp["ai/expectimax.py<br/>Easy bot + strong_decision<br/>+ propose_combo/cheap/big/lurk"]:::ai
  ppo["ai/ppo_agent.py<br/>22-dim encode_state, N_ACTIONS=5<br/>train_ppo (continuation), ppo_decision"]:::ai

  appjs["web/app.js<br/>thin client (canvas + audio +<br/>animations + replay + steal SFX)"]:::web
  html["web/index.html"]:::web
  css["web/style.css"]:::web

  tests["tests/test_core.py<br/>14 unittest cases"]:::test

  main --> server
  main --> console
  main --> board
  main --> ppo
  main --> exp

  server --> board
  server --> engine
  server --> models
  server --> ppo
  server --> exp

  console --> engine
  console --> board
  console --> ppo
  console --> exp

  ppo --> exp
  ppo --> engine
  ppo --> models
  exp --> engine
  exp --> models

  engine --> models
  engine --> glog
  board --> models
  board --> glog

  appjs -. HTTP /api/new, /turn,<br/>/buy, /shop-options, /quit .-> server
  html --> appjs
  html --> css

  tests --> engine
  tests --> board
  tests --> models
```

---

## 3. Feature → implementation map

Each gameplay/AI feature and the code that realizes it.

```mermaid
flowchart LR
  classDef feat fill:#3a1a52,stroke:#b35aff,color:#f4e8ff
  classDef code fill:#1a523a,stroke:#5affb3,color:#e8fff4

  F1["Cunning PPO<br/>(22-dim / 5-action)"]:::feat
  F2["Combo action<br/>(snake tail on bomb)"]:::feat
  F3["Point stealing<br/>(STEAL_FLAT=15, STEAL_PCT=0.30)"]:::feat
  F4["Bankruptcy-immunity<br/>(anti death-loop)"]:::feat
  F5["Stochastic inference<br/>+ ent_coef=0.03"]:::feat
  F6["Anti-hoard penalty<br/>+ heavy sabotage reward"]:::feat
  F7["Frozen self-play<br/>shape guard"]:::feat
  F8["train_ppo<br/>continuation"]:::feat
  F9["Engine-driven web<br/>(thin client)"]:::feat
  F10["BGM autoplay-unlock<br/>+ replay system"]:::feat
  F11["Steal SFX<br/>+ bankruptcy animation"]:::feat
  F12["Server PPO<br/>try/except fallback"]:::feat

  C1["ai/ppo_agent.py<br/>encode_state, N_ACTIONS=5,<br/>_action_to_shop, ppo_decision"]:::code
  C2["ai/expectimax.py<br/>propose_combo"]:::code
  C3["game/engine.py<br/>_steal_points, _apply_snakes,<br/>do_turn (immunity tick)"]:::code
  C4["game/models.py<br/>BANKRUPT_IMMUNITY_TURNS,<br/>Player.deduct_points/go_bankrupt"]:::code
  C5["ai/ppo_agent.py<br/>train_ppo (PPO.load +<br/>reset_num_timesteps=False)"]:::code
  C6["ai/ppo_agent.py<br/>build_training_env<br/>shape-check on frozen opp"]:::code
  C7["server.py<br/>_ai_decision try/except"]:::code
  C8["web/app.js<br/>thin client + steal SFX +<br/>setupAudioAutoplayUnlock +<br/>replay state"]:::code
  C9["web/index.html<br/>+ web/style.css"]:::code

  F1  --> C1
  F1  --> C2
  F2  --> C1
  F2  --> C2
  F2  --> C3
  F3  --> C3
  F3  --> C4
  F4  --> C4
  F4  --> C3
  F5  --> C1
  F6  --> C1
  F7  --> C6
  F8  --> C5
  F9  --> C7
  F9  --> C8
  F10 --> C8
  F10 --> C9
  F11 --> C8
  F12 --> C7
```

---

## 4. Training & inference pipeline

Two-stage training recipe → shipped model → in-game decisions.

```mermaid
flowchart TD
  classDef stage fill:#523a1a,stroke:#ffb35a,color:#fff4e8
  classDef code fill:#1a523a,stroke:#5affb3,color:#e8fff4
  classDef file fill:#3a1a52,stroke:#b35aff,color:#f4e8ff

  S1["Stage 1 — base<br/>train_ppo(3_000_000,<br/>opponent_pool=True)"]:::stage
  S2["BACKUP<br/>copy ppo_model.zip → backup"]:::stage
  S3["Stage 2 — self-play polish<br/>train_ppo(2_000_000,<br/>opponent_pool=True)"]:::stage
  S4["Eval<br/>200g vs Easy / vs Strong"]:::stage

  T1["build_training_env<br/>vec_env (n=4)<br/>deciders = {Easy, Strong, frozen self*}<br/>*shape guard skips legacy"]:::code
  T2["PPO continuation:<br/>PPO.load(env=vec_env, device=cpu)<br/>if shape matches"]:::code
  T3["model.learn(<br/>total_timesteps, reset_num_timesteps=False)"]:::code
  T4["model.save → ai/ppo_model.zip"]:::code

  I1["server preload<br/>load_ppo_model()"]:::code
  I2["per turn:<br/>_ai_decision → ppo_decision<br/>(try/except → Expectimax)"]:::code
  I3["encode_state (22-dim) →<br/>model.predict(deterministic=False) →<br/>_action_to_shop"]:::code
  I4["do_turn(shop_decision) →<br/>buy_snake → roll → move →<br/>snakes/ladders/bombs →<br/>_steal_points + immunity tick"]:::code

  M1[("ai/ppo_model.zip<br/>shipped 3M cunning")]:::file
  M2[("ai/ppo_model_backup.zip<br/>last known-good restore point")]:::file
  M3[("ai/ppo_model_prev_reserved.zip<br/>DELETED")]:::file

  S1 --> T1
  T1 --> T2
  T2 --> T3
  T3 --> T4
  T4 --> M1
  M1 --> S2
  S2 --> M2
  S2 --> S3
  S3 --> T1
  S3 --> S4
  S4 -. regress .-> M2
  M2 -. restore .-> M1

  M1 --> I1
  I1 --> I2
  I2 --> I3
  I3 --> I4

  M3 -. removed in merge .- M2
```

---

## 5. Documentation topic map

Where to look for what. Doc → topic coverage.

```mermaid
flowchart LR
  classDef doc fill:#1a3a52,stroke:#5ab3ff,color:#e8f4ff
  classDef topic fill:#523a3a,stroke:#ff8888,color:#ffe8e8

  D1["README.md<br/>(external)"]:::doc
  D2["CHANGELOG.md<br/>(team)"]:::doc
  D3["knowledge/overview.md"]:::doc
  D4["knowledge/architecture.md"]:::doc
  D5["knowledge/training.md"]:::doc
  D6["knowledge/status.md"]:::doc
  D7["knowledge/revisions.md"]:::doc
  D8["knowledge/manuscript.md"]:::doc
  D9["knowledge/graph.md<br/>(this file)"]:::doc

  T1["How to run / install"]:::topic
  T2["Game rules<br/>(snakes, bombs, economy)"]:::topic
  T3["Code map<br/>(modules, functions)"]:::topic
  T4["PPO training<br/>(two-stage recipe, eval)"]:::topic
  T5["Current status + caveats"]:::topic
  T6["Change log<br/>(P0/P1/P2 + cunning rebuild)"]:::topic
  T7["Manuscript divergence<br/>(PDF vs code)"]:::topic
  T8["Branches + merge history"]:::topic

  D1 --> T1
  D1 --> T2
  D1 --> T4
  D2 --> T2
  D2 --> T6
  D3 --> T1
  D3 --> T2
  D4 --> T3
  D5 --> T4
  D6 --> T1
  D6 --> T5
  D7 --> T6
  D8 --> T7
  D9 --> T3
  D9 --> T4
  D9 --> T8
```

---

## 6. People → contributions

```mermaid
flowchart LR
  classDef p fill:#3a1a52,stroke:#b35aff,color:#f4e8ff
  classDef w fill:#1a523a,stroke:#5affb3,color:#e8fff4

  P1["Geuel John Rivera<br/>(repo owner)"]:::p
  P2["JJEEYYSSEE<br/>(Caparas)"]:::p
  P3["Enami345 / web-app2 author"]:::p

  W1["Economy + AI overhaul<br/>(refactor/, beast mode)"]:::w
  W2["Cunning PPO rebuild<br/>(22/5, continuation fix)<br/>cc5163a"]:::w
  W3["Premium Flask web app<br/>(PR #1)"]:::w
  W4["Engine-driven thin client<br/>refactor 29105e9"]:::w
  W5["BGM/responsiveness/emote<br/>31d9472"]:::w
  W6["BGM autoplay-unlock + replay<br/>676d742"]:::w

  P1 --> W1
  P1 --> W2
  P1 --> W4
  P3 --> W3
  P1 --> W5
  P2 --> W6
```

---

## How to update this graph

When you ship a change that adds a new feature, file, or branch:

1. Add the node to whichever section it belongs in (feature → §3, code →
   §2, training change → §4, doc → §5, branch/commit → §1, contributor → §6).
2. Wire it to its neighbors (don't add isolated nodes — the value of this
   doc is the edges).
3. Keep each subgraph ≤ ~20 nodes; if a section is bursting, split it.
4. Don't paste full code or paragraphs of prose here — the node labels
   should point at the source file or doc that has the details.
