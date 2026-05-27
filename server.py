"""
Snakes & Lenders — Flask Backend Server (STATEFUL, engine-driven)

The Python game engine in game/ is the single source of truth. The web
frontend (web/) is a thin view: it renders state and sends actions
(new game / buy snake / take turn); all rules run here via game.engine.

Run:
    python server.py            # http://localhost:5000
    python main.py --web
"""

import os
import sys
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.models import Player
from game.board import generate_board
from game.engine import (do_turn, buy_snake, can_place_snake,
                         calculate_snake_cost, MAX_SNAKE_HEAD)
from ai.expectimax import expectimax_decision
from ai.ppo_agent import load_ppo_model, ppo_decision

app = Flask(__name__, static_folder="web")
CORS(app)

# Load the PPO model once at startup (Hard AI). Falls back to Expectimax.
PPO_MODEL = None
try:
    PPO_MODEL = load_ppo_model()
    print("[PPO] Model pre-loaded for Hard AI.")
except Exception as e:
    print(f"[PPO] Preload warning: {e} — Hard AI falls back to Expectimax.")


# ── Single in-memory game session ─────────────────────────────────────────────

class Session:
    def __init__(self):
        self.board = None
        self.winner = None

    def new_game(self, players_payload, seed=None):
        players = []
        for idx, p in enumerate(players_payload):
            players.append(Player(
                player_id=idx,
                name=str(p.get("name", f"Player {idx+1}")),
                is_ai=bool(p.get("is_ai", False)),
                ai_difficulty=p.get("difficulty"),
            ))
        self.board = generate_board(seed=seed, players=players)
        self.winner = None
        return self.state()

    def _ai_decision(self, player):
        if player.ai_difficulty == "hard" and PPO_MODEL is not None:
            return ppo_decision(self.board, player, PPO_MODEL)
        return expectimax_decision(self.board, player)

    def buy(self, head, tail):
        if self.board is None or self.winner:
            return False, "No active game."
        p = self.board.active_player
        if p.is_ai:
            return False, "Not a human turn."
        ok, msg = can_place_snake(self.board, p, head, tail)
        if not ok:
            return False, msg
        return buy_snake(self.board, p, head, tail)

    def take_turn(self):
        """Roll for the active player. AI auto-decides its shop move; a
        human's snakes are already bought via buy()."""
        if self.board is None or self.winner:
            return {"logs": [], "winner": self.winner, "move": None}
        p = self.board.active_player
        shop = self._ai_decision(p) if p.is_ai else None
        result = do_turn(self.board, shop_decision=shop)
        if result["winner"]:
            self.winner = result["winner"].name
        return {"logs": result["logs"], "winner": self.winner,
                "move": result.get("move")}

    def shop_options(self):
        """Valid + affordable placements for the active human, computed by
        the engine (single source of truth). Returns heads and, per head,
        the list of [tail, cost] options."""
        b = self.board
        if b is None or self.winner:
            return {"heads": [], "tails": {}}
        p = b.active_player
        if p.is_ai or p.position < 1 or not p.can_buy_snake:
            return {"heads": [], "tails": {}}
        heads, tails = [], {}
        for head in range(20, MAX_SNAKE_HEAD + 1):
            opts = []
            for tail in range(1, head):
                cost = calculate_snake_cost(p, head, tail)
                if p.points < cost:
                    continue
                ok, _ = can_place_snake(b, p, head, tail)
                if ok:
                    opts.append([tail, cost])
            if opts:
                heads.append(head)
                tails[head] = opts
        return {"heads": heads, "tails": tails}

    def state(self):
        b = self.board
        if b is None:
            return {"started": False}
        a = b.active_player
        return {
            "started": True,
            "tiles": {str(k): v for k, v in b.tiles.items()},
            "ladders": [[l.bottom, l.top] for l in b.ladders],
            "snakes": [[s.head, s.tail, s.owner_id] for s in b.snakes],
            "bombs": list(b.bombs),
            "players": [{
                "id": p.player_id, "name": p.name, "position": p.position,
                "points": p.points, "is_ai": p.is_ai, "difficulty": p.ai_difficulty,
                "snake_count": p.snake_count, "can_buy_snake": p.can_buy_snake,
                "bankrupt_count": p.bankrupt_count,
            } for p in b.players],
            "current_turn": b.current_turn,
            "turn_number": b.turn_number,
            "winner": self.winner,
        }


SESSION = Session()


# ── REST API ──────────────────────────────────────────────────────────────────

@app.route("/api/generate-board", methods=["POST"])
def api_generate_board():
    """Stateless board generator — used by the lobby preview only."""
    body = request.json or {}
    seed = body.get("seed")
    seed = random.randint(0, 999999) if seed is None else int(seed)
    players = [Player(player_id=i, name=p.get("name", f"P{i+1}"))
               for i, p in enumerate(body.get("players", []))] or None
    try:
        board = generate_board(seed=seed, players=players)
        return jsonify({
            "seed": seed,
            "ladders": [[l.bottom, l.top] for l in board.ladders],
            "snakes": [[s.head, s.tail, s.owner_id] for s in board.snakes],
            "bombs": list(board.bombs),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/new", methods=["POST"])
def api_new():
    body = request.json or {}
    seed = body.get("seed")
    seed = None if seed in (None, "") else int(seed)
    try:
        return jsonify(SESSION.new_game(body.get("players", []), seed=seed))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/state", methods=["GET"])
def api_state():
    return jsonify(SESSION.state())


@app.route("/api/shop-options", methods=["GET"])
def api_shop_options():
    return jsonify(SESSION.shop_options())


@app.route("/api/buy", methods=["POST"])
def api_buy():
    body = request.json or {}
    ok, msg = SESSION.buy(int(body.get("head", 0)), int(body.get("tail", 0)))
    return jsonify({"ok": ok, "message": msg, "state": SESSION.state()})


@app.route("/api/turn", methods=["POST"])
def api_turn():
    try:
        res = SESSION.take_turn()
        return jsonify({"logs": res["logs"], "winner": res["winner"],
                        "move": res["move"], "state": SESSION.state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/quit", methods=["POST"])
def api_quit():
    SESSION.board = None
    SESSION.winner = None
    return jsonify({"ok": True})


# ── Static file server ──────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory("web", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("web", path)


def run_server_flask(port=5000):
    print(f"[Web] Snakes & Lenders on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
