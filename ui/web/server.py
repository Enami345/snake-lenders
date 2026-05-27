"""
Snakes & Lenders — Web UI (stdlib, no extra dependencies).

A tiny HTTP server exposing a JSON game API plus a static frontend.
Intro screen (players / humans / difficulty) → board.

Run:
    python -m ui.web.server          # then open http://localhost:8000
    python main.py --web             # same thing
"""

import os
import json
import traceback
import http.server

from game.board import generate_board
from game.engine import do_turn, buy_snake, can_place_snake, calculate_snake_cost
from ai.expectimax import expectimax_decision

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8000


# ── In-memory game session ────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.board = None
        self.ppo_model = None
        self.winner = None

    def new_game(self, n_players, n_humans, n_hard, seed=None):
        from main import build_players, load_ppo_if_needed
        players = build_players(n_players, n_humans, n_hard)
        self.ppo_model = load_ppo_if_needed(players)
        self.board = generate_board(seed=seed, players=players)
        self.winner = None
        return self.state()

    def _ai_decision(self, player):
        if player.ai_difficulty == "hard" and self.ppo_model is not None:
            from ai.ppo_agent import ppo_decision
            return ppo_decision(self.board, player, self.ppo_model)
        return expectimax_decision(self.board, player)

    def buy(self, head, tail):
        """Human places a snake before rolling. Returns (ok, message)."""
        if self.board is None or self.winner:
            return False, "No active game."
        p = self.board.active_player
        if p.is_ai:
            return False, "Not your turn."
        ok, msg = can_place_snake(self.board, p, head, tail)
        if not ok:
            return False, msg
        return buy_snake(self.board, p, head, tail)

    def turn(self):
        """Advance one turn: roll for the active player (AI auto-decides
        its shop move; a human's snakes are bought beforehand via buy())."""
        if self.board is None or self.winner:
            return {"logs": [], "winner": self.winner}
        p = self.board.active_player
        shop = self._ai_decision(p) if p.is_ai else None
        result = do_turn(self.board, shop_decision=shop)
        if result["winner"]:
            self.winner = result["winner"].name
        return {"logs": result["logs"], "winner": self.winner,
                "move": result.get("move")}

    def state(self):
        b = self.board
        if b is None:
            return {"started": False}
        active = b.active_player
        return {
            "started": True,
            "tiles": {str(k): v for k, v in b.tiles.items()},
            "ladders": [[l.bottom, l.top] for l in b.ladders],
            "snakes": [[s.head, s.tail, s.owner_id] for s in b.snakes],
            "bombs": list(b.bombs),
            "players": [{
                "id": p.player_id, "name": p.name, "position": p.position,
                "points": p.points, "snakes": p.snake_count,
                "is_ai": p.is_ai, "difficulty": p.ai_difficulty,
            } for p in b.players],
            "current": active.player_id,
            "current_name": active.name,
            "current_is_ai": active.is_ai,
            "can_shop": (not active.is_ai and active.position > 0
                         and active.can_buy_snake),
            "turn": b.turn_number,
            "winner": self.winner,
        }


SESSION = Session()


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj))

    def _file(self, name, ctype):
        path = os.path.join(WEB_DIR, name)
        if not os.path.exists(path):
            self._send(404, "not found", "text/plain")
            return
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _body_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._file("index.html", "text/html")
        elif self.path == "/app.js":
            self._file("app.js", "application/javascript")
        elif self.path == "/style.css":
            self._file("style.css", "text/css")
        elif self.path == "/api/state":
            self._json(200, SESSION.state())
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        body = self._body_json()
        try:
            if self.path == "/api/new":
                n = max(2, min(4, int(body.get("players", 2))))
                h = max(0, min(n, int(body.get("humans", 1))))
                hard = max(0, min(n - h, int(body.get("hard_ais", 0))))
                self._json(200, SESSION.new_game(n, h, hard))
            elif self.path == "/api/buy":
                ok, msg = SESSION.buy(int(body.get("head", 0)),
                                      int(body.get("tail", 0)))
                self._json(200, {"ok": ok, "message": msg,
                                 "state": SESSION.state()})
            elif self.path == "/api/quit":
                SESSION.board = None
                SESSION.winner = None
                self._json(200, {"ok": True})
            elif self.path == "/api/turn":
                res = SESSION.turn()
                self._json(200, {"logs": res["logs"], "winner": res["winner"],
                                 "move": res.get("move"),
                                 "state": SESSION.state()})
            elif self.path == "/api/cost":
                b = SESSION.board
                p = b.active_player if b else None
                cost = (calculate_snake_cost(p, int(body.get("head", 0)),
                                             int(body.get("tail", 0)))
                        if p else None)
                self._json(200, {"cost": cost})
            else:
                self._send(404, "not found", "text/plain")
        except Exception as e:
            # Always respond (otherwise the browser hangs forever) AND log
            # the full traceback to the server terminal so we can see it.
            tb = traceback.format_exc()
            print("[Web] ERROR handling", self.path, "\n", tb)
            self._json(500, {"error": str(e), "trace": tb})


def run_server(port=PORT):
    # ThreadingHTTPServer: browsers open multiple/keep-alive connections; a
    # single-threaded server would deadlock (one request blocks the rest).
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("", port), Handler) as httpd:
        print(f"[Web] Snakes & Lenders at http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Web] stopped.")


if __name__ == "__main__":
    run_server()
