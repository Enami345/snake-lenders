"""
Snakes & Lenders — Flask Backend Server
Hosts the stateless AI decision REST API and serves the game's premium static assets.

Run:
    python server.py
"""

import os
import sys
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.models import Player, BoardState, Snake, Ladder
from game.board import generate_board
from game.engine import calculate_snake_cost
from ai.expectimax import expectimax_decision
from ai.ppo_agent import load_ppo_model, ppo_decision

app = Flask(__name__, static_folder="web")
CORS(app)

# Load PPO Model once during startup
PPO_MODEL = None
try:
    PPO_MODEL = load_ppo_model()
    print("[PPO] PPO Model successfully pre-loaded for Hard AI.")
except Exception as e:
    print(f"[PPO] Warning during PPO Model preload: {e}")
    print("[PPO] Hard AI will fall back to Expectimax decision tree on demand.")


def reconstruct_board_and_player(data, active_player_id):
    """
    Reconstructs Python game structures (BoardState, Player, Snake, Ladder)
    from frontend JSON payload.
    """
    # 1. Build list of player objects
    players = []
    active_player = None

    for p_data in data["players"]:
        p = Player(
            player_id=int(p_data["id"]),
            name=str(p_data["name"]),
            position=int(p_data["position"]),
            points=int(p_data["points"]),
            is_ai=bool(p_data["is_ai"]),
            ai_difficulty=p_data.get("difficulty"),
            bankrupt_count=int(p_data.get("bankrupt_count", 0))
        )
        # Reconstruct player owned snakes
        for s_data in data["snakes"]:
            # s_data is: [head, tail, owner_id]
            if s_data[2] == p.player_id:
                p.snakes_owned.append(Snake(head=int(s_data[0]), tail=int(s_data[1]), owner_id=int(s_data[2])))
        players.append(p)
        if p.player_id == active_player_id:
            active_player = p

    # 2. Build global snakes and ladders list
    snakes = [Snake(head=int(s[0]), tail=int(s[1]), owner_id=int(s[2])) for s in data["snakes"]]
    ladders = [Ladder(bottom=int(l[0]), top=int(l[1])) for l in data["ladders"]]
    bombs = [int(b) for b in data["bombs"]]

    # 3. Create BoardState
    board = BoardState(
        tiles={int(k): int(v) for k, v in data["tiles"].items()},
        ladders=ladders,
        snakes=snakes,
        bombs=bombs,
        players=players,
        current_turn=int(data.get("current_turn", 0)),
        turn_number=int(data.get("turn_number", 1))
    )

    return board, active_player


# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.route("/api/generate-board", methods=["POST"])
def api_generate_board():
    """
    Generates a BFS-validated board state given a seed and list of players.
    """
    body = request.json or {}
    seed = body.get("seed")
    if seed is None:
        seed = random.randint(0, 999999)
    else:
        seed = int(seed)

    player_list_data = body.get("players", [])
    players = []
    for idx, p_data in enumerate(player_list_data):
        players.append(Player(
            player_id=idx,
            name=str(p_data.get("name", f"Player {idx+1}")),
            is_ai=bool(p_data.get("is_ai", False)),
            ai_difficulty=p_data.get("difficulty")
        ))

    try:
        board = generate_board(seed=seed, players=players)
        
        # Serialize the generated board to JSON
        state = {
            "seed": seed,
            "tiles": {str(k): v for k, v in board.tiles.items()},
            "ladders": [[l.bottom, l.top] for l in board.ladders],
            "snakes": [[s.head, s.tail, s.owner_id] for s in board.snakes],
            "bombs": list(board.bombs),
            "players": [{
                "id": p.player_id,
                "name": p.name,
                "position": p.position,
                "points": p.points,
                "is_ai": p.is_ai,
                "difficulty": p.ai_difficulty,
                "bankrupt_count": p.bankrupt_count
            } for p in board.players],
            "current_turn": board.current_turn,
            "turn_number": board.turn_number
        }
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/ai-move", methods=["POST"])
def api_ai_move():
    """
    Stateless decider for AI moves.
    Accepts current board state and active player ID, returns the AI decision.
    """
    body = request.json or {}
    board_data = body.get("board")
    active_player_id = body.get("activePlayerId")

    if not board_data or active_player_id is None:
        return jsonify({"error": "Missing board or activePlayerId parameter."}), 400

    try:
        board, active_player = reconstruct_board_and_player(board_data, int(active_player_id))

        if not active_player or not active_player.is_ai:
            return jsonify({"error": "Active player is not an AI player."}), 400

        # AI Decision choosing strategy
        decision = None
        if active_player.ai_difficulty == "hard":
            if PPO_MODEL is not None:
                decision = ppo_decision(board, active_player, PPO_MODEL)
            else:
                print("[PPO API] Warning: PPO Model not available. Falling back to Expectimax.")
                decision = expectimax_decision(board, active_player)
        else:
            decision = expectimax_decision(board, active_player)

        # Return {"head": H, "tail": T} or null
        return jsonify(decision)
    except Exception as e:
        print(f"[AI Move API Error] {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/cost", methods=["POST"])
def api_cost():
    """
    Utility endpoint to calculate the cost of a snake for the UI.
    """
    body = request.json or {}
    head = int(body.get("head", 0))
    tail = int(body.get("tail", 0))
    snake_count = int(body.get("snake_count", 0))
    points = int(body.get("points", 0))

    if head <= tail or head < 1 or tail < 1:
        return jsonify({"cost": 0, "affordable": False})

    # Create helper mock Player
    dummy_player = Player(player_id=99, name="Dummy", points=points)
    dummy_player.snakes_owned = [None] * snake_count

    cost = calculate_snake_cost(dummy_player, head, tail)
    return jsonify({
        "cost": cost,
        "affordable": points >= cost
    })


# ── Static File Server ────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory("web", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    # Standard static files in 'web/' directory
    return send_from_directory("web", path)


def run_server_flask(port=5000):
    print(f"[Web] Launching Snakes & Lenders premium Flask web app on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    # Start on port 5000 as required
    app.run(host="0.0.0.0", port=5000, debug=True)
