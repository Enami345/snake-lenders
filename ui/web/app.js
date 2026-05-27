// Snakes & Lenders — minimal frontend. Teammate will redesign.
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  return r.json();
};

let state = null;

// ── Setup ──
$("players").addEventListener("change", syncHumansMax);
function syncHumansMax() {
  const n = +$("players").value;
  [...$("humans").options].forEach(o => { o.disabled = +o.value > n; });
  if (+$("humans").value > n) $("humans").value = n;
}
syncHumansMax();

function enterGame() {
  $("setup").classList.add("hidden");
  $("game").classList.remove("hidden");
  $("rollBtn").disabled = false;
  $("winner").classList.add("hidden");
}

$("startBtn").addEventListener("click", async () => {
  state = await api("/api/new", {
    players: +$("players").value,
    humans: +$("humans").value,
    difficulty: $("difficulty").value,
  });
  $("log").textContent = "";
  enterGame();
  render();
});

$("newBtn").addEventListener("click", () => {
  $("game").classList.add("hidden");
  $("setup").classList.remove("hidden");
  $("log").textContent = "";
  $("winner").classList.add("hidden");
});

// Resume an in-progress game on page refresh (server keeps the session).
async function init() {
  const s = await api("/api/state");
  if (s && s.started) {
    state = s;
    enterGame();
    render();
    if (s.winner) showWinner(s.winner);
  }
}
init();

// ── Turn / shop ──
$("rollBtn").addEventListener("click", async () => {
  const res = await api("/api/turn", {});
  state = res.state;
  appendLog(res.logs);
  render();
  if (res.winner) showWinner(res.winner);
});

$("buyBtn").addEventListener("click", async () => {
  const res = await api("/api/buy", {
    head: +$("head").value, tail: +$("tail").value,
  });
  $("shopMsg").textContent = res.message || "";
  state = res.state;
  render();
});

function showWinner(name) {
  const w = $("winner");
  w.textContent = "🏆 " + name + " wins!";
  w.classList.remove("hidden");
  $("rollBtn").disabled = true;
}

function appendLog(lines) {
  if (!lines || !lines.length) return;
  $("log").textContent += lines.join("\n") + "\n";
  $("log").scrollTop = $("log").scrollHeight;
}

// ── Render ──
// tile (1-100) -> grid row/col (boustrophedon, bottom row = tile 1-10)
function tileToCell(tile) {
  const t = tile - 1;
  let row = Math.floor(t / 10);
  let col = t % 10;
  if (row % 2 === 1) col = 9 - col;
  return { gridRow: 10 - row, gridCol: col + 1 };
}

function render() {
  if (!state || !state.started) return;

  // turn info
  const cur = state.players[state.current];
  $("turnInfo").textContent =
    `Turn ${state.turn} — ${state.current_name}` +
    (state.current_is_ai ? " 🤖 (press Next)" : " 👤 (your turn)");

  // shop visibility
  $("shop").classList.toggle("hidden", !state.can_shop);

  // players panel
  $("players").innerHTML = state.players.map(p =>
    `<div class="pcard ${p.id === state.current ? "active" : ""}">
       <b class="p${p.id}">${p.name}</b>
       ${p.is_ai ? "[" + p.difficulty + " AI]" : "[human]"}<br>
       tile ${p.position} · ${p.points} pts · snakes ${p.snakes}/3
     </div>`).join("");

  // board
  const ladTop = new Map(state.ladders.map(([b, t]) => [t, b]));
  const ladBot = new Map(state.ladders.map(([b, t]) => [b, t]));
  const snHead = new Map(state.snakes.map(([h, t]) => [h, t]));
  const snTail = new Map(state.snakes.map(([h, t]) => [t, h]));
  const bombs = new Set(state.bombs);

  const board = $("board");
  board.innerHTML = "";
  for (let tile = 1; tile <= 100; tile++) {
    const { gridRow, gridCol } = tileToCell(tile);
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.style.gridRow = gridRow;
    cell.style.gridColumn = gridCol;
    if (tile === 100) cell.classList.add("goal");
    else if (snHead.has(tile)) cell.classList.add("snake");
    else if (snTail.has(tile)) cell.classList.add("snaketail");
    else if (ladBot.has(tile)) cell.classList.add("ladder");
    else if (ladTop.has(tile)) cell.classList.add("ladtop");
    else if (bombs.has(tile)) cell.classList.add("bomb");

    let label = `<span class="num">${tile}</span>`;
    if (snHead.has(tile)) label += "🐍" + snHead.get(tile);
    else if (ladBot.has(tile)) label += "🪜" + ladBot.get(tile);
    else if (bombs.has(tile)) label += "💣";
    else if (tile === 100) label += "🏆";
    cell.innerHTML = label;
    board.appendChild(cell);
  }

  // tokens (stack offset per player on a tile)
  const perTile = {};
  state.players.forEach(p => {
    if (p.position < 1) return;
    const { gridRow, gridCol } = tileToCell(p.position);
    const cell = [...board.children].find(c =>
      +c.style.gridRow === gridRow && +c.style.gridColumn === gridCol);
    if (!cell) return;
    const idx = (perTile[p.position] = (perTile[p.position] || 0)) ;
    perTile[p.position]++;
    const tok = document.createElement("div");
    tok.className = `token p${p.id} t${idx}`;
    tok.textContent = p.name[0];
    cell.appendChild(tok);
  });
}
