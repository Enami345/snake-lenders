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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const STEP_MS = 170;   // per-tile walk
const JUMP_MS = 480;   // ladder climb / snake slide
let busy = false;

// ── Screen management (title → setup → game; loading is an overlay) ──
const SCREENS = ["title", "setup", "game"];
function show(id) {
  SCREENS.forEach((s) => $(s).classList.toggle("hidden", s !== id));
}

$("playBtn").addEventListener("click", () => show("setup"));
$("backBtn").addEventListener("click", () => show("title"));

// ── Setup dropdown bounding ──
$("players").addEventListener("change", syncSetup);
$("humans").addEventListener("change", syncSetup);
function syncSetup() {
  const n = +$("players").value;
  [...$("humans").options].forEach((o) => { o.disabled = +o.value > n; });
  if (+$("humans").value > n) $("humans").value = n;
  const nAi = n - +$("humans").value;          // Hard AIs can't exceed AI count
  [...$("hardais").options].forEach((o) => { o.disabled = +o.value > nAi; });
  if (+$("hardais").value > nAi) $("hardais").value = nAi;
}
syncSetup();

// ── Loading overlay with a staged progress bar ──
async function runLoading(steps) {
  $("loading").classList.remove("hidden");
  $("barFill").style.width = "0%";
  for (let i = 0; i < steps.length; i++) {
    $("loadMsg").textContent = steps[i];
    $("barFill").style.width = Math.round(((i + 1) / steps.length) * 100) + "%";
    await sleep(380);
  }
}

$("startBtn").addEventListener("click", async () => {
  const payload = {
    players: +$("players").value,
    humans: +$("humans").value,
    hard_ais: +$("hardais").value,
  };
  // Run the staged loading animation and the actual request together.
  const [, st] = await Promise.all([
    runLoading(["Generating board…", "Loading AI model…",
                "Placing snakes & ladders…", "Ready!"]),
    api("/api/new", payload),
  ]);
  state = st;
  $("loading").classList.add("hidden");
  $("log").textContent = "";
  show("game");
  $("rollBtn").disabled = false;
  $("winner").classList.add("hidden");
  render();
});

$("newBtn").addEventListener("click", async () => {
  await api("/api/quit", {});      // fully reset the server session
  state = null;
  $("log").textContent = "";
  $("winner").classList.add("hidden");
  show("title");
});

// On load: resume an in-progress game, else show the title page.
async function init() {
  const s = await api("/api/state");
  if (s && s.started) {
    state = s;
    show("game");
    $("rollBtn").disabled = !!s.winner;
    render();
    if (s.winner) showWinner(s.winner);
  } else {
    show("title");
  }
}
init();

$("rollBtn").addEventListener("click", async () => {
  if (busy) return;
  busy = true; $("rollBtn").disabled = true;
  try {
    const res = await api("/api/turn", {});
    if (res.error) { appendLog(["[server error] " + res.error]); console.error(res.trace || res.error); return; }
    state = res.state;
    if (res.move) {
      appendLog([`🎲 ${res.move.name} rolled ${res.move.roll}`]);
      await animateMove(res.move);
    }
    appendLog(res.logs);
    render();
    if (res.winner) showWinner(res.winner);
  } finally {
    busy = false;
    if (!(state && state.winner)) $("rollBtn").disabled = false;
  }
});

// Walk the mover's token one tile per dice step, then animate any
// ladder/snake/bankruptcy jump. Mutates state then renders each frame.
async function animateMove(m) {
  const pl = state.players.find((p) => p.id === m.player_id);
  if (!pl) return;
  const final = pl.position;            // server's authoritative end tile
  for (let t = m.from; t <= m.landing; t++) {
    pl.position = t; render(); await sleep(STEP_MS);
  }
  if (final !== m.landing) {             // ladder / snake / reset
    pl.position = final; render(); await sleep(JUMP_MS);
  }
  pl.position = final;
}

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
