/**
 * Snakes & Lenders — High-Fidelity Front-End Game Engine & Visual Renderer
 * 
 * Inspired by the juiced aesthetics, screen shakes, and coin bursts of Monopoly Go.
 * Built entirely with Vanilla JS, HTML5 Canvas, and Procedural Web Audio API.
 */

"use strict";

const $ = (id) => document.getElementById(id);

// ── GLOBAL STATE ──
let gameSession = {
  started: false,
  tiles: {},
  ladders: [],
  snakes: [],
  bombs: [],
  players: [],
  current_turn: 0,
  turn_number: 1,
  winner: null,
  seed: null,
  
  // Client-side statistics
  stats: {},
  // Turn Timer
  timerInterval: null,
  timerSecondsLeft: 0,
  // Match history for replay
  actionHistory: []
};

let settings = {
  volSFX: 80,
  volBGM: 60,
  bgmEnabled: true,
  speed: "normal", // slow, normal, fast, instant
  durDice: 12, // 1.2s default
  shakeEnabled: true,
  particlesEnabled: true,
  floatNumsEnabled: true,
  aiEmotesEnabled: true,
  shopHintsEnabled: true,
  turnTimerEnabled: true,
  turnTimerDur: 30
};

// Animation and rendering systems
let animationQueue = [];
let isAnimating = false;
let canvas, ctx;
let particles = [];
let floatingNumbers = [];
let comboPopups = [];
let boardShakeOffset = { x: 0, y: 0 };
let currentHoveredTile = null;

// Player names and stable colors
const NEON_COLORS = {
  0: "#ff4a5a", // Neon Red
  1: "#3b82f6", // Neon Blue
  2: "#10b981", // Neon Green
  3: "#f59e0b"  // Neon Orange
};

const EMOTE_EMOJIS = {
  laughing: "😂",
  crying: "😭",
  shocked: "😱",
  taunting: "😏",
  celebrating: "🎉",
  sad: "😢"
};

// Load settings from localStorage
function loadSettings() {
  const saved = localStorage.getItem("snakes_lenders_settings");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      settings = { ...settings, ...parsed };
    } catch (e) {
      console.error("Error parsing settings:", e);
    }
  }
  // Sync UI sliders
  $("slider-vol-sfx").value = settings.volSFX;
  $("slider-vol-bgm").value = settings.volBGM;
  $("toggle-bgm").checked = settings.bgmEnabled;
  $("select-anim-speed").value = settings.speed;
  $("slider-dur-dice").value = settings.durDice;
  $("toggle-shake").checked = settings.shakeEnabled;
  $("toggle-particles").checked = settings.particlesEnabled;
  $("toggle-floating-nums").checked = settings.floatNumsEnabled;
  $("toggle-ai-emotes").checked = settings.aiEmotesEnabled;
  $("toggle-shop-hints").checked = settings.shopHintsEnabled;
  $("toggle-turn-timer").checked = settings.turnTimerEnabled;
  $("slider-dur-timer").value = settings.turnTimerDur;

  updateSettingsLabels();
}

function updateSettingsLabels() {
  $("label-vol-sfx").textContent = `${settings.volSFX}%`;
  $("label-vol-bgm").textContent = `${settings.volBGM}%`;
  $("label-dur-dice").textContent = `${(settings.durDice / 10).toFixed(1)}s`;
  $("label-dur-timer").textContent = `${settings.turnTimerDur}s`;
}

function saveSettings() {
  localStorage.setItem("snakes_lenders_settings", JSON.stringify(settings));
}

// ── PROCEDURAL WEB AUDIO SYNTH ENGINE ──
let audioCtx = null;
let bgmSourceNode = null;
let bgmGainNode = null;

function initAudio() {
  if (audioCtx) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioContextClass();
  
  if (settings.bgmEnabled) {
    startBGM();
  }
}

// Procedural sound effects using Web Audio API nodes
function playSFX(type, param = null) {
  if (!audioCtx) return;
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }

  const speedMultiplier = getSpeedMultiplier();
  const vol = (settings.volSFX / 100) * 0.15; // master gain reduction

  try {
    const t = audioCtx.currentTime;
    
    if (type === "click") {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.frequency.setValueAtTime(800, t);
      osc.frequency.exponentialRampToValueAtTime(100, t + 0.05);
      gain.gain.setValueAtTime(vol * 0.8, t);
      gain.gain.exponentialRampToValueAtTime(0.01, t + 0.05);
      osc.start(t);
      osc.stop(t + 0.06);
    }
    
    else if (type === "hop") {
      // Bouncy pop synth sound. Distinct pitch based on player ID
      const playerId = param || 0;
      const baseFreq = 220 + (playerId * 80);
      const osc = audioCtx.createOscillator();
      const osc2 = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      
      osc.connect(gain);
      osc2.connect(gain);
      gain.connect(audioCtx.destination);
      
      osc.type = "sine";
      osc2.type = "triangle";
      osc2.frequency.setValueAtTime(baseFreq * 1.5, t);
      
      osc.frequency.setValueAtTime(baseFreq, t);
      osc.frequency.exponentialRampToValueAtTime(baseFreq * 2.2, t + 0.12);
      
      gain.gain.setValueAtTime(vol * 1.2, t);
      gain.gain.exponentialRampToValueAtTime(0.01, t + 0.12);
      
      osc.start(t);
      osc2.start(t);
      osc.stop(t + 0.13);
      osc2.stop(t + 0.13);
    }
    
    else if (type === "land") {
      // Dust landing thump
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.type = "triangle";
      osc.frequency.setValueAtTime(100, t);
      osc.frequency.exponentialRampToValueAtTime(30, t + 0.08);
      gain.gain.setValueAtTime(vol * 1.5, t);
      gain.gain.exponentialRampToValueAtTime(0.01, t + 0.08);
      osc.start(t);
      osc.stop(t + 0.09);
    }
    
    else if (type === "snake_bite") {
      // Dramatic horror sweep with a white noise hiss
      const duration = 0.8;
      
      // Hiss
      const bufferSize = audioCtx.sampleRate * duration;
      const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = audioCtx.createBufferSource();
      noise.buffer = buffer;
      const noiseFilter = audioCtx.createBiquadFilter();
      noiseFilter.type = "highpass";
      noiseFilter.frequency.setValueAtTime(1500, t);
      
      const noiseGain = audioCtx.createGain();
      noiseGain.gain.setValueAtTime(vol * 0.7, t);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, t + duration);
      
      noise.connect(noiseFilter);
      noiseFilter.connect(noiseGain);
      noiseGain.connect(audioCtx.destination);
      noise.start(t);

      // Low cartoon slide sound
      const osc = audioCtx.createOscillator();
      const oscGain = audioCtx.createGain();
      osc.connect(oscGain);
      oscGain.connect(audioCtx.destination);
      
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(180, t);
      osc.frequency.linearRampToValueAtTime(50, t + duration);
      
      oscGain.gain.setValueAtTime(vol * 0.9, t);
      oscGain.gain.exponentialRampToValueAtTime(0.01, t + duration);
      
      osc.start(t);
      osc.stop(t + duration + 0.05);
    }
    
    else if (type === "ladder") {
      // Ascending arpeggio
      const notes = [261.63, 329.63, 392.00, 523.25, 659.25, 783.99, 1046.50]; // C major
      const step = 0.07 / speedMultiplier;
      
      notes.forEach((freq, idx) => {
        const timeOffset = idx * step;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, t + timeOffset);
        gain.gain.setValueAtTime(0, t + timeOffset);
        gain.gain.linearRampToValueAtTime(vol * 1.2, t + timeOffset + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.001, t + timeOffset + 0.15);
        osc.start(t + timeOffset);
        osc.stop(t + timeOffset + 0.2);
      });
    }
    
    else if (type === "bomb") {
      // Subwoofer rumbling cinematic boom
      const duration = 1.5;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(90, t);
      osc.frequency.linearRampToValueAtTime(20, t + duration);
      
      const filter = audioCtx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(120, t);
      filter.frequency.linearRampToValueAtTime(30, t + duration);
      
      osc.disconnect(gain);
      osc.connect(filter);
      filter.connect(gain);
      
      gain.gain.setValueAtTime(vol * 2.2, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
      osc.start(t);
      osc.stop(t + duration + 0.1);
      
      // Explosion crackle noise
      const bufferSize = audioCtx.sampleRate * 0.4;
      const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = audioCtx.createBufferSource();
      noise.buffer = buffer;
      const noiseGain = audioCtx.createGain();
      noiseGain.gain.setValueAtTime(vol * 0.8, t);
      noiseGain.gain.exponentialRampToValueAtTime(0.01, t + 0.35);
      noise.connect(noiseGain);
      noiseGain.connect(audioCtx.destination);
      noise.start(t);
    }
    
    else if (type === "bankrupt") {
      // Descending fail trombone sweep
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(330, t);
      osc.frequency.linearRampToValueAtTime(110, t + 0.7);
      gain.gain.setValueAtTime(vol * 1.5, t);
      gain.gain.exponentialRampToValueAtTime(0.01, t + 0.75);
      osc.start(t);
      osc.stop(t + 0.8);
    }
    
    else if (type === "steal") {
      // Coins stealing ding-ding cascade + cash register bell
      const step = 0.05 / speedMultiplier;
      for (let i = 0; i < 6; i++) {
        const timeOffset = i * step;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.type = "sine";
        osc.frequency.setValueAtTime(987.77 + (i * 120), t + timeOffset); // High pitch coin ticks
        gain.gain.setValueAtTime(0, t + timeOffset);
        gain.gain.linearRampToValueAtTime(vol * 0.6, t + timeOffset + 0.005);
        gain.gain.exponentialRampToValueAtTime(0.001, t + timeOffset + 0.08);
        osc.start(t + timeOffset);
        osc.stop(t + timeOffset + 0.1);
      }
      
      // Bell end
      const bellTime = t + (6 * step);
      const osc1 = audioCtx.createOscillator();
      const osc2 = audioCtx.createOscillator();
      const bellGain = audioCtx.createGain();
      osc1.connect(bellGain);
      osc2.connect(bellGain);
      bellGain.connect(audioCtx.destination);
      
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(1760, bellTime); // High A
      osc2.type = "triangle";
      osc2.frequency.setValueAtTime(2200, bellTime); // Minor third harmony
      
      bellGain.gain.setValueAtTime(0, bellTime);
      bellGain.gain.linearRampToValueAtTime(vol * 1.5, bellTime + 0.01);
      bellGain.gain.exponentialRampToValueAtTime(0.001, bellTime + 0.8);
      
      osc1.start(bellTime);
      osc2.start(bellTime);
      osc1.stop(bellTime + 0.85);
      osc2.stop(bellTime + 0.85);
    }
    
    else if (type === "dice") {
      // Dice shake rattle clicks
      const rollsCount = 12;
      const rollStep = 0.06;
      for (let i = 0; i < rollsCount; i++) {
        const timeOffset = i * rollStep;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.type = "triangle";
        osc.frequency.setValueAtTime(250 - (i * 10), t + timeOffset);
        gain.gain.setValueAtTime(0, t + timeOffset);
        gain.gain.linearRampToValueAtTime(vol * 0.7, t + timeOffset + 0.002);
        gain.gain.exponentialRampToValueAtTime(0.001, t + timeOffset + 0.04);
        osc.start(t + timeOffset);
        osc.stop(t + timeOffset + 0.05);
      }
      
      // Final roll landed check
      const isSix = param === 6;
      const endOffset = rollsCount * rollStep;
      
      // Thud
      const thud = audioCtx.createOscillator();
      const thudGain = audioCtx.createGain();
      thud.connect(thudGain);
      thudGain.connect(audioCtx.destination);
      thud.type = "sine";
      thud.frequency.setValueAtTime(70, t + endOffset);
      thud.frequency.exponentialRampToValueAtTime(30, t + endOffset + 0.1);
      thudGain.gain.setValueAtTime(0, t + endOffset);
      thudGain.gain.linearRampToValueAtTime(vol * 2.2, t + endOffset + 0.005);
      thudGain.gain.exponentialRampToValueAtTime(0.001, t + endOffset + 0.15);
      thud.start(t + endOffset);
      thud.stop(t + endOffset + 0.2);

      // If roll is 6, play special celestial chime!
      if (isSix) {
        const chimes = [523.25, 659.25, 783.99, 1046.50]; // CEG C arpeggio
        chimes.forEach((freq, index) => {
          const chimeTime = t + endOffset + (index * 0.04);
          const chime = audioCtx.createOscillator();
          const chimeGain = audioCtx.createGain();
          chime.connect(chimeGain);
          chimeGain.connect(audioCtx.destination);
          chime.type = "sine";
          chime.frequency.setValueAtTime(freq, chimeTime);
          chimeGain.gain.setValueAtTime(0, chimeTime);
          chimeGain.gain.linearRampToValueAtTime(vol * 1.5, chimeTime + 0.01);
          chimeGain.gain.exponentialRampToValueAtTime(0.001, chimeTime + 0.6);
          chime.start(chimeTime);
          chime.stop(chimeTime + 0.7);
        });
      }
    }
    
    else if (type === "shop_buy") {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.frequency.setValueAtTime(523.25, t); // C5
      osc.frequency.setValueAtTime(659.25, t + 0.08); // E5
      gain.gain.setValueAtTime(vol * 1.2, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
      osc.start(t);
      osc.stop(t + 0.26);
    }
    
    else if (type === "emote") {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      
      const mood = param || "laughing";
      if (mood === "taunting" || mood === "shocked") {
        osc.frequency.setValueAtTime(440, t);
        osc.frequency.exponentialRampToValueAtTime(880, t + 0.15);
      } else {
        osc.frequency.setValueAtTime(880, t);
        osc.frequency.exponentialRampToValueAtTime(440, t + 0.15);
      }
      
      gain.gain.setValueAtTime(vol * 0.8, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
      osc.start(t);
      osc.stop(t + 0.25);
    }
    
    else if (type === "win") {
      // Rich Fanfare Chords using simple additive oscillators
      const baseFreqs = [261.63, 329.63, 392.00, 523.25]; // C major chord
      baseFreqs.forEach((freq, idx) => {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.type = idx % 2 === 0 ? "triangle" : "sine";
        osc.frequency.setValueAtTime(freq, t);
        osc.frequency.linearRampToValueAtTime(freq * 1.5, t + 1.5); // Rise
        
        gain.gain.setValueAtTime(0, t);
        gain.gain.linearRampToValueAtTime(vol * 0.8, t + 0.1);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 1.8);
        osc.start(t);
        osc.stop(t + 2.0);
      });
    }
    
    else if (type === "thinking") {
      // Soft ticking clock
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.type = "sine";
      osc.frequency.setValueAtTime(3000, t);
      gain.gain.setValueAtTime(vol * 0.2, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.02);
      osc.start(t);
      osc.stop(t + 0.03);
    }
    
  } catch (e) {
    console.error("Web Audio SFX failed:", e);
  }
}

function startBGM() {
  if (!audioCtx) return;
  try {
    if (bgmSourceNode) {
      bgmSourceNode.stop();
      bgmSourceNode = null;
    }
    
    bgmGainNode = audioCtx.createGain();
    bgmGainNode.gain.setValueAtTime((settings.volBGM / 100) * 0.06, audioCtx.currentTime);
    bgmGainNode.connect(audioCtx.destination);
    
    // Create an oscillator arpeggiator synthesizer loop for loop BGM!
    const tempo = 120; // 120 bpm
    const beatSec = 60 / tempo;
    
    // Notes: Am - F - C - G progression
    const chords = [
      [220, 261.63, 329.63, 440], // Am
      [174.61, 220, 261.63, 349.23], // F
      [261.63, 329.63, 392.00, 523.25], // C
      [196.00, 246.94, 293.66, 392.00]  // G
    ];
    
    let chordIdx = 0;
    
    const playNextBeat = () => {
      if (!bgmGainNode || !settings.bgmEnabled) return;
      const t = audioCtx.currentTime;
      const chord = chords[chordIdx % chords.length];
      
      // Play 4 notes in a beautiful ascending/descending arpeggio per chord
      for (let i = 0; i < 4; i++) {
        const noteTime = t + (i * (beatSec / 2));
        const freq = chord[i];
        const osc = audioCtx.createOscillator();
        const chordGain = audioCtx.createGain();
        
        osc.connect(chordGain);
        chordGain.connect(bgmGainNode);
        
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, noteTime);
        
        chordGain.gain.setValueAtTime(0, noteTime);
        chordGain.gain.linearRampToValueAtTime((settings.volBGM / 100) * 0.08, noteTime + 0.02);
        chordGain.gain.exponentialRampToValueAtTime(0.001, noteTime + beatSec);
        
        osc.start(noteTime);
        osc.stop(noteTime + beatSec + 0.05);
      }
      
      chordIdx++;
      
      // Schedule the next chord in beatSec * 2 seconds (half notes)
      const nextCallMs = beatSec * 2 * 1000;
      bgmSourceNode = setTimeout(playNextBeat, nextCallMs);
    };
    
    playNextBeat();
    
  } catch (e) {
    console.error("Looping arpeggiator synthesizer failed:", e);
  }
}

function stopBGM() {
  if (bgmSourceNode) {
    clearTimeout(bgmSourceNode);
    bgmSourceNode = null;
  }
  if (bgmGainNode) {
    bgmGainNode.disconnect();
    bgmGainNode = null;
  }
}

function updateBGMVolume() {
  if (bgmGainNode) {
    bgmGainNode.gain.setValueAtTime((settings.volBGM / 100) * 0.06, audioCtx.currentTime);
  }
}

// ── GET ANIMATION SPEED MULTIPLIER ──
function getSpeedMultiplier() {
  if (settings.speed === "slow") return 0.5;
  if (settings.speed === "fast") return 2.0;
  if (settings.speed === "instant") return 100.0;
  return 1.0;
}


// ── REST API HELPERS ──
const api = async (path, body) => {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  if (!r.ok) {
    const err = await r.json();
    throw new Error(err.error || "Server request failed.");
  }
  return r.json();
};


// ── SCREEN ROUTER / INITIALIZATION ──
function showScreen(screenId) {
  document.querySelectorAll(".screen").forEach(s => s.classList.add("hidden"));
  $(`screen-${screenId}`).classList.remove("hidden");
  
  if (screenId === "menu") {
    $("main-menu-bg").classList.remove("hidden");
  } else {
    $("main-menu-bg").classList.add("hidden");
  }
}

// Populate background particles for premium Menu look
function spawnMenuParticles() {
  const wrap = $("bg-particles");
  if (!wrap) return;
  wrap.innerHTML = "";
  for (let i = 0; i < 25; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    p.style.left = `${Math.random() * 100}%`;
    p.style.width = `${Math.random() * 8 + 4}px`;
    p.style.height = p.style.width;
    p.style.animationDelay = `${Math.random() * 8}s`;
    p.style.animationDuration = `${Math.random() * 5 + 6}s`;
    wrap.appendChild(p);
  }
}

// Setup player slots on config select change
function renderLobbySlots() {
  const container = $("player-slots-container");
  const count = +$("select-players").value;
  container.innerHTML = "";

  for (let i = 0; i < count; i++) {
    const slot = document.createElement("div");
    slot.className = "player-slot active-slot";
    
    // Choose initial name, type and default color index
    const defaultName = i === 0 ? "You" : (i === 1 ? "Easy Bot" : (i === 2 ? "Hard Bot" : `Player ${i+1}`));
    const defaultType = i === 0 ? "human" : (i === 1 ? "easy" : "hard");
    
    slot.innerHTML = `
      <div class="slot-number">${i + 1}</div>
      <div class="slot-name">
        <input type="text" id="slot-name-${i}" value="${defaultName}" class="lobby-input w-100" />
      </div>
      <div class="slot-type">
        <select id="slot-type-${i}" class="lobby-select w-100">
          <option value="human" ${defaultType === 'human' ? 'selected' : ''}>👤 Human</option>
          <option value="easy" ${defaultType === 'easy' ? 'selected' : ''}>🤖 Easy AI</option>
          <option value="hard" ${defaultType === 'hard' ? 'selected' : ''}>🔥 Hard AI</option>
        </select>
      </div>
      <div class="slot-color">
        <div class="color-picker">
          <div class="color-dot selected" style="background-color: ${NEON_COLORS[i]}" data-color-idx="${i}"></div>
        </div>
      </div>
    `;
    container.appendChild(slot);
  }
}


// ── MAIN MENU ATTACHMENTS ──
$("btn-play-lobby").addEventListener("click", () => {
  initAudio();
  showScreen("lobby");
  renderLobbySlots();
  generateLobbyPreview();
});

$("btn-open-rules").addEventListener("click", () => {
  initAudio();
  $("modal-rules").classList.remove("hidden");
});

$("btn-close-rules").addEventListener("click", () => {
  $("modal-rules").classList.add("hidden");
});

$("btn-open-settings").addEventListener("click", () => {
  initAudio();
  $("modal-settings").classList.remove("hidden");
});

$("btn-open-ingame-settings").addEventListener("click", () => {
  $("modal-settings").classList.remove("hidden");
});

$("btn-close-settings").addEventListener("click", () => {
  $("modal-settings").classList.add("hidden");
  saveSettings();
});

// Settings Changes Syncing
$("slider-vol-sfx").addEventListener("input", (e) => {
  settings.volSFX = +e.target.value;
  updateSettingsLabels();
  playSFX("click");
});

$("slider-vol-bgm").addEventListener("input", (e) => {
  settings.volBGM = +e.target.value;
  updateSettingsLabels();
  updateBGMVolume();
});

$("toggle-bgm").addEventListener("change", (e) => {
  settings.bgmEnabled = e.target.checked;
  if (settings.bgmEnabled) {
    startBGM();
  } else {
    stopBGM();
  }
});

$("select-anim-speed").addEventListener("change", (e) => {
  settings.speed = e.target.value;
});

$("slider-dur-dice").addEventListener("input", (e) => {
  settings.durDice = +e.target.value;
  updateSettingsLabels();
});

$("toggle-shake").addEventListener("change", (e) => { settings.shakeEnabled = e.target.checked; });
$("toggle-particles").addEventListener("change", (e) => { settings.particlesEnabled = e.target.checked; });
$("toggle-floating-nums").addEventListener("change", (e) => { settings.floatNumsEnabled = e.target.checked; });
$("toggle-ai-emotes").addEventListener("change", (e) => { settings.aiEmotesEnabled = e.target.checked; });
$("toggle-shop-hints").addEventListener("change", (e) => { settings.shopHintsEnabled = e.target.checked; });
$("toggle-turn-timer").addEventListener("change", (e) => { settings.turnTimerEnabled = e.target.checked; });
$("slider-dur-timer").addEventListener("input", (e) => {
  settings.turnTimerDur = +e.target.value;
  updateSettingsLabels();
});


// ── LOBBY SETUP & SEEDING ──
$("select-players").addEventListener("change", renderLobbySlots);

$("btn-randomize-seed").addEventListener("click", () => {
  $("input-seed").value = Math.floor(Math.random() * 999999);
  playSFX("click");
  generateLobbyPreview();
});

$("btn-preview-board").addEventListener("click", () => {
  playSFX("click");
  generateLobbyPreview();
});

$("btn-lobby-back").addEventListener("click", () => {
  showScreen("menu");
});

async function generateLobbyPreview() {
  const count = +$("select-players").value;
  const seedInput = $("input-seed").value.trim();
  const seed = seedInput ? parseInt(seedInput) : Math.floor(Math.random() * 999999);
  if (!seedInput) $("input-seed").value = seed;

  const previewWrap = $("board-preview");
  previewWrap.innerHTML = "";

  const mockPlayers = [];
  for (let i = 0; i < count; i++) {
    mockPlayers.push({ name: `P${i}` });
  }

  try {
    const res = await api("/api/generate-board", { seed, players: mockPlayers });
    
    // Draw micro preview block cells
    const ladders = new Set(res.ladders.map(l => l[0]));
    const snakes = new Set(res.snakes.map(s => s[0]));
    const bombs = new Set(res.bombs);

    for (let tile = 1; tile <= 100; tile++) {
      const cell = document.createElement("div");
      cell.className = "preview-cell";
      if (ladders.has(tile)) cell.classList.add("l");
      else if (snakes.has(tile)) cell.classList.add("s");
      else if (bombs.has(tile)) cell.classList.add("b");
      previewWrap.appendChild(cell);
    }
  } catch (e) {
    console.error("Board generation preview failed:", e);
    previewWrap.innerHTML = `<div class="preview-placeholder">Generation Error</div>`;
  }
}

// ── PLAY BUTTON - LAUNCHES CLIENT ENGINE ──
$("btn-lobby-start").addEventListener("click", async () => {
  playSFX("shop_buy");
  
  const count = +$("select-players").value;
  const seed = parseInt($("input-seed").value.trim()) || Math.floor(Math.random() * 999999);
  
  const playersPayload = [];
  for (let i = 0; i < count; i++) {
    const inputName = $(`slot-name-${i}`).value.trim() || `Player ${i+1}`;
    const selectedType = $(`slot-type-${i}`).value;
    
    playersPayload.push({
      name: inputName,
      is_ai: selectedType !== "human",
      difficulty: selectedType === "human" ? null : selectedType
    });
  }

  try {
    const startState = await api("/api/generate-board", { seed, players: playersPayload });
    
    // Initialize Game Session locally
    gameSession = {
      started: true,
      tiles: startState.tiles,
      ladders: startState.ladders,
      snakes: startState.snakes,
      bombs: startState.bombs,
      players: startState.players,
      current_turn: startState.current_turn,
      turn_number: startState.turn_number,
      winner: null,
      seed: startState.seed,
      stats: {},
      actionHistory: []
    };

    // Reset statistics cards
    gameSession.players.forEach(p => {
      gameSession.stats[p.id] = {
        snakesPlaced: 0,
        bittenCount: 0,
        bankruptCount: 0,
        laddersClimbed: 0,
        bombsHit: 0,
        pointsEarned: 0,
        pointsStolen: 0,
        turnsTaken: 0
      };
    });

    // Enter screen
    showScreen("game");
    
    // Hook up canvas sizing
    canvas = $("game-canvas");
    ctx = canvas.getContext("2d");
    
    // Clear logs
    $("battle-log-lines").innerHTML = "";
    appendLog(`🎲 Game started! Seed = ${gameSession.seed}`);
    appendLog(`Seat shuffle turn order: ${gameSession.players.map(p => p.name).join(" ➔ ")}`);

    particles = [];
    floatingNumbers = [];
    comboPopups = [];
    isAnimating = false;
    animationQueue = [];
    
    renderGameScreen();
    startDrawLoop();
    
    // If first player is an AI, trigger AI loop
    checkAndExecuteAITurn();

  } catch (e) {
    alert(`Could not start game: ${e.message}`);
  }
});


// ── CANVAS DRAW ENGINE (HIGH FIDELITY) ──
let requestDrawId = null;

function startDrawLoop() {
  if (requestDrawId) cancelAnimationFrame(requestDrawId);
  
  const tick = () => {
    if (!gameSession.started) return;
    drawBoard();
    updateParticles();
    updateFloatingNumbers();
    requestDrawId = requestAnimationFrame(tick);
  };
  requestDrawId = requestAnimationFrame(tick);
}

// Boustrophedon board mappings: tile -> grid positions (row/col)
function tileToGridCoords(tile) {
  const t = tile - 1;
  let row = Math.floor(t / 10);
  let col = t % 10;
  if (row % 2 === 1) col = 9 - col; // serpentine zig-zag
  return { col, row };
}

// Center point of a tile index on canvas
function tileCenter(tile) {
  const { col, row } = tileToGridCoords(tile);
  const size = 750 / 10;
  const x = col * size + size / 2;
  const y = (9 - row) * size + size / 2;
  return { x, y };
}

function drawBoard() {
  const size = 750 / 10;
  ctx.save();
  
  // Camera Rumble screen shake
  ctx.translate(boardShakeOffset.x, boardShakeOffset.y);
  
  // Clear background
  ctx.fillStyle = "#0c0c16";
  ctx.fillRect(0, 0, 750, 750);

  // 1. Draw Checkerboard Squares with wood textured borders
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      const tileNum = (r % 2 === 0) ? (r * 10 + c + 1) : (r * 10 + (10 - c));
      
      const x = c * size;
      const y = (9 - r) * size;
      
      // Checker styling
      const isEven = (r + c) % 2 === 0;
      
      // Warm dark-wood-obsidian grain squares
      if (tileNum === 100) {
        ctx.fillStyle = "rgba(245, 196, 67, 0.25)"; // Glowing target gold
      } else if (isEven) {
        ctx.fillStyle = "#161628";
      } else {
        ctx.fillStyle = "#1e1e36";
      }
      ctx.fillRect(x, y, size, size);
      
      // Soft wood frame line
      ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, size, size);

      // Tile numbers
      ctx.font = "600 11px 'Outfit', sans-serif";
      ctx.fillStyle = tileNum === 100 ? "#f5c443" : "rgba(255, 255, 255, 0.15)";
      ctx.fillText(tileNum, x + 8, y + 18);
    }
  }

  // Spotlight Dim effect for shop placement previews
  if (currentHoveredTile && gameSession.started && settings.shopHintsEnabled) {
    ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
    ctx.fillRect(0, 0, 750, 750);
    
    // Draw highlighters spotlights
    const previewHead = +$("input-shop-head").value;
    const previewTail = +$("input-shop-tail").value;
    const activePlayer = gameSession.players[gameSession.current_turn];

    if (previewHead && previewTail) {
      // Validate placement to color preview spotlight
      const cost = calculateSnakeCostMock(activePlayer.snakes_owned.length, previewHead - previewTail);
      const isAffordable = activePlayer.points >= cost;
      
      const headCenter = tileCenter(previewHead);
      const tailCenter = tileCenter(previewTail);
      
      // Head Spotlight
      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath();
      ctx.arc(headCenter.x, headCenter.y, size * 0.7, 0, Math.PI * 2);
      ctx.fill();
      
      // Tail Spotlight
      ctx.beginPath();
      ctx.arc(tailCenter.x, tailCenter.y, size * 0.7, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Spotlight rings
      ctx.strokeStyle = isAffordable ? "#10b981" : "#ef4444";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(headCenter.x, headCenter.y, size * 0.6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(tailCenter.x, tailCenter.y, size * 0.6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // 2. Draw Fixed Bombs (burning sphere fuses)
  gameSession.bombs.forEach(bombTile => {
    const center = tileCenter(bombTile);
    const rad = size * 0.22;
    
    // Shadow
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.beginPath();
    ctx.arc(center.x + 3, center.y + 4, rad, 0, Math.PI * 2);
    ctx.fill();

    // 3D Spherical bomb gradient
    const grad = ctx.createRadialGradient(center.x - 3, center.y - 3, 2, center.x, center.y, rad);
    grad.addColorStop(0, "#4b5563");
    grad.addColorStop(0.8, "#1f2937");
    grad.addColorStop(1, "#111827");
    ctx.fillStyle = grad;
    
    ctx.beginPath();
    ctx.arc(center.x, center.y, rad, 0, Math.PI * 2);
    ctx.fill();

    // Burning fuse wire
    ctx.strokeStyle = "#ea580c";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(center.x + 8, center.y - 12, 10, 0, Math.PI, true);
    ctx.stroke();

    // Fuse Spark
    const sparkX = center.x + 18 + (Math.random() * 4 - 2);
    const sparkY = center.y - 12 + (Math.random() * 4 - 2);
    ctx.fillStyle = "#f5c443";
    ctx.beginPath();
    ctx.arc(sparkX, sparkY, 4 + Math.random() * 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // 3. Draw Ladders (detailed metallic outlined with shadows, spacing offsets, gaps, and highlights)
  const LADDER_THEMES = [
    { rail: "#f5c443", shine: "#ffe596", rungs: "#d4af37", name: "Gold" },
    { rail: "#06b6d4", shine: "#67e8f9", rungs: "#0891b2", name: "Cyan" },
    { rail: "#10b981", shine: "#34d399", rungs: "#059669", name: "Emerald" },
    { rail: "#ec4899", shine: "#f472b6", rungs: "#db2777", name: "Rose" },
    { rail: "#8b5cf6", shine: "#a78bfa", rungs: "#7c3aed", name: "Purple" },
    { rail: "#f97316", shine: "#fb923c", rungs: "#ea580c", name: "Orange" },
    { rail: "#cbd5e1", shine: "#f1f5f9", rungs: "#94a3b8", name: "Silver" }
  ];

  function getLineIntersection(A, B, C, D) {
    const r_xs = B.x - A.x;
    const r_ys = B.y - A.y;
    const s_xs = D.x - C.x;
    const s_ys = D.y - C.y;

    const denom = r_xs * s_ys - r_ys * s_xs;
    if (Math.abs(denom) < 0.0001) return null; // Parallel

    const u = ((C.x - A.x) * r_ys - (C.y - A.y) * r_xs) / denom;
    const t = ((C.x - A.x) * s_ys - (C.y - A.y) * s_xs) / denom;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      return {
        x: A.x + t * r_xs,
        y: A.y + t * r_ys,
        t: t // fraction along AB
      };
    }
    return null;
  }

  // Pre-calculate spacing offsets so nearby ladders are offset parallel to each other
  const ladderOffsets = new Array(gameSession.ladders.length).fill(0);
  for (let i = 0; i < gameSession.ladders.length; i++) {
    let closeCount = 0;
    for (let j = 0; j < gameSession.ladders.length; j++) {
      if (i === j) continue;
      const li = gameSession.ladders[i];
      const lj = gameSession.ladders[j];
      
      const closeBottom = Math.abs(li[0] - lj[0]) < 8;
      const closeTop = Math.abs(li[1] - lj[1]) < 8;
      
      if (closeBottom || closeTop) {
        closeCount++;
        if (j < i) {
          ladderOffsets[i] = ladderOffsets[j] === 0 ? -9 : -ladderOffsets[j];
        }
      }
    }
    if (closeCount > 0 && ladderOffsets[i] === 0) {
      ladderOffsets[i] = 9;
    }
  }

  gameSession.ladders.forEach(([b, t], idx) => {
    const originalBottom = tileCenter(b);
    const originalTop = tileCenter(t);
    
    let dx = originalTop.x - originalBottom.x;
    let dy = originalTop.y - originalBottom.y;
    let angle = Math.atan2(dy, dx);
    let length = Math.sqrt(dx * dx + dy * dy);
    
    // Normal vector perpendicular to ladder path for side-by-side offsets
    const nx = -Math.sin(angle);
    const ny = Math.cos(angle);
    
    const offsetVal = ladderOffsets[idx];
    const bottom = {
      x: originalBottom.x + nx * offsetVal,
      y: originalBottom.y + ny * offsetVal
    };
    const top = {
      x: originalTop.x + nx * offsetVal,
      y: originalTop.y + ny * offsetVal
    };
    
    const width = 16;
    const theme = LADDER_THEMES[idx % LADDER_THEMES.length];
    const isHovered = currentHoveredTile === b || currentHoveredTile === t;

    // Collect gaps where other ladders (in front, i.e., j > idx) intersect this ladder
    const gaps = [];
    const halfGapPx = 13; // 26px total gap at cross intersection
    for (let j = 0; j < gameSession.ladders.length; j++) {
      if (j <= idx) continue; // Only ladders in front of this one cause a gap
      
      const otherB = gameSession.ladders[j][0];
      const otherT = gameSession.ladders[j][1];
      const origOtherB = tileCenter(otherB);
      const origOtherT = tileCenter(otherT);
      
      const oDx = origOtherT.x - origOtherB.x;
      const oDy = origOtherT.y - origOtherB.y;
      const oAngle = Math.atan2(oDy, oDx);
      const oNx = -Math.sin(oAngle);
      const oNy = Math.cos(oAngle);
      
      const oOffset = ladderOffsets[j];
      const otherBOffset = { x: origOtherB.x + oNx * oOffset, y: origOtherB.y + oNy * oOffset };
      const otherTOffset = { x: origOtherT.x + oNx * oOffset, y: origOtherT.y + oNy * oOffset };
      
      const intersect = getLineIntersection(bottom, top, otherBOffset, otherTOffset);
      if (intersect) {
        const gapFrac = halfGapPx / length;
        gaps.push({
          start: Math.max(0, intersect.t - gapFrac),
          end: Math.min(1, intersect.t + gapFrac)
        });
      }
    }
    
    // Subtract gaps from [0, 1] interval
    gaps.sort((a, b) => a.start - b.start);
    let activeSegments = [{ start: 0, end: 1 }];
    gaps.forEach(gap => {
      let nextSegments = [];
      activeSegments.forEach(seg => {
        if (gap.end <= seg.start || gap.start >= seg.end) {
          nextSegments.push(seg);
        } else {
          if (gap.start > seg.start) {
            nextSegments.push({ start: seg.start, end: gap.start });
          }
          if (gap.end < seg.end) {
            nextSegments.push({ start: gap.end, end: seg.end });
          }
        }
      });
      activeSegments = nextSegments;
    });

    // Render each remaining visible segment of the ladder
    activeSegments.forEach(seg => {
      const startX = bottom.x + dx * seg.start;
      const startY = bottom.y + dy * seg.start;
      const endX = bottom.x + dx * seg.end;
      const endY = bottom.y + dy * seg.end;
      
      const segLength = (seg.end - seg.start) * length;
      
      ctx.save();
      ctx.translate(startX, startY);
      ctx.rotate(angle);

      // Setup drop shadows and outlines
      ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
      ctx.shadowBlur = 4;
      ctx.shadowOffsetX = 3;
      ctx.shadowOffsetY = 4;

      if (isHovered) {
        // High fidelity golden/neon pulsing outer glow trace highlights
        ctx.shadowColor = theme.rail;
        ctx.shadowBlur = 12 + Math.abs(Math.sin(Date.now() * 0.015) * 5);
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
      }

      // 1. Draw Rails (Thick black outline segment)
      ctx.strokeStyle = "rgba(0, 0, 0, 0.65)";
      ctx.lineWidth = 6;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(0, -width/2);
      ctx.lineTo(segLength, -width/2);
      ctx.moveTo(0, width/2);
      ctx.lineTo(segLength, width/2);
      ctx.stroke();

      // 2. Draw Rails (Neon themed color)
      ctx.strokeStyle = theme.rail;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(0, -width/2);
      ctx.lineTo(segLength, -width/2);
      ctx.moveTo(0, width/2);
      ctx.lineTo(segLength, width/2);
      ctx.stroke();

      // 3. Rails bright steel/neon shine center
      ctx.strokeStyle = theme.shine;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(0, -width/2);
      ctx.lineTo(segLength, -width/2);
      ctx.moveTo(0, width/2);
      ctx.lineTo(segLength, width/2);
      ctx.stroke();

      // 4. Rungs along this segment
      const totalRungs = Math.floor(length / 15);
      ctx.strokeStyle = theme.rungs;
      ctx.lineWidth = 3;
      
      for (let i = 1; i < totalRungs; i++) {
        const rxFrac = i / totalRungs;
        if (rxFrac >= seg.start && rxFrac <= seg.end) {
          const rx = (rxFrac - seg.start) * length;
          ctx.beginPath();
          ctx.moveTo(rx, -width/2 + 1.5);
          ctx.lineTo(rx, width/2 - 1.5);
          ctx.stroke();
        }
      }
      
      ctx.restore();
    });
  });

  // 4. Draw Snakes (Bezier gradient curves, segment swaying, blinking eyes)
  gameSession.snakes.forEach(([h, t, ownerId]) => {
    const head = tileCenter(h);
    const tail = tileCenter(t);
    
    const dx = tail.x - head.x;
    const dy = tail.y - head.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    
    // Bezier control point to curve the snake's slithery body
    // sway slightly based on global time
    const time = Date.now() * 0.003;
    const controlSway = Math.sin(time + h) * 22;
    const midX = (head.x + tail.x) / 2 + Math.cos(time) * 10;
    const midY = (head.y + tail.y) / 2 + controlSway;

    // Draw shadow
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(head.x + 3, head.y + 4);
    ctx.quadraticCurveTo(midX + 3, midY + 4, tail.x + 3, tail.y + 4);
    ctx.stroke();

    // Draw segment colored gradient body
    const grad = ctx.createLinearGradient(head.x, head.y, tail.x, tail.y);
    if (ownerId < 0) {
      grad.addColorStop(0, "#b91c1c"); // Board Snake Red
      grad.addColorStop(0.5, "#ea580c");
      grad.addColorStop(1, "#b91c1c");
    } else {
      // Color matches owner token
      const ownerColor = NEON_COLORS[ownerId] || "#e2e8f0";
      grad.addColorStop(0, ownerColor);
      grad.addColorStop(0.5, "#0f172a");
      grad.addColorStop(1, ownerColor);
    }
    
    ctx.strokeStyle = grad;
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(head.x, head.y);
    ctx.quadraticCurveTo(midX, midY, tail.x, tail.y);
    ctx.stroke();

    // Inner details - textured dotted scales
    ctx.strokeStyle = ownerId < 0 ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.1)";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(head.x, head.y);
    ctx.quadraticCurveTo(midX, midY, tail.x, tail.y);
    ctx.stroke();
    ctx.setLineDash([]); // Reset

    // 5. Draw head detailing (eyes blinking, animated forked tongue)
    const angle = Math.atan2(midY - head.y, midX - head.x);
    
    ctx.save();
    ctx.translate(head.x, head.y);
    ctx.rotate(angle);

    // Head base shape
    ctx.fillStyle = ownerId < 0 ? "#991b1b" : (NEON_COLORS[ownerId] || "#cbd5e1");
    ctx.beginPath();
    ctx.arc(0, 0, 8, 0, Math.PI * 2);
    ctx.fill();

    // Biting jaw highlight
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Blinking Snake Eyes
    const isBlinking = Math.sin(time * 0.4) > 0.93;
    ctx.fillStyle = isBlinking ? "#475569" : "#facc15";
    ctx.beginPath();
    ctx.arc(2, -3, 2, 0, Math.PI * 2);
    ctx.arc(2, 3, 2, 0, Math.PI * 2);
    ctx.fill();

    // Forked Red Tongue
    if (Math.sin(time * 2.5) > 0.4) {
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(8, 0);
      ctx.lineTo(13, 0);
      ctx.moveTo(13, 0);
      ctx.lineTo(16, -2);
      ctx.moveTo(13, 0);
      ctx.lineTo(16, 2);
      ctx.stroke();
    }

    ctx.restore();
  });

  // 6. Draw Player Tokens (with bouncing coordinates or idle floating bobbing)
  gameSession.players.forEach(p => {
    if (p.position < 1) return; // Not entered yet
    
    // Check if token coordinates are currently being animated by the queue
    let coord = null;
    if (p.animatingCoords) {
      coord = p.animatingCoords;
    } else {
      // Idle float bobbing
      const center = tileCenter(p.position);
      const idleBob = Math.sin(Date.now() * 0.004 + p.id) * 3;
      coord = { x: center.x, y: center.y + idleBob };
    }

    // Offset coordinates if multiple tokens are stacking on the same tile
    if (!p.animatingCoords) {
      const tileIndexStack = gameSession.players
        .filter(other => other.id < p.id && other.position === p.position && !other.animatingCoords)
        .length;
      if (tileIndexStack > 0) {
        coord.x += (tileIndexStack % 2 === 0 ? -12 : 12);
        coord.y += (tileIndexStack > 1 ? -12 : 12);
      }
    }

    // Shadow
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.beginPath();
    ctx.arc(coord.x + 1, coord.y + 4, 10, 0, Math.PI * 2);
    ctx.fill();

    // Highlight outer glowing turn ring if active
    if (gameSession.current_turn === p.id && !gameSession.winner) {
      ctx.strokeStyle = "#f5c443";
      ctx.lineWidth = 3 + Math.abs(Math.sin(Date.now() * 0.005) * 2);
      ctx.beginPath();
      ctx.arc(coord.x, coord.y, 13, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Colored Neon Sphere token
    const grad = ctx.createRadialGradient(coord.x - 3, coord.y - 3, 2, coord.x, coord.y, 10);
    const neonColor = NEON_COLORS[p.id];
    grad.addColorStop(0, "#fff");
    grad.addColorStop(0.4, neonColor);
    grad.addColorStop(1, "#0a0a14");

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(coord.x, coord.y, 10, 0, Math.PI * 2);
    ctx.fill();
    
    // Token board outline border
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(coord.x, coord.y, 10, 0, Math.PI * 2);
    ctx.stroke();

    // Floating text name label above
    ctx.font = "bold 9px 'Outfit', sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(p.name, coord.x, coord.y - 12);
  });

  // 7. Draw Placement preview highlighting cost previews
  if (currentHoveredTile && gameSession.started && settings.shopHintsEnabled) {
    const activePlayer = gameSession.players[gameSession.current_turn];
    if (activePlayer.position > 0 && activePlayer.can_buy_snake && !activePlayer.is_ai) {
      const head = +$("input-shop-head").value;
      const tail = +$("input-shop-tail").value;
      
      if (head && tail && head > tail) {
        const cost = calculateSnakeCostMock(activePlayer.snakes_owned.length, head - tail);
        const isAffordable = activePlayer.points >= cost;
        const valid = canPlaceSnakeMock(head, tail, activePlayer.points, cost, activePlayer.snakes_owned.length);
        
        // Render dashed connection line preview
        const headCenter = tileCenter(head);
        const tailCenter = tileCenter(tail);
        ctx.strokeStyle = valid ? "#10b981" : "#ef4444";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(headCenter.x, headCenter.y);
        ctx.lineTo(tailCenter.x, tailCenter.y);
        ctx.stroke();
        ctx.setLineDash([]); // Reset
      }
    }
  }

  ctx.restore();
}


// ── PARTICLES & EFFECTS RENDERERS ──
function spawnExplosionParticles(x, y, colorCount = 20) {
  if (!settings.particlesEnabled) return;
  for (let i = 0; i < colorCount; i++) {
    particles.push({
      x, y,
      vx: (Math.random() * 2 - 1) * 5,
      vy: (Math.random() * 2 - 1) * 5,
      rad: Math.random() * 4 + 2,
      color: i % 2 === 0 ? "#f5c443" : "#ff4a5a",
      life: 1.0,
      decay: Math.random() * 0.04 + 0.02
    });
  }
}

function spawnStealCoins(startX, startY, endX, endY) {
  if (!settings.particlesEnabled) return;
  const count = 12;
  for (let i = 0; i < count; i++) {
    setTimeout(() => {
      particles.push({
        x: startX,
        y: startY,
        targetX: endX,
        targetY: endY,
        isCoin: true,
        vx: (Math.random() * 2 - 1) * 3,
        vy: -Math.random() * 4 - 2, // blast up first
        rad: 3.5,
        color: "#f5c443", // Gold shiny
        life: 1.0,
        decay: 0.015,
        speed: 0.04 + Math.random() * 0.02
      });
    }, i * 60);
  }
}

function updateParticles() {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    
    if (p.isCoin) {
      // Coin particle slides towards target
      p.vx *= 0.95;
      p.vy += 0.2; // gravity drop
      p.x += p.vx;
      p.y += p.vy;

      // Arc Interpolation to target
      const dx = p.targetX - p.x;
      const dy = p.targetY - p.y;
      p.x += dx * p.speed;
      p.y += dy * p.speed;

      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < 10) {
        particles.splice(i, 1);
        playSFX("click"); // tiny click ding
        continue;
      }
    } else {
      // Standard spark explosion gravity sparks
      p.vx *= 0.96;
      p.vy += 0.15; // gravity
      p.x += p.vx;
      p.y += p.vy;
      p.life -= p.decay;

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }
    }

    // Draw
    ctx.save();
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.rad, 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
  }
}

function triggerFloatingNumber(tile, text, isGain = true) {
  if (!settings.floatNumsEnabled) return;
  const center = tileCenter(tile);
  floatingNumbers.push({
    x: center.x,
    y: center.y,
    text: text,
    isGain: isGain,
    life: 1.0,
    decay: 0.015
  });
}

function updateFloatingNumbers() {
  for (let i = floatingNumbers.length - 1; i >= 0; i--) {
    const f = floatingNumbers[i];
    f.y -= 1.1; // Drift upward
    f.life -= f.decay;
    
    if (f.life <= 0) {
      floatingNumbers.splice(i, 1);
      continue;
    }

    ctx.save();
    ctx.globalAlpha = f.life;
    ctx.font = "900 22px 'Outfit', sans-serif";
    ctx.fillStyle = f.isGain ? "#10b981" : "#ef4444";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 4;
    ctx.textAlign = "center";
    ctx.strokeText(f.text, f.x, f.y);
    ctx.fillText(f.text, f.x, f.y);
    ctx.restore();
  }
}


// ── ASYNCHRONOUS ANIMATION QUEUE & HOPPER ENGINE ──
function triggerScreenShake(strength = 15, durationMs = 400) {
  if (!settings.shakeEnabled) return;
  const startTime = Date.now();
  const shake = () => {
    const elapsed = Date.now() - startTime;
    if (elapsed < durationMs) {
      const decay = 1 - (elapsed / durationMs);
      boardShakeOffset.x = (Math.random() * 2 - 1) * strength * decay;
      boardShakeOffset.y = (Math.random() * 2 - 1) * strength * decay;
      requestAnimationFrame(shake);
    } else {
      boardShakeOffset.x = 0;
      boardShakeOffset.y = 0;
    }
  };
  requestAnimationFrame(shake);
}

function queueAnimation(animObj) {
  animationQueue.push(animObj);
  processAnimationQueue();
}

async function processAnimationQueue() {
  if (isAnimating) return;
  if (animationQueue.length === 0) return;

  isAnimating = true;
  const anim = animationQueue.shift();

  try {
    await runAnimation(anim);
  } catch (e) {
    console.error("Animation execution failed:", e);
  }

  isAnimating = false;
  // Chain next
  processAnimationQueue();
}

function runAnimation(anim) {
  const speedMult = getSpeedMultiplier();
  
  return new Promise((resolve) => {
    const player = gameSession.players.find(p => p.id === anim.playerId);
    
    if (anim.type === "hop_path") {
      // hop step by step through a series of tiles
      let stepIndex = 0;
      const path = anim.path;
      
      const doNextHop = () => {
        if (stepIndex >= path.length) {
          resolve();
          return;
        }
        
        const nextTile = path[stepIndex];
        const startTile = stepIndex === 0 ? anim.startTile : path[stepIndex - 1];
        
        animateSingleHop(player, startTile, nextTile, 300 / speedMult).then(() => {
          stepIndex++;
          doNextHop();
        });
      };
      
      doNextHop();
    }
    
    else if (anim.type === "slide") {
      // Snake sliding smooth animation along Bezier curve
      playSFX("snake_bite");
      const startCenter = tileCenter(anim.startTile);
      const endCenter = tileCenter(anim.endTile);
      
      // Control curves
      const timeOffset = Date.now() * 0.003;
      const controlSway = Math.sin(timeOffset + anim.startTile) * 22;
      const midX = (startCenter.x + endCenter.x) / 2;
      const midY = (startCenter.y + endCenter.y) / 2 + controlSway;

      let startTime = null;
      const duration = 750 / speedMult;

      const slideStep = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1.0);
        
        // Quad Bezier ease-in-out formula
        const t = progress;
        const x = (1 - t) * (1 - t) * startCenter.x + 2 * (1 - t) * t * midX + t * t * endCenter.x;
        const y = (1 - t) * (1 - t) * startCenter.y + 2 * (1 - t) * t * midY + t * t * endCenter.y;
        
        player.animatingCoords = { x, y };

        if (progress < 1.0) {
          requestAnimationFrame(slideStep);
        } else {
          player.animatingCoords = null; // restore idle bob
          player.position = anim.endTile;
          playSFX("land");
          spawnExplosionParticles(endCenter.x, endCenter.y, 10);
          resolve();
        }
      };
      requestAnimationFrame(slideStep);
    }
    
    else if (anim.type === "climb") {
      // Ladder rung by rung jump climb
      playSFX("ladder");
      const startCenter = tileCenter(anim.startTile);
      const endCenter = tileCenter(anim.endTile);
      
      let startTime = null;
      const duration = 600 / speedMult;

      const climbStep = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1.0);
        
        // Linear path scaling with an additive cheerful hop sine wave
        const x = startCenter.x + (endCenter.x - startCenter.x) * progress;
        const y = startCenter.y + (endCenter.y - startCenter.y) * progress;
        
        // Cheerful arpeggiator bounce jumps up rungs
        const hopsSine = Math.abs(Math.sin(progress * Math.PI * 4)) * 12;
        
        player.animatingCoords = { x, y: y - hopsSine };

        if (progress < 1.0) {
          requestAnimationFrame(climbStep);
        } else {
          player.animatingCoords = null;
          player.position = anim.endTile;
          playSFX("land");
          spawnExplosionParticles(endCenter.x, endCenter.y, 12);
          resolve();
        }
      };
      requestAnimationFrame(climbStep);
    }
    
    else if (anim.type === "bomb_explode") {
      // bomb hit: camera shake, screen explosion particles
      playSFX("bomb");
      triggerScreenShake(20, 600);
      const center = tileCenter(anim.tile);
      spawnExplosionParticles(center.x, center.y, 35);
      
      // fly back token hit jump
      animateSingleHop(player, anim.tile, anim.tile, 200 / speedMult).then(resolve);
    }
    
    else if (anim.type === "bankrupt_spin") {
      // fly spinning wipeout back to start
      playSFX("bankrupt");
      triggerScreenShake(25, 900);
      const startCenter = tileCenter(anim.startTile);
      const endCenter = tileCenter(0); // tile 0 center = start off board
      
      let startTime = null;
      const duration = 1100 / speedMult;

      const spinStep = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1.0);
        
        // High parabolic height arc jump back to start
        const arcHeight = 120 * Math.sin(progress * Math.PI);
        const x = startCenter.x + (endCenter.x - startCenter.x) * progress;
        const y = startCenter.y + (endCenter.y - startCenter.y) * progress - arcHeight;
        
        player.animatingCoords = { x, y };

        if (progress < 1.0) {
          requestAnimationFrame(spinStep);
        } else {
          player.animatingCoords = null;
          player.position = 0;
          playSFX("land");
          spawnExplosionParticles(endCenter.x, endCenter.y, 25);
          resolve();
        }
      };
      requestAnimationFrame(spinStep);
    }
    
    else if (anim.type === "point_steal_particles") {
      // generate gold coin flow from victim to owner
      playSFX("steal");
      const victimCenter = tileCenter(anim.victimTile);
      const ownerCenter = tileCenter(anim.ownerTile);
      
      spawnStealCoins(victimCenter.x, victimCenter.y, ownerCenter.x, ownerCenter.y);
      // Wait for coins fly animation delay
      setTimeout(resolve, 800 / speedMult);
    }
    
    else {
      resolve();
    }
  });
}

// Hop animation: squash and stretch bounce steps
function animateSingleHop(player, startTile, endTile, durationMs) {
  const start = tileCenter(startTile);
  const end = tileCenter(endTile);
  
  return new Promise((resolve) => {
    let startTime = null;
    playSFX("hop", player.id);

    const hopStep = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / durationMs, 1.0);
      
      // Hop parabola arc curve
      const arcHeight = 22 * Math.sin(progress * Math.PI);
      const x = start.x + (end.x - start.x) * progress;
      const y = start.y + (end.y - start.y) * progress - arcHeight;
      
      player.animatingCoords = { x, y };

      if (progress < 1.0) {
        requestAnimationFrame(hopStep);
      } else {
        player.animatingCoords = null;
        player.position = endTile;
        playSFX("land");
        resolve();
      }
    };
    requestAnimationFrame(hopStep);
  });
}


// ── 3D DICE SPINNING REVELATOR ──
function trigger3DDiceRoll(rollValue) {
  return new Promise((resolve) => {
    const diceCube = $("dice-cube");
    if (!diceCube) {
      resolve();
      return;
    }
    
    // Play Noise roll Klik and rattle SFX
    playSFX("dice", rollValue);

    // Apply spin CSS class
    diceCube.className = "dice-cube dice-tumbling";
    
    const durSec = settings.durDice / 10;
    
    setTimeout(() => {
      // Remove tumble class and lock face angle
      diceCube.className = `dice-cube show-${rollValue}`;
      
      // Screen shake bump pulse on landing
      triggerScreenShake(8, 250);
      
      // Slight delay for player to absorb result number
      setTimeout(resolve, 500);
    }, durSec * 1000);
  });
}


// ── GAME LOOP TURN CONTROLLERS ──
function renderGameScreen() {
  if (!gameSession.started) return;

  const activePlayer = gameSession.players[gameSession.current_turn];

  // Turn Header Name
  $("turn-pname").textContent = activePlayer.name;
  $("turn-pname").style.color = NEON_COLORS[activePlayer.id];
  
  const desc = activePlayer.is_ai 
    ? `🤖 ${activePlayer.difficulty.toUpperCase()} AI thinking...` 
    : "👤 YOUR TURN — Make moves!";
  $("turn-pdesc").textContent = desc;

  // Active status ring highlight around timers
  if (activePlayer.is_ai) {
    $("timer-ring-container").classList.add("hidden");
  } else {
    $("timer-ring-container").classList.remove("hidden");
  }

  // Shop configuration panel visibility
  const canShop = !activePlayer.is_ai && activePlayer.position > 0 && activePlayer.snake_count < 3;
  if (canShop) {
    $("shop-panel").classList.remove("hidden");
    
    // Autofill limits safely
    $("input-shop-head").value = 50;
    $("input-shop-tail").value = 10;
    updateShopCostPreview();
  } else {
    $("shop-panel").classList.add("hidden");
  }

  // Roll Button availability
  if (activePlayer.is_ai || isAnimating) {
    $("btn-action-roll").disabled = true;
  } else {
    $("btn-action-roll").disabled = false;
  }

  // Standings player cards
  $("players-cards").innerHTML = gameSession.players.map(p => {
    const isActive = p.id === gameSession.current_turn && !gameSession.winner;
    const progressPct = Math.min((p.position / 100) * 100, 100);
    return `
      <div class="player-card ${isActive ? 'active-card' : ''}">
        <div class="pcard-indicator" style="background-color: ${NEON_COLORS[p.id]}"></div>
        <div class="pcard-main">
          <div class="pcard-name" style="color: ${NEON_COLORS[p.id]}">${p.name}</div>
          <div class="pcard-meta">${p.is_ai ? p.difficulty + ' bot' : 'human'}</div>
          
          <div class="pcard-position-wrap">
            <div class="pcard-progress-bar">
              <div class="pcard-progress-fill" style="width: ${progressPct}%; background-color: ${NEON_COLORS[p.id]}"></div>
            </div>
            <span class="pcard-pos-text">Tile ${p.position}</span>
          </div>
        </div>
        <div class="pcard-score">
          <div class="pcard-points">${p.points} <span style="font-size: 0.75rem; color: #a0a0a0">pts</span></div>
          <div class="pcard-snakes">🐍 ${p.snake_count}/3 owned</div>
        </div>
      </div>
    `;
  }).join("");

  // Clean in-game background dim preview overlays if hover set
  $("game-canvas").style.cursor = canShop ? "crosshair" : "default";
}

// Emote click triggers
document.querySelectorAll(".btn-emote").forEach(btn => {
  btn.addEventListener("click", () => {
    const type = btn.getAttribute("data-emote");
    const activePlayer = gameSession.players[gameSession.current_turn];
    if (activePlayer.is_ai) return; // Only humans click
    
    executeEmote(activePlayer.id, type);
  });
});

function executeEmote(playerId, type) {
  playSFX("emote", type);
  const player = gameSession.players.find(p => p.id === playerId);
  if (!player || player.position < 1) return;

  const center = tileCenter(player.position);
  
  // Create bubble element over token coordinates on screen
  const container = $("canvas-overlay-container");
  const bubble = document.createElement("div");
  bubble.className = "emote-bubble";
  
  // Translate center to canvas viewport CSS percentages
  const leftPct = (center.x / 750) * 100;
  const topPct = (center.y / 750) * 100;
  
  bubble.style.left = `${leftPct}%`;
  bubble.style.top = `${topPct}%`;
  bubble.innerHTML = `<span>${EMOTE_EMOJIS[type]}</span>`;
  
  container.appendChild(bubble);

  // Auto remove after animation completes
  setTimeout(() => {
    bubble.remove();
  }, 2200);

  appendLog(`💬 [Emote] ${player.name} reacts: ${type.toUpperCase()}!`);
}

// ── CUSTOM STATE CLIENT ENGINE TURN LOGIC (STALL-FREE TRACED LOOP) ──

$("btn-action-roll").addEventListener("click", async () => {
  // Humans roll the dice
  const activePlayer = gameSession.players[gameSession.current_turn];
  if (activePlayer.is_ai || isAnimating) return;

  stopTurnTimer();
  await executeCoreTurn(null);
});

// Shop Cost Preview Calculations
$("input-shop-head").addEventListener("input", updateShopCostPreview);
$("input-shop-tail").addEventListener("input", updateShopCostPreview);

async function updateShopCostPreview() {
  const head = +$("input-shop-head").value;
  const tail = +$("input-shop-tail").value;
  const activePlayer = gameSession.players[gameSession.current_turn];
  
  const label = $("shop-preview-cost");
  if (head <= tail || head < 20 || head > 90 || tail < 1) {
    label.textContent = "-- pts";
    label.style.color = "inherit";
    return;
  }

  const cost = calculateSnakeCostMock(activePlayer.snakes_owned.length, head - tail);
  label.textContent = `${cost} pts`;
  
  if (activePlayer.points >= cost) {
    label.style.color = "#10b981";
  } else {
    label.style.color = "#ef4444";
  }
}

// Buy Snake click handler
$("btn-shop-buy").addEventListener("click", async () => {
  const head = +$("input-shop-head").value;
  const tail = +$("input-shop-tail").value;
  const activePlayer = gameSession.players[gameSession.current_turn];

  if (activePlayer.is_ai || isAnimating) return;

  const cost = calculateSnakeCostMock(activePlayer.snakes_owned.length, head - tail);
  const valid = canPlaceSnakeMock(head, tail, activePlayer.points, cost, activePlayer.snakes_owned.length);

  if (!valid.ok) {
    $("shop-feedback").textContent = `❌ ${valid.reason}`;
    playSFX("click");
    // Shake buy panel button to register frustration juice
    const buyBtn = $("btn-shop-buy");
    buyBtn.classList.add("btn-shake-error");
    setTimeout(() => buyBtn.classList.remove("btn-shake-error"), 500);
    return;
  }

  $("shop-feedback").textContent = "";
  
  // Deduct locally immediately
  activePlayer.points -= cost;
  const newSnake = [head, tail, activePlayer.id];
  gameSession.snakes.push(newSnake);
  activePlayer.snakes_owned.push(newSnake);
  
  gameSession.stats[activePlayer.id].snakesPlaced++;

  // Animate placing
  playSFX("shop_buy");
  appendLog(`🛒 [Shop] ${activePlayer.name} placed snake ${head}➔${tail} for ${cost} pts.`);
  
  renderGameScreen();

  // Hide shop to proceed with rolling
  $("shop-panel").classList.add("hidden");
});

// Mock rule checks in client for quick placements checks
function calculateSnakeCostMock(currentSnakeCount, length) {
  const count = currentSnakeCount + 1;
  const cost = Math.floor(2 * count * Math.pow(length, 0.9));
  return Math.max(cost, 12);
}

function canPlaceSnakeMock(head, tail, walletPoints, cost, ownedCount) {
  if (head <= tail) return { ok: false, reason: "Head must be higher than tail." };
  if (head > 90) return { ok: false, reason: "Snake head cannot be placed above tile 90." };
  if (ownedCount >= 3) return { ok: false, reason: "You already own the maximum of 3 snakes." };
  if (walletPoints < cost) return { ok: false, reason: `Not enough points. Need ${cost}, have ${walletPoints}` };

  // Occupied/ladder check
  const playerTiles = new Set(gameSession.players.map(p => p.position).filter(pos => pos > 0));
  if (playerTiles.has(head) || playerTiles.has(tail)) return { ok: false, reason: "Cannot place on occupied tiles." };

  const ladderBottoms = new Set(gameSession.ladders.map(l => l[0]));
  const ladderTops = new Set(gameSession.ladders.map(l => l[1]));
  if (ladderBottoms.has(head) || ladderTops.has(head) || ladderBottoms.has(tail) || ladderTops.has(tail)) {
    return { ok: false, reason: "Cannot place snake on a ladder tile." };
  }

  // Chaining check
  const snakeHeads = new Set(gameSession.snakes.map(s => s[0]));
  const snakeTails = new Set(gameSession.snakes.map(s => s[1]));
  if (snakeTails.has(head)) return { ok: false, reason: "Cannot place head at another snake's tail." };
  if (snakeHeads.has(tail)) return { ok: false, reason: "Cannot place tail at another snake's head." };

  // Anti-wall check
  const prospectiveHeads = new Set([...snakeHeads, head]);
  if (checkHeadRunMock(prospectiveHeads, head) > 2) {
    return { ok: false, reason: "Would create too long a wall of snake heads." };
  }

  return { ok: true };
}

function checkHeadRunMock(headsSet, newHead) {
  let left = newHead - 1;
  while (headsSet.has(left)) left--;
  let right = newHead + 1;
  while (headsSet.has(right)) right++;
  return right - left - 1;
}

// ── THE CRITICAL STALL-FREE CORE TURN EXECUTOR ──
async function executeCoreTurn(aiShopDecision) {
  const activePlayer = gameSession.players[gameSession.current_turn];
  gameSession.stats[activePlayer.id].turnsTaken++;

  // 1. Check if AI bought a snake
  if (aiShopDecision && activePlayer.is_ai) {
    const head = aiShopDecision.head;
    const tail = aiShopDecision.tail;
    const cost = calculateSnakeCostMock(activePlayer.snakes_owned.length, head - tail);

    // Spotlight animations before placing snake
    currentHoveredTile = head;
    $("input-shop-head").value = head;
    $("input-shop-tail").value = tail;
    renderGameScreen();

    // Small delay to visually highlight AI shop placement
    await new Promise(r => setTimeout(r, 800));

    // Place it
    activePlayer.points -= cost;
    const newSnake = [head, tail, activePlayer.id];
    gameSession.snakes.push(newSnake);
    activePlayer.snakes_owned.push(newSnake);
    
    gameSession.stats[activePlayer.id].snakesPlaced++;
    playSFX("shop_buy");
    appendLog(`🛒 [Shop] ${activePlayer.name} (AI) bought snake ${head}➔${tail} for ${cost} pts.`);

    currentHoveredTile = null;
    renderGameScreen();
    await new Promise(r => setTimeout(r, 400));
  }

  // 2. Roll Dice & Spin
  const roll = Math.floor(Math.random() * 6) + 1;
  
  // Record History for watch replay
  gameSession.actionHistory.push({
    type: "turn_start",
    player: { ...activePlayer },
    shop: aiShopDecision ? { ...aiShopDecision } : null,
    roll: roll,
    boardState: JSON.parse(JSON.stringify({
      snakes: gameSession.snakes,
      players: gameSession.players
    }))
  });

  // Roll animations
  await trigger3DDiceRoll(roll);

  // 3. Move Player logic
  const oldPos = activePlayer.position;
  const rawPos = oldPos + roll;

  if (rawPos > 100) {
    // Overshoot exact rule stay put
    appendLog(`🚫 [Skip] ${activePlayer.name} rolled ${roll} and overshot tile 100! (Exact roll needed)`);
    triggerFloatingNumber(oldPos, "SKIPPED", false);
    
    // Advance turns
    advanceTurnLoop();
    return;
  }

  // Hop animations step by step
  const hopPath = [];
  for (let step = oldPos + 1; step <= rawPos; step++) {
    hopPath.push(step);
  }
  
  appendLog(`🎲 [Roll] ${activePlayer.name} rolled ${roll} and moves from ${oldPos}➔${rawPos}.`);
  
  // Push hop animation to queue and wait
  await new Promise((resolve) => {
    queueAnimation({
      type: "hop_path",
      playerId: activePlayer.id,
      path: hopPath,
      startTile: oldPos
    });
    // Wait for queue execution
    checkAnimationCompletion(resolve);
  });

  activePlayer.position = rawPos;
  renderGameScreen();

  // Apply sequential Triggers: Ladders, Snakes, Bombs
  await executeLandingsCascade(activePlayer);

  // 4. Earn Tile Income
  const baseValue = 3;
  const tilePoints = activePlayer.position === 100 ? 40 : ((baseValue + 1) + Math.floor(activePlayer.position / 15) + Math.floor(Math.random() * 5));
  activePlayer.points += tilePoints;
  gameSession.stats[activePlayer.id].pointsEarned += tilePoints;
  
  triggerFloatingNumber(activePlayer.position, `+${tilePoints} pts`, true);
  appendLog(`💰 [Income] ${activePlayer.name} earned ${tilePoints} points on tile ${activePlayer.position}.`);
  renderGameScreen();
  await new Promise(r => setTimeout(r, 450));

  // 5. Landed on Bombs Check
  const bombsSet = new Set(gameSession.bombs);
  if (bombsSet.has(activePlayer.position)) {
    const bombLoss = 18 + Math.floor(activePlayer.position / 6);
    appendLog(`💣 [Bomb] ${activePlayer.name} triggered a bomb on tile ${activePlayer.position}! Wiping ${bombLoss} pts.`);
    gameSession.stats[activePlayer.id].bombsHit++;

    await new Promise((resolve) => {
      queueAnimation({
        type: "bomb_explode",
        playerId: activePlayer.id,
        tile: activePlayer.position
      });
      checkAnimationCompletion(resolve);
    });

    const isSolvent = deductPoints(activePlayer, bombLoss);
    if (!isSolvent) {
      // Bankruptcy resets
      await executeBankruptcySequence(activePlayer);
    }
    renderGameScreen();
    await new Promise(r => setTimeout(r, 450));
  }

  // 6. Win check
  if (activePlayer.position >= 100) {
    handleVictory(activePlayer);
    return;
  }

  // 7. Advance turn cycle
  advanceTurnLoop();
}

// Deduction point helper that triggers bankruptcy resets
function deductPoints(player, amount) {
  player.points -= amount;
  if (player.points < 0) {
    player.points = 0;
    return false;
  }
  return true;
}

// Executes consecutive Ladder climbs or Snake slides until stable
async function executeLandingsCascade(player) {
  let cascadeLimit = 0;
  
  while (cascadeLimit < 15) {
    cascadeLimit++;
    const currentTile = player.position;

    // Check ladders first
    const ladder = gameSession.ladders.find(l => l[0] === currentTile);
    if (ladder) {
      appendLog(`🪜 [Ladder] ${player.name} climbed ladder ${ladder[0]}➔${ladder[1]}!`);
      gameSession.stats[player.id].laddersClimbed++;

      await new Promise((resolve) => {
        queueAnimation({
          type: "climb",
          playerId: player.id,
          startTile: ladder[0],
          endTile: ladder[1]
        });
        checkAnimationCompletion(resolve);
      });
      player.position = ladder[1];
      renderGameScreen();
      continue;
    }

    // Check snakes triggers
    const strikingSnake = getStrikingSnake(currentTile, player.id);
    if (strikingSnake) {
      const head = strikingSnake[0];
      const tail = strikingSnake[1];
      const ownerId = strikingSnake[2];
      
      appendLog(`🐍 [Bite] ${player.name} stepped into snake trap at head ${head} and slides➔${tail}!`);
      gameSession.stats[player.id].bittenCount++;

      await new Promise((resolve) => {
        queueAnimation({
          type: "slide",
          playerId: player.id,
          startTile: currentTile,
          endTile: tail
        });
        checkAnimationCompletion(resolve);
      });

      player.position = tail;
      renderGameScreen();

      // Steal point from victim to owner
      if (ownerId >= 0) {
        const flatSteal = 15;
        const pctSteal = Math.floor(player.points * 0.3);
        const stealAmount = Math.min(player.points, flatSteal + pctSteal);

        if (stealAmount > 0) {
          const owner = gameSession.players.find(p => p.id === ownerId);
          
          // Deduct from victim
          const solvent = deductPoints(player, stealAmount);
          
          // Add to owner
          owner.points += stealAmount;
          gameSession.stats[ownerId].pointsStolen += stealAmount;

          appendLog(`💸 [Theft] Snake trap robs ${stealAmount} pts from ${player.name}➔${owner.name}!`);

          // Coin animations
          await new Promise((resolve) => {
            queueAnimation({
              type: "point_steal_particles",
              playerId: owner.id,
              victimTile: tail,
              ownerTile: owner.position
            });
            checkAnimationCompletion(resolve);
          });

          triggerFloatingNumber(tail, `-${stealAmount}`, false);
          triggerFloatingNumber(owner.position, `+${stealAmount}`, true);
          
          if (!solvent) {
            await executeBankruptcySequence(player);
          }
        }
      }

      // Consume Player placed snakes (single-use)
      if (ownerId >= 0) {
        gameSession.snakes = gameSession.snakes.filter(s => !(s[0] === head && s[1] === tail && s[2] === ownerId));
        const owner = gameSession.players.find(p => p.id === ownerId);
        owner.snakes_owned = owner.snakes_owned.filter(s => !(s[0] === head && s[1] === tail && s[2] === ownerId));
        appendLog(`💥 [Trap] Trap at head ${head} is consumed and removed.`);
      }

      renderGameScreen();
      continue;
    }

    // Stable, nothing triggered
    break;
  }
}

// Owner immune strike range finder
function getStrikingSnake(tile, playerId) {
  let best = null;
  gameSession.snakes.forEach(s => {
    const head = s[0];
    const tail = s[1];
    const ownerId = s[2];
    
    if (ownerId === playerId) return; // Immune

    const range = ownerId >= 0 ? 4 : 0; // strike range head+4 below, board=exact head
    if (head - range <= tile && tile <= head) {
      if (best === null || head > best[0]) {
        best = s;
      }
    }
  });
  return best;
}

// Executes Bankruptcy Sequence resets to start
async function executeBankruptcySequence(player) {
  appendLog(`☠️ [BANKRUPT] ${player.name} is bankrupted! Reset to start.`);
  gameSession.stats[player.id].bankruptCount++;
  player.bankrupt_count++;

  const oldPos = player.position;

  await new Promise((resolve) => {
    queueAnimation({
      type: "bankrupt_spin",
      playerId: player.id,
      startTile: oldPos
    });
    checkAnimationCompletion(resolve);
  });

  player.position = 0;
  player.points = 0;
  renderGameScreen();
}

function checkAnimationCompletion(callback) {
  const check = () => {
    if (!isAnimating && animationQueue.length === 0) {
      callback();
    } else {
      setTimeout(check, 50);
    }
  };
  check();
}


// ── TURN SYSTEM ADVANCEMENT LOOP ──
function advanceTurnLoop() {
  if (gameSession.winner) return;

  // Advance player turn index
  gameSession.current_turn = (gameSession.current_turn + 1) % gameSession.players.length;
  if (gameSession.current_turn === 0) {
    gameSession.turn_number++;
  }

  renderGameScreen();

  // If next is AI, trigger check
  checkAndExecuteAITurn();
}

async function checkAndExecuteAITurn() {
  if (gameSession.winner) return;

  const activePlayer = gameSession.players[gameSession.current_turn];
  if (!activePlayer.is_ai) {
    // Human turn: start countdown turn timer
    startTurnTimer();
    return;
  }

  // AI Turn Loop
  stopTurnTimer();
  
  // 1. Show "AI is thinking..." ticking panel
  $("thinking-indicator").classList.remove("hidden");
  $("btn-action-roll").disabled = true;

  // Play light ticking clock SFX
  const tickInterval = setInterval(() => {
    if (!$("thinking-indicator").classList.contains("hidden")) {
      playSFX("thinking");
    } else {
      clearInterval(tickInterval);
    }
  }, 400);

  // Artificial thinking delay (1.5s) to feel organic
  await new Promise(r => setTimeout(r, 1200));

  // Build JSON BoardState to send to stateless Flask API
  const boardPayload = {
    tiles: gameSession.tiles,
    ladders: gameSession.ladders,
    snakes: gameSession.snakes,
    bombs: gameSession.bombs,
    players: gameSession.players.map(p => ({
      id: p.id,
      name: p.name,
      position: p.position,
      points: p.points,
      is_ai: p.is_ai,
      difficulty: p.difficulty,
      bankrupt_count: p.bankrupt_count
    })),
    current_turn: gameSession.current_turn,
    turn_number: gameSession.turn_number
  };

  try {
    const aiMoveDecision = await api("/api/ai-move", {
      board: boardPayload,
      activePlayerId: activePlayer.id
    });

    // Hide indicator spinner
    clearInterval(tickInterval);
    $("thinking-indicator").classList.add("hidden");

    // Execute core turn with AI decision
    await executeCoreTurn(aiMoveDecision);

    // AI occasionally emotes automatically to feel ALIVE!
    if (settings.aiEmotesEnabled && Math.random() < 0.28) {
      triggerAutomaticAIEmote(activePlayer.id);
    }

  } catch (e) {
    console.error("AI Turn API execution error:", e);
    // Graceful fallback to avoid lockups: just execute simple roll
    clearInterval(tickInterval);
    $("thinking-indicator").classList.add("hidden");
    await executeCoreTurn(null);
  }
}

// Automatic AI emote selector
function triggerAutomaticAIEmote(playerId) {
  const emotesList = ["laughing", "shocked", "taunting", "celebrating", "sad"];
  const selected = emotesList[Math.floor(Math.random() * emotesList.length)];
  
  // Brief delay after move completes to react
  setTimeout(() => {
    executeEmote(playerId, selected);
  }, 1000);
}


// ── TURN TIMER RING CONTROLLER ──
function startTurnTimer() {
  stopTurnTimer();
  if (!settings.turnTimerEnabled) {
    $("timer-text").textContent = "--";
    return;
  }

  gameSession.timerSecondsLeft = settings.turnTimerDur;
  $("timer-text").textContent = gameSession.timerSecondsLeft;

  const timerFill = $("timer-ring-fill");
  const circumference = 2 * Math.PI * 16;
  timerFill.style.strokeDasharray = `${circumference} ${circumference}`;
  timerFill.style.strokeDashoffset = 0;

  gameSession.timerInterval = setInterval(() => {
    gameSession.timerSecondsLeft--;
    $("timer-text").textContent = gameSession.timerSecondsLeft;

    // Redraw ring offset
    const offset = circumference - (gameSession.timerSecondsLeft / settings.turnTimerDur) * circumference;
    timerFill.style.strokeDashoffset = offset;

    // Ring colors turn yellow/red when low
    if (gameSession.timerSecondsLeft <= 5) {
      timerFill.style.stroke = "#ef4444";
    } else if (gameSession.timerSecondsLeft <= 10) {
      timerFill.style.stroke = "#f59e0b";
    } else {
      timerFill.style.stroke = "#f5c443";
    }

    if (gameSession.timerSecondsLeft <= 0) {
      stopTurnTimer();
      appendLog(`⏳ [Timer] Time ran out! Force rolling.`);
      $("btn-action-roll").click(); // Auto roll
    }
  }, 1000);
}

function stopTurnTimer() {
  if (gameSession.timerInterval) {
    clearInterval(gameSession.timerInterval);
    gameSession.timerInterval = null;
  }
}


// ── VICTORY & CONFETTI CELEBRATIONS ──
let confettiInterval = null;

function handleVictory(winner) {
  gameSession.winner = winner.name;
  stopTurnTimer();
  stopBGM();
  playSFX("win");

  // Open overlay screen
  $("victory-winner-announcement").textContent = winner.name;
  $("victory-winner-announcement").style.color = NEON_COLORS[winner.id];
  
  // Render match summary grid cards
  const stats = gameSession.stats[winner.id];
  $("victory-stats-grid").innerHTML = `
    <div class="vstat-card">
      <div class="vstat-label">🐍 Snakes Placed</div>
      <div class="vstat-val">${stats.snakesPlaced}</div>
    </div>
    <div class="vstat-card">
      <div class="vstat-label">🦷 Bitten Count</div>
      <div class="vstat-val">${stats.bittenCount}</div>
    </div>
    <div class="vstat-card">
      <div class="vstat-label">☠️ Bankrupt resets</div>
      <div class="vstat-val">${stats.bankruptCount}</div>
    </div>
    <div class="vstat-card">
      <div class="vstat-label">🪜 Ladders Climbed</div>
      <div class="vstat-val">${stats.laddersClimbed}</div>
    </div>
    <div class="vstat-card">
      <div class="vstat-label">💣 Bombs Exploded</div>
      <div class="vstat-val">${stats.bombsHit}</div>
    </div>
    <div class="vstat-card">
      <div class="vstat-label">📈 Turns Taken</div>
      <div class="vstat-val">${stats.turnsTaken}</div>
    </div>
  `;

  $("screen-victory").classList.remove("hidden");
  startConfettiEngine();
}

function startConfettiEngine() {
  const confCanvas = $("victory-confetti");
  const confCtx = confCanvas.getContext("2d");
  
  const resize = () => {
    confCanvas.width = window.innerWidth;
    confCanvas.height = window.innerHeight;
  };
  window.addEventListener("resize", resize);
  resize();

  const confettiPieces = [];
  const colors = ["#ff4a5a", "#3b82f6", "#10b981", "#f59e0b", "#f5c443", "#a78bfa"];

  for (let i = 0; i < 150; i++) {
    confettiPieces.push({
      x: Math.random() * confCanvas.width,
      y: Math.random() * confCanvas.height - confCanvas.height,
      r: Math.random() * 6 + 4,
      d: Math.random() * confCanvas.height,
      color: colors[Math.floor(Math.random() * colors.length)],
      tilt: Math.random() * 10 - 5,
      tiltAngleIncremental: Math.random() * 0.07 + 0.02,
      tiltAngle: 0
    });
  }

  const drawConfetti = () => {
    confCtx.clearRect(0, 0, confCanvas.width, confCanvas.height);
    
    let active = false;
    confettiPieces.forEach(p => {
      p.tiltAngle += p.tiltAngleIncremental;
      p.y += (Math.cos(p.d) + 3 + p.r / 2) * 0.5;
      p.x += Math.sin(p.tiltAngle);
      p.tilt = Math.sin(p.tiltAngle - p.r / 2) * 15;

      if (p.y <= confCanvas.height) {
        active = true;
      } else {
        // Reset to top to loop fireworks confetti
        p.y = -20;
        p.x = Math.random() * confCanvas.width;
      }

      confCtx.beginPath();
      confCtx.lineWidth = p.r;
      confCtx.strokeStyle = p.color;
      confCtx.moveTo(p.x + p.tilt + p.r / 2, p.y);
      confCtx.lineTo(p.x + p.tilt, p.y + p.tilt + p.r / 2);
      confCtx.stroke();
    });

    if (active && !$("screen-victory").classList.contains("hidden")) {
      confettiInterval = requestAnimationFrame(drawConfetti);
    }
  };

  confettiInterval = requestAnimationFrame(drawConfetti);
}

function stopConfettiEngine() {
  if (confettiInterval) {
    cancelAnimationFrame(confettiInterval);
    confettiInterval = null;
  }
}

// ── MATCH WATCH REPLAY CONTROLLER (2x ACCELERATED PLAYBACK) ──
$("btn-victory-replay").addEventListener("click", () => {
  stopConfettiEngine();
  $("screen-victory").classList.add("hidden");
  
  executeGameReplay();
});

async function executeGameReplay() {
  appendLog("\n📺 [Replay] STARTING MATCH WATCH REPLAY AT 2x SPEED...");
  
  // Temporarily force Turbo speed
  const originalSpeed = settings.speed;
  settings.speed = "fast";
  
  // Reset players coordinates
  gameSession.players.forEach(p => {
    p.position = 0;
    p.points = 0;
    p.snakes_owned = [];
  });
  gameSession.snakes = [];
  
  isAnimating = false;
  animationQueue = [];

  for (let idx = 0; idx < gameSession.actionHistory.length; idx++) {
    const act = gameSession.actionHistory[idx];
    const player = gameSession.players.find(p => p.id === act.player.id);
    
    appendLog(`🎬 [Replay Turn ${idx+1}] ${player.name} plays:`);
    
    // Re-apply snake buys
    if (act.shop) {
      const head = act.shop.head;
      const tail = act.shop.tail;
      const cost = calculateSnakeCostMock(player.snakes_owned.length, head - tail);
      player.points -= cost;
      
      const newSnake = [head, tail, player.id];
      gameSession.snakes.push(newSnake);
      player.snakes_owned.push(newSnake);
      
      playSFX("shop_buy");
      renderGameScreen();
      await new Promise(r => setTimeout(r, 200));
    }

    // 3D dice rolls
    await trigger3DDiceRoll(act.roll);

    // Reconstruct moves path
    const oldPos = player.position;
    const rawPos = oldPos + act.roll;

    if (rawPos <= 100) {
      const path = [];
      for (let s = oldPos + 1; s <= rawPos; s++) path.push(s);
      
      await new Promise((resolve) => {
        queueAnimation({
          type: "hop_path",
          playerId: player.id,
          path: path,
          startTile: oldPos
        });
        checkAnimationCompletion(resolve);
      });
      player.position = rawPos;

      // Triggers climb slide checks
      await executeLandingsReplayCascade(player, act.boardState);
    }
    
    renderGameScreen();
    await new Promise(r => setTimeout(r, 300));
  }

  // Restore Speed
  settings.speed = originalSpeed;
  appendLog("📺 [Replay] MATCH WATCH REPLAY COMPLETED!");
  
  // Re-open victory modal
  const winner = gameSession.players.find(p => p.name === gameSession.winner);
  handleVictory(winner);
}

// Ladder and snake checks during replays mapping state snapshots
async function executeLandingsReplayCascade(player, stateSnapshot) {
  let cascadeLimit = 0;
  while (cascadeLimit < 15) {
    cascadeLimit++;
    const currentTile = player.position;

    // Check ladders
    const ladder = gameSession.ladders.find(l => l[0] === currentTile);
    if (ladder) {
      await new Promise((resolve) => {
        queueAnimation({
          type: "climb",
          playerId: player.id,
          startTile: ladder[0],
          endTile: ladder[1]
        });
        checkAnimationCompletion(resolve);
      });
      player.position = ladder[1];
      continue;
    }

    // Check snakes using snapshot
    const strikingSnake = getStrikingSnake(currentTile, player.id);
    if (strikingSnake) {
      const head = strikingSnake[0];
      const tail = strikingSnake[1];
      const ownerId = strikingSnake[2];
      
      await new Promise((resolve) => {
        queueAnimation({
          type: "slide",
          playerId: player.id,
          startTile: currentTile,
          endTile: tail
        });
        checkAnimationCompletion(resolve);
      });

      player.position = tail;

      if (ownerId >= 0) {
        const owner = gameSession.players.find(p => p.id === ownerId);
        await new Promise((resolve) => {
          queueAnimation({
            type: "point_steal_particles",
            playerId: owner.id,
            victimTile: tail,
            ownerTile: owner.position
          });
          checkAnimationCompletion(resolve);
        });
      }
      
      // Consume
      if (ownerId >= 0) {
        gameSession.snakes = gameSession.snakes.filter(s => !(s[0] === head && s[1] === tail && s[2] === ownerId));
      }
      continue;
    }
    break;
  }
}


// ── LOG PANEL SCROLLER ──
function appendLog(text) {
  const battleLog = $("battle-log-lines");
  if (!battleLog) return;
  
  const div = document.createElement("div");
  div.className = "log-line-item";
  div.innerHTML = text;
  battleLog.appendChild(div);
  
  // Auto scroll
  battleLog.scrollTop = battleLog.scrollHeight;
}


// ── RESIGN / RESET TRIGGERS ──
$("btn-game-quit").addEventListener("click", () => {
  const confirmResign = confirm("Are you sure you want to resign and quit this game session?");
  if (confirmResign) {
    stopTurnTimer();
    stopBGM();
    gameSession.started = false;
    showScreen("menu");
  }
});

$("btn-victory-lobby").addEventListener("click", () => {
  stopConfettiEngine();
  $("screen-victory").classList.add("hidden");
  showScreen("lobby");
});

$("btn-victory-menu").addEventListener("click", () => {
  stopConfettiEngine();
  $("screen-victory").classList.add("hidden");
  showScreen("menu");
});


// ── INITIAL BOOTSTRAP ──
window.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  spawnMenuParticles();
  showScreen("menu");
  
  // Add dynamic coordinate tracking listeners on window hover for spotlights
  const boardWrap = $("boardWrap");
  if (boardWrap) {
    boardWrap.addEventListener("mousemove", (e) => {
      // Find matching hovered tile index based on grid mouse position offsets
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      
      const sizeX = rect.width / 10;
      const sizeY = rect.height / 10;
      
      const col = Math.floor(mouseX / sizeX);
      const row = 9 - Math.floor(mouseY / sizeY);
      
      if (col >= 0 && col < 10 && row >= 0 && row < 10) {
        currentHoveredTile = (row % 2 === 0) ? (row * 10 + col + 1) : (row * 10 + (10 - col));
      }
    });

    boardWrap.addEventListener("mouseleave", () => {
      currentHoveredTile = null;
    });
  }
});
