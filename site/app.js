"use strict";

const SIZE = 6;
const START = 36;
const PASS = 37;
const END = 38;
const replays = window.REVERSI_REPLAYS ?? [];
const element = (id) => document.getElementById(id);

let mode = "replay";
let replay = replays[0];
let liveFrames = [];
let position = replay?.focus ?? 0;
let requestVersion = 0;
let waiting = false;

function tokenName(token) {
  if (token === START) return "START";
  if (token === PASS) return "PASS";
  if (token === END) return "END";
  return String.fromCharCode(65 + token % SIZE) + (Math.floor(token / SIZE) + 1);
}

function make(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const scenarioButtons = replays.map((item, index) => {
  const button = make("button", "", String(index + 1).padStart(2, "0") + "  " + item.title);
  button.type = "button";
  button.addEventListener("click", () => showReplay(item));
  element("scenarios").append(button);
  return button;
});

const squares = [];
const board = element("board");
for (let row = -1; row < SIZE; row += 1) {
  for (let column = -1; column < SIZE; column += 1) {
    if (row < 0 || column < 0) {
      const coordinate = row < 0 ? (column < 0 ? "" : String.fromCharCode(65 + column)) : String(row + 1);
      const label = make("span", "coordinate", coordinate);
      label.setAttribute("aria-hidden", "true");
      board.append(label);
      continue;
    }
    const token = row * SIZE + column;
    const square = make("button", "square");
    square.type = "button";
    square.classList.toggle("alt", (row + column) % 2 === 1);
    square.addEventListener("click", () => play(token));
    board.append(square);
    squares.push(square);
  }
}

function error(message = "") {
  element("error").hidden = !message;
  element("error").textContent = message;
}

function frames() {
  return mode === "live" ? liveFrames : (replay?.frames ?? []);
}

function percent(probability) {
  return (probability * 100).toFixed(1) + "%";
}

function rankedMoves(frame, filtered, mass) {
  return (frame.probabilities ?? [])
    .map((value, token) => ({ token, value, legal: frame.legal.includes(token) }))
    .filter((move) => !filtered || move.legal)
    .sort((a, b) => b.value - a.value)
    .slice(0, 5)
    .map((move) => ({ ...move, value: filtered && mass > 0 ? move.value / mass : move.value }));
}

function drawBoard(frame, prediction) {
  const last = frame.history.at(-1);
  for (const [token, square] of squares.entries()) {
    const cell = frame.board[token];
    const legal = frame.legal.includes(token);
    const predicted = prediction === token;
    const label = [tokenName(token), cell === 1 ? "Black disc" : cell === -1 ? "White disc" : "Empty"];
    if (legal) label.push("legal move");
    if (predicted) label.push("top prediction");
    if (last === token) label.push("last move");
    square.classList.toggle("legal", legal);
    square.classList.toggle("predicted", predicted);
    square.setAttribute("aria-label", label.join(", "));
    square.disabled = mode !== "live" || !legal || waiting;
    if (cell) {
      const disc = make("span", cell === 1 ? "disc" : "disc white");
      if (last === token) disc.classList.add("last");
      square.replaceChildren(disc);
    } else {
      square.replaceChildren();
    }
  }
}

function drawPredictions(moves) {
  const entries = moves.map((move) => {
    const row = make("li", move.legal ? "prediction" : "prediction illegal");
    const detail = make("div", "prediction-main");
    const track = make("div", "prediction-bar");
    const fill = document.createElement("span");
    track.setAttribute("aria-hidden", "true");
    fill.style.width = percent(move.value);
    track.append(fill);
    detail.append(track, make("small", "", move.legal ? "Legal" : "Illegal"));
    row.append(make("span", "prediction-name", tokenName(move.token)), detail, make("span", "prediction-value", percent(move.value)));
    return row;
  });
  element("predictions").replaceChildren(...entries);
}

function draw() {
  const live = mode === "live";
  const availableFrames = frames();
  const frame = availableFrames[position];
  element("replay-mode").setAttribute("aria-pressed", String(!live));
  element("live-mode").setAttribute("aria-pressed", String(live));
  element("live-mode").disabled = waiting;
  element("scenarios").hidden = live;
  scenarioButtons.forEach((button, index) => button.setAttribute("aria-pressed", String(replay === replays[index])));
  element("source-tag").textContent = live ? "LIVE MODEL" : "RECORDED CHECKPOINT";
  element("scenario-description").textContent = waiting
    ? "Running the local checkpoint…"
    : live
      ? "Choose a legal square to play both sides and ask the checkpoint for its next prediction."
      : replay?.description ?? "";
  element("seek").max = Math.max(0, availableFrames.length - 1);
  element("seek").value = position;
  element("seek").disabled = availableFrames.length < 2 || waiting;
  element("previous").disabled = position === 0 || waiting;
  element("forward").disabled = position >= availableFrames.length - 1 || waiting;
  element("position-label").textContent = availableFrames.length ? "POSITION " + (position + 1) + " / " + availableFrames.length : "POSITION";
  element("reset").hidden = !live;
  element("reset").disabled = waiting;
  element("special").hidden = !live || !frame?.legal.some((token) => token === PASS || token === END);
  element("special").disabled = waiting;
  element("special").textContent = frame?.legal.includes(PASS) ? "Pass turn" : "Finish game";

  if (!frame) {
    drawBoard({ board: Array(36).fill(0), history: [], legal: [] });
    return;
  }

  const filtered = element("filter").checked;
  const hasPrediction = Boolean(frame.probabilities?.length) && frame.legal.length > 0;
  const rawMass = hasPrediction
    ? Math.min(1, frame.legal.reduce((sum, token) => sum + frame.probabilities[token], 0))
    : 0;
  const moves = hasPrediction ? rankedMoves(frame, filtered, rawMass) : [];
  const top = moves[0];
  drawBoard(frame, top?.token);
  drawPredictions(moves);

  element("filter").disabled = !hasPrediction;
  element("filter-caption").textContent = filtered
    ? "Rescale the remaining legal choices."
    : "Show the model’s raw probabilities.";
  element("legal-mass").textContent = hasPrediction ? percent(rawMass) : "—";
  element("mass-fill").style.width = percent(rawMass);
  element("mass-bar").setAttribute("aria-valuenow", String(rawMass * 100));
  element("mass-bar").setAttribute("aria-valuetext", hasPrediction ? percent(rawMass) : "No prediction");
  element("history").replaceChildren(...frame.history.map((token) => make("span", "token", tokenName(token))));

  const done = !frame.legal.length || frame.legal.includes(END);
  const forcedPass = frame.legal.includes(PASS);
  const side = frame.turn === 1 ? "BLACK" : "WHITE";
  element("turn").textContent = done ? "GAME OVER" : side + (forcedPass ? " MUST PASS" : " TO MOVE");
  const predictionStatus = top
    ? (filtered ? "Top after filtering: " : "Raw top pick: ") + tokenName(top.token) + (top.legal ? " is legal." : " is illegal.")
    : "This game has ended.";
  const actual = frame.next == null ? "" : " Recorded next: " + tokenName(frame.next) + ".";
  element("prediction-status").textContent = predictionStatus + actual;
}

function showReplay(selected) {
  requestVersion += 1;
  waiting = false;
  mode = "replay";
  replay = selected;
  position = selected?.focus ?? 0;
  error(selected ? "" : "No recorded replays were found.");
  draw();
}

async function predict(history, onSuccess) {
  const localHosts = ["localhost", "127.0.0.1", "[::1]"];
  if (!localHosts.includes(location.hostname)) {
    error("Live play needs the local prediction server. See the source README for setup. The recorded replays work here.");
    return;
  }
  const version = ++requestVersion;
  waiting = true;
  error();
  draw();
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history }),
    });
    if (!response.ok) throw new Error("Prediction endpoint did not accept the request");
    const frame = await response.json();
    if (!Array.isArray(frame.board) || !Array.isArray(frame.legal) || !Array.isArray(frame.history)) {
      throw new Error("Invalid prediction response");
    }
    if (version === requestVersion) onSuccess(frame);
  } catch {
    if (version === requestVersion) error("Live predictions are unavailable. Start the local prediction server to play.");
  } finally {
    if (version === requestVersion) {
      waiting = false;
      draw();
    }
  }
}

function startLive() {
  predict([START], (frame) => {
    mode = "live";
    liveFrames = [frame];
    position = 0;
  });
}

function play(token) {
  const frame = liveFrames[position];
  if (mode !== "live" || waiting || !frame?.legal.includes(token)) return;
  predict([...frame.history, token], (nextFrame) => {
    liveFrames = [...liveFrames.slice(0, position + 1), nextFrame];
    position += 1;
  });
}

element("replay-mode").addEventListener("click", () => showReplay(replay));
element("live-mode").addEventListener("click", () => {
  if (mode !== "live") startLive();
});
element("reset").addEventListener("click", startLive);
element("special").addEventListener("click", () => {
  const frame = frames()[position];
  play(frame.legal.includes(PASS) ? PASS : END);
});
element("filter").addEventListener("change", draw);
element("seek").addEventListener("input", (event) => {
  position = Number(event.target.value);
  draw();
});
element("previous").addEventListener("click", () => {
  position -= 1;
  draw();
});
element("forward").addEventListener("click", () => {
  position += 1;
  draw();
});

showReplay(replay);
