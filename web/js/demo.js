import { setupPen } from "/static/js/common.js";

const KNN_K = 5;
const $ = (id) => document.getElementById(id);
const status = $("status");
const cnnView = $("cnn-view").getContext("2d");

// Text = committed words + the word being written. Each character of the current word keeps
// what every model read, so the word can be shown with whichever model is chosen.
const state = { committed: "", word: [], chosen: "ensemble" };

const rawWord = () => state.word.map((c) => c[state.chosen]).join("");

function render() {
  $("committed").textContent = state.committed;
  $("current").textContent = rawWord();
}

function commit(word) {
  state.committed += word + " ";
  state.word = [];
  render();
}

function backspace() {
  if (state.word.length) state.word = state.word.slice(0, -1);
  else state.committed = state.committed.slice(0, -1);
  render();
}

function showImage(img) {
  const data = cnnView.createImageData(28, 28);
  img.flat().forEach((v, i) => {
    const g = Math.round(v * 255);
    data.data.set([g, g, g, 255], i * 4);
  });
  cnnView.putImageData(data, 0, 0);
}

function showModels(result) {
  for (const [name, r] of Object.entries(result.models)) {
    const card = document.querySelector(`.model[data-model="${name}"]`);
    card.querySelector(".char").textContent = r.top[0].char;
    // KNN's score is the share of the k nearest neighbours that voted, not a probability.
    const fmt = (p) => (name === "knn_dtw" ? `${Math.round(p * KNN_K)}/${KNN_K} โหวต` : `${Math.round(p * 100)}%`);
    card.querySelector(".prob").textContent = fmt(r.top[0].prob);
    card.querySelector(".others").textContent = r.top
      .slice(1)
      .filter((t) => t.prob >= 0.01)
      .map((t) => `${t.char} ${fmt(t.prob)}`)
      .join(" · ");
    card.querySelector(".ms").textContent = `${r.ms} ms`;
  }
}

async function recognize(strokes) {
  status.textContent = "กำลังทาย…";
  const t0 = performance.now();
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ strokes }),
  });
  if (!res.ok) {
    status.textContent = `ผิดพลาด: ${(await res.json()).detail}`;
    return;
  }
  const result = await res.json();
  showModels(result);
  showImage(result.image);
  const chars = Object.fromEntries(Object.entries(result.models).map(([m, r]) => [m, r.top[0].char]));
  state.word = [...state.word, chars];
  status.textContent = `${Math.round(performance.now() - t0)} ms (รวมส่งข้อมูล)`;
  render();
}

document.querySelectorAll(".model").forEach((el) => {
  el.onclick = () => {
    document.querySelectorAll(".model").forEach((m) => m.classList.remove("selected"));
    el.classList.add("selected");
    state.chosen = el.dataset.model;
    render();
  };
});
$("space").onclick = () => commit(rawWord());
$("backspace").onclick = backspace;
$("clear-text").onclick = () => {
  Object.assign(state, { committed: "", word: [] });
  render();
};

render();

const onGesture = (name) => {
  if (name === "delete") {
    backspace();
    $("pen-status").textContent = "✋ ลบแล้ว";
  }
};

setupPen(recognize, onGesture).catch((e) => {
  $("pose").textContent = `เปิดกล้องไม่ได้: ${e.message}`;
});
