import { setupPen } from "/static/js/common.js";

const CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const writerInput = document.getElementById("writer");
const repsInput = document.getElementById("reps");
const targetEl = document.getElementById("target");
const overlayEl = document.getElementById("target-overlay");
const statusEl = document.getElementById("status");
const gridEl = document.getElementById("grid");
const totalEl = document.getElementById("total");
let counts = Object.fromEntries(CLASSES.map((c) => [c, 0]));
let target = null;
let pen = null;

try {
  writerInput.value = localStorage.getItem("airwrite.writer") || "";
} catch {}

const writer = () => writerInput.value.trim();

function renderGrid() {
  const reps = Number(repsInput.value);
  gridEl.innerHTML = CLASSES.map((c) => {
    const cls = [counts[c] >= reps ? "done" : "", c === target ? "current" : ""].join(" ");
    return `<div class="${cls}">${c}<br>${counts[c]}</div>`;
  }).join("");
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  totalEl.textContent = `(${total} / ${reps * CLASSES.length})`;
}

// Pick the least-written character next (random among ties) so a session covers every class evenly.
function nextTarget(exclude) {
  const reps = Number(repsInput.value);
  const pool = CLASSES.filter((c) => counts[c] < reps && c !== exclude);
  if (pool.length === 0) {
    target = null;
    targetEl.textContent = overlayEl.textContent = "✓";
    statusEl.textContent = "ครบแล้ว ขอบคุณครับ 🎉";
  } else {
    const min = Math.min(...pool.map((c) => counts[c]));
    const least = pool.filter((c) => counts[c] === min);
    target = least[Math.floor(Math.random() * least.length)];
    targetEl.textContent = overlayEl.textContent = target;
  }
  renderGrid();
}

async function loadStats() {
  if (!writer()) return;
  const res = await fetch(`/api/samples/${encodeURIComponent(writer())}/stats`);
  if (res.ok) counts = await res.json();
  nextTarget();
}

async function save(strokes) {
  if (!target || !writer()) {
    statusEl.textContent = "กรอกชื่อผู้เขียนก่อนเริ่ม";
    return;
  }
  const res = await fetch("/api/samples", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ writer: writer(), label: target, strokes }),
  });
  if (!res.ok) {
    statusEl.textContent = `บันทึกไม่ได้: ${(await res.json()).detail}`;
    return;
  }
  counts[target]++;
  statusEl.textContent = `บันทึก "${target}" แล้ว`;
  nextTarget(target);
}

writerInput.onchange = () => {
  try {
    localStorage.setItem("airwrite.writer", writer());
  } catch {}
  loadStats();
};
repsInput.onchange = () => nextTarget();
document.getElementById("skip").onclick = () => nextTarget(target);
async function undoLast() {
  const res = await fetch(`/api/samples/${encodeURIComponent(writer())}/undo`, { method: "POST" });
  if (!res.ok) return;
  const { removed } = await res.json();
  const label = removed.split("_")[0];
  counts[label]--;
  target = label;
  targetEl.textContent = overlayEl.textContent = label;
  statusEl.textContent = `ลบ "${label}" ล่าสุดแล้ว เขียนใหม่ได้เลย`;
  renderGrid();
}
document.getElementById("undo").onclick = undoLast;
document.getElementById("toggle").onclick = (e) => {
  pen.enabled = !pen.enabled;
  pen.clear();
  e.target.textContent = pen.enabled ? "หยุดพัก" : "เขียนต่อ";
  statusEl.textContent = pen.enabled ? "เขียนต่อได้" : "หยุดพักอยู่ — ระบบไม่บันทึก";
};

renderGrid();
loadStats();
// ✋ on an empty board removes the last saved sample — once per hold, so holding it cannot wipe several.
setupPen(save, (name, count) => name === "delete" && count === 1 && undoLast())
  .then((p) => (pen = p))
  .catch((e) => {
    document.getElementById("pose").textContent = `เปิดกล้องไม่ได้: ${e.message}`;
  });
