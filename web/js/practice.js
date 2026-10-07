import { guideLoopMs } from "/static/js/airpen.js";
import { segmentize, setupPen } from "/static/js/common.js";

// Practice mode: show a character or a picture word, the child writes it in the air one
// character at a time, and the ensemble checks each character.
const CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const SETS = { digits: "0123456789", letters: LETTERS, all: CLASSES };
// Pairs that are drawn the same way; in the mixed set either one counts as correct.
const SAME_SHAPE = { 0: "O", O: "0", 1: "I", I: "1" };
// Simple picture words for young learners: [word, picture, Thai meaning].
const WORDS = {
  words3: [
    ["CAT", "🐱", "แมว"], ["DOG", "🐶", "สุนัข"], ["PIG", "🐷", "หมู"], ["COW", "🐮", "วัว"],
    ["FOX", "🦊", "สุนัขจิ้งจอก"], ["BEE", "🐝", "ผึ้ง"], ["ANT", "🐜", "มด"], ["OWL", "🦉", "นกฮูก"],
    ["HEN", "🐔", "แม่ไก่"], ["BAT", "🦇", "ค้างคาว"], ["SUN", "☀️", "ดวงอาทิตย์"], ["BUS", "🚌", "รถบัส"],
    ["CAR", "🚗", "รถยนต์"], ["HAT", "🎩", "หมวก"], ["CUP", "☕", "ถ้วย"], ["EGG", "🥚", "ไข่"],
    ["BOX", "📦", "กล่อง"], ["KEY", "🔑", "กุญแจ"], ["BED", "🛏️", "เตียง"], ["PEN", "🖊️", "ปากกา"],
  ],
  words4: [
    ["FISH", "🐟", "ปลา"], ["BIRD", "🐦", "นก"], ["FROG", "🐸", "กบ"], ["DUCK", "🦆", "เป็ด"],
    ["BEAR", "🐻", "หมี"], ["LION", "🦁", "สิงโต"], ["CAKE", "🎂", "เค้ก"], ["BALL", "⚽", "ลูกบอล"],
    ["BOOK", "📖", "หนังสือ"], ["STAR", "⭐", "ดาว"], ["MOON", "🌙", "ดวงจันทร์"], ["TREE", "🌳", "ต้นไม้"],
    ["BOAT", "⛵", "เรือ"], ["MILK", "🥛", "นม"], ["RAIN", "🌧️", "ฝน"], ["KITE", "🪁", "ว่าว"],
  ],
};
const WORD_INFO = Object.fromEntries(Object.values(WORDS).flat().map(([w, pic, th]) => [w, { pic, th }]));

const $ = (id) => document.getElementById(id);
let pen = null;
// { words, readSet, queue, size, target, pos, wrongAt, results: [{ target, ok, wrongAt?, skipped? }] }
// `target` is one character, or a whole word written one character at a time (`pos`);
// `wrongAt` holds the positions in the word that were not right the first time.
let state = null;
let checking = false;
let guides = {}; // character -> strokes, from /api/guides
// "Look first" and the replay after a mistake show the stroke-order animation once through
// (longer for characters with more strokes), and never less than this.
const MIN_SHOW_MS = 2500;
const showMs = (strokes) => Math.max(MIN_SHOW_MS, guideLoopMs(strokes) + 300);

// always check for updated guides (an older browser copy may lack the no-cache header)
fetch("/api/guides", { cache: "no-cache" })
  .then((r) => (r.ok ? r.json() : {}))
  .then((g) => {
    guides = g;
    showGuide();
  })
  .catch(() => {});

const isWords = () => $("set").value in WORDS;
const expected = () => state?.target?.[state.pos] ?? null;

// Put the guide for the character to write now on the camera view, according to the help level.
function showGuide() {
  if (!pen) return;
  const strokes = expected() && guides[expected()];
  // Whatever the help level, give the child time to move between strokes of E, H, …
  pen.expectStrokes = strokes ? strokes.length : 0;
  const mode = $("help").value;
  if (!strokes || mode === "none") pen.guide = null;
  else pen.guide = mode === "trace" ? { strokes } : { strokes, until: performance.now() + showMs(strokes) };
}

// Draw `size` items from the pool, without repeating until the pool runs out.
function makeQueue(pool, size) {
  const queue = [];
  while (queue.length < size) {
    const bag = [...pool].sort(() => Math.random() - 0.5);
    for (const c of bag) if (queue.length < size && (pool.length === 1 || c !== queue[queue.length - 1])) queue.push(c);
  }
  return queue;
}

function startRound(pool) {
  const words = isWords();
  pool ??= words ? WORDS[$("set").value].map(([w]) => w) : SETS[$("set").value].split("");
  const size = Math.max(3, Math.min(36, Number($("round-size").value) || 10));
  // `pool` may be just the ones missed last round; characters are still read against the
  // whole set (all letters for words).
  state = {
    words, readSet: words ? LETTERS : SETS[$("set").value],
    queue: makeQueue(pool, size), size, target: null, pos: 0, wrongAt: new Set(), results: [],
  };
  document.querySelectorAll(".unit").forEach((el) => (el.textContent = words ? "คำ" : "ตัว"));
  $("target-title").textContent = words ? "เขียนคำนี้" : "เขียนตัวนี้";
  $("history").classList.toggle("words", words);
  $("summary").innerHTML = "";
  pen?.clear();
  next();
}

function next() {
  state.target = state.queue.shift() ?? null;
  state.pos = 0;
  state.wrongAt = new Set();
  showTarget();
  render();
  if (!state.target) finishRound();
}

// The target in the side panel and over the camera; in a word, letters already written
// are green and the one to write now is underlined.
function showTarget() {
  const { target, pos, words } = state;
  const overlay = $("target-overlay");
  overlay.classList.toggle("word-mode", words && !!target);
  if (!target || !words) {
    $("target").textContent = overlay.textContent = target ?? "✓";
  } else {
    const letters = [...target].map((c, i) => `<span class="${i < pos ? "done" : i === pos ? "now" : ""}">${c}</span>`).join("");
    const { pic, th } = WORD_INFO[target];
    $("target").innerHTML = `<div class="picture">${pic}</div><div class="word">${letters}</div><div class="meaning">${th}</div>`;
    overlay.innerHTML = `<span class="picture">${pic}</span><span class="word">${letters}</span>`;
  }
  showGuide();
}

// Words are scored letter by letter: one letter rewritten does not make the whole word wrong.
function render() {
  const { results, size, words } = state;
  $("progress-bar").style.width = `${(results.length / size) * 100}%`;
  if (!words) {
    $("score").innerHTML = `<b>${results.filter((r) => r.ok).length}</b> ถูก จาก <b>${results.length}</b> / ${size} ตัว`;
    $("history").innerHTML = results
      .map((r) => `<div class="${r.ok ? "done" : "wrong"}" title="${r.ok ? "ถูก" : r.skipped ? "ข้าม" : "ผิด"}">${r.target}<br>${r.ok ? "✓" : "✗"}</div>`)
      .join("");
    return;
  }
  const letters = results.reduce((n, r) => n + r.target.length, 0);
  const right = letters - results.reduce((n, r) => n + r.wrongAt.length, 0);
  $("score").innerHTML = `<b>${right}</b> ตัวถูก จาก <b>${letters}</b> ตัวอักษร · คำที่ ${results.length} / ${size}`;
  $("history").innerHTML = results
    .map((r) => {
      const word = [...r.target].map((c, i) => (r.wrongAt.includes(i) ? `<span class="bad">${c}</span>` : c)).join("");
      const mark = r.skipped ? "ข้าม" : r.ok ? "✓" : `${r.target.length - r.wrongAt.length}/${r.target.length}`;
      return `<div class="${r.ok ? "done" : r.skipped ? "wrong" : "partial"}">${word}<br>${mark}</div>`;
    })
    .join("");
}

function pushWord(skipped = false) {
  // a skipped word counts its unwritten letters as missed
  if (skipped) for (let i = state.pos; i < state.target.length; i++) state.wrongAt.add(i);
  const wrongAt = [...state.wrongAt].sort((a, b) => a - b);
  state.results.push({ target: state.target, ok: wrongAt.length === 0, wrongAt, skipped });
}

function finishRound() {
  const missed = state.results.filter((r) => !r.ok);
  const wrong = [...new Set(missed.map((r) => r.target))];
  if (state.words) {
    const letters = [...new Set(missed.flatMap((r) => r.wrongAt.map((i) => r.target[i])))].sort();
    const total = state.results.reduce((n, r) => n + r.target.length, 0);
    const right = total - state.results.reduce((n, r) => n + r.wrongAt.length, 0);
    $("status").textContent = `จบรอบ เขียนถูก ${right} จาก ${total} ตัวอักษร`;
    $("summary").innerHTML = wrong.length
      ? `<p class="hint">ตัวที่ควรฝึกเพิ่ม: <b>${letters.join(" ")}</b></p><button id="retry">ฝึกคำที่มีตัวผิดอีกครั้ง</button>`
      : `<p class="hint">ถูกทุกตัว เยี่ยมมาก 🎉</p>`;
  } else {
    $("status").textContent = `จบรอบ ได้ ${state.results.length - missed.length} จาก ${state.results.length} ตัว`;
    $("summary").innerHTML = wrong.length
      ? `<p class="hint">ตัวที่ควรฝึกเพิ่ม: <b>${wrong.join(" ")}</b></p><button id="retry">ฝึกเฉพาะตัวที่ผิด</button>`
      : `<p class="hint">ถูกทุกตัว เยี่ยมมาก 🎉</p>`;
  }
  $("retry")?.addEventListener("click", () => startRound(wrong));
}

function flash(ok, text, ms = 1200) {
  const el = $("feedback");
  el.textContent = text;
  el.className = `feedback ${ok ? "ok" : "bad"}`;
  el.hidden = false;
  clearTimeout(flash.timer);
  flash.timer = setTimeout(() => (el.hidden = true), ms);
}

// The model only chooses among the characters being practised (e.g. digits only),
// so a 0 in the digits set is never read as the letter O.
function readAs(probs) {
  let best = null;
  for (const c of state.readSet) if (best === null || probs[CLASSES.indexOf(c)] > probs[CLASSES.indexOf(best)]) best = c;
  return best;
}

async function check(strokes) {
  if (!state?.target || checking) return;
  checking = true;
  $("status").textContent = "กำลังตรวจ…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strokes }),
    });
    if (!res.ok) {
      $("status").textContent = `ตรวจไม่ได้: ${(await res.json()).detail}`;
      return;
    }
    const got = readAs((await res.json()).probs.ensemble);
    const want = expected();
    if (got === want || SAME_SHAPE[want] === got) {
      state.pos += 1;
      if (state.pos < state.target.length) {
        // a letter inside a word: go on to the next letter of the same word
        flash(true, "✓ ถูกต้อง", 900);
        $("status").textContent = `ถูกต้อง! ต่อไปเขียน ${expected()}`;
        showTarget();
        return;
      }
      if (state.words) {
        pushWord();
        const n = state.target.length;
        const right = n - state.wrongAt.size;
        flash(true, `✓ ถูกต้อง · ${state.target} ${WORD_INFO[state.target].pic}`, 1600);
        $("status").textContent = right === n ? `ถูกต้อง! เขียนคำว่า ${state.target} ได้` : `เขียนคำว่า ${state.target} ครบแล้ว (ถูก ${right} จาก ${n} ตัว)`;
      } else {
        state.results.push({ target: state.target, ok: true });
        flash(true, "✓ ถูกต้อง");
        $("status").textContent = `ถูกต้อง! เขียน ${want} ได้`;
      }
      next();
    } else {
      // Only right or wrong: what the model read it as means nothing to a child learning the letter.
      flash(false, "✗ ผิด");
      if (state.words) state.wrongAt.add(state.pos);
      $("status").textContent = state.words ? `ผิด — ดูวิธีเขียน ${want} แล้วลองใหม่` : `ผิด — ดูวิธีเขียน ${want} ที่ถูกบนจอ`;
      // Show how the missed character is written, whatever the help level.
      if (guides[want] && pen) {
        const ms = showMs(guides[want]);
        pen.guide = { strokes: guides[want], until: performance.now() + ms };
        await new Promise((r) => setTimeout(r, ms));
      }
      // A single character counts as missed and the round moves on; inside a word the
      // child writes the same letter again so the word can still be finished.
      if (state.words) showGuide();
      else {
        state.results.push({ target: want, ok: false });
        next();
      }
    }
  } finally {
    checking = false;
  }
}

segmentize($("set"));
segmentize($("help"));
$("restart").onclick = () => startRound();
$("set").onchange = () => {
  $("round-size").value = isWords() ? 5 : 10;
  startRound();
};
$("round-size").onchange = () => startRound();
$("help").onchange = showGuide;
$("skip").onclick = () => {
  if (!state.target) return;
  if (state.words) pushWord(true);
  else state.results.push({ target: state.target, ok: false, skipped: true });
  next();
};

startRound();
setupPen(check)
  .then((p) => {
    pen = p;
    showGuide();
  })
  .catch((e) => {
    $("pose").textContent = `เปิดกล้องไม่ได้: ${e.message}`;
  });
