import { AirPen, POSE_LABEL } from "/static/js/airpen.js";

function load(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

// Show a <select> as a row of buttons: bigger targets and every choice in view. The select
// stays the source of truth, so page code keeps using .value and its "change" event.
export function segmentize(select) {
  const bar = document.createElement("div");
  bar.className = "segmented";
  bar.setAttribute("role", "group");
  const buttons = [...select.options].map((opt) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = opt.dataset.short || opt.text;
    b.title = opt.text;
    b.onclick = () => {
      if (select.value === opt.value) return;
      select.value = opt.value;
      select.dispatchEvent(new Event("change"));
    };
    bar.append(b);
    return b;
  });
  const sync = () =>
    buttons.forEach((b, i) => {
      b.classList.toggle("active", select.options[i].selected);
      b.setAttribute("aria-pressed", String(select.options[i].selected));
    });
  select.addEventListener("change", sync);
  select.hidden = true;
  select.after(bar);
  sync();
}

// Start the pen, fill the camera picker and keep the status badge up to date.
// The chosen camera is remembered per browser.
export async function setupPen(onCharDone, onGesture) {
  const status = document.getElementById("pen-status");
  const pen = new AirPen({
    video: document.getElementById("video"),
    canvas: document.getElementById("canvas"),
    onCharDone,
    onGesture,
    onStatus: (msg) => (status.textContent = msg),
  });

  const camSelect = document.getElementById("camera");
  const saved = load("airwrite.camera");
  await pen.start(saved || undefined).catch(() => pen.start());
  const cams = await AirPen.listCameras();
  camSelect.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || "Camera " + (i + 1)}</option>`).join("");
  camSelect.value = pen.stream.getVideoTracks()[0].getSettings().deviceId;
  camSelect.onchange = async () => {
    await pen.openCamera(camSelect.value);
    save("airwrite.camera", camSelect.value);
  };

  const badge = document.getElementById("pose");
  setInterval(() => {
    const toCome = pen.strokesToCome();
    const text = pen.waitForLift
      ? "✅ ส่งทายแล้ว — ปล่อยนิ้วก่อนเขียนตัวถัดไป"
      : pen.penDown
        ? POSE_LABEL.draw
        : toCome && pen.pose !== "palm"
          ? `ไปเริ่มเส้นที่ ${pen.expectStrokes - toCome + 1} ต่อได้เลย`
          : POSE_LABEL[pen.pose] || POSE_LABEL.none;
    const hold = pen.holdProgress();
    const progress = hold && hold.pose === "palm" && hold.progress < 1 ? ` ${Math.round(hold.progress * 100)}%` : "";
    badge.textContent = `${text}${progress} · ${Math.round(pen.fps)} fps`;
  }, 100);

  const pause = document.getElementById("pause");
  pause.onchange = () => (pen.pauseMs = Number(pause.value));
  return pen;
}
