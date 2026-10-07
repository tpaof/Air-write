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
    const text = pen.waitForLift
      ? "✅ ส่งทายแล้ว — ปล่อยนิ้วก่อนเขียนตัวถัดไป"
      : pen.penDown
        ? POSE_LABEL.draw
        : POSE_LABEL[pen.pose] || POSE_LABEL.none;
    const hold = pen.holdProgress();
    const progress = hold && hold.pose === "palm" && hold.progress < 1 ? ` ${Math.round(hold.progress * 100)}%` : "";
    badge.textContent = `${text}${progress} · ${Math.round(pen.fps)} fps`;
  }, 100);

  const pause = document.getElementById("pause");
  pause.onchange = () => (pen.pauseMs = Number(pause.value));
  return pen;
}
