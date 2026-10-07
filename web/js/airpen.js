// Webcam + MediaPipe hand tracking -> fingertip strokes.
//
// Gestures:  🤏 thumb + index touching = pen down, release = pen up
//            ✋ open palm held             = cancel the half-written character, or, when the
//                                            board is empty, onGesture("delete") (repeats while held)
//            🤏 on a word chip (top strip)  = onGesture("select", 1 | 2 | 3) — pick a word suggestion
// The pen point is the midpoint of the two fingertips, which barely moves when you pinch
// or release, so strokes do not get hooks at the start and end.
// A character is sent to onCharDone(strokes) when the pen stays up for `pauseMs`, or when
// the pen is held still for `pauseMs`. Strokes are in pixels as seen by the writer
// (mirror-corrected).

import { FilesetResolver, HandLandmarker } from "/static/vendor/mediapipe/vision_bundle.mjs";

// One Euro filter (Casiez et al., CHI 2012): smooths jitter when the finger is slow,
// stays responsive when it is fast.
class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.02, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff, prev: null, dPrev: 0, tPrev: 0 });
  }
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(x, t) {
    if (this.prev === null) {
      this.prev = x;
      this.tPrev = t;
      return x;
    }
    const dt = Math.max(t - this.tPrev, 1e-3);
    const dx = (x - this.prev) / dt;
    const aD = OneEuro.alpha(this.dCutoff, dt);
    this.dPrev = aD * dx + (1 - aD) * this.dPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dPrev);
    const a = OneEuro.alpha(cutoff, dt);
    this.prev = a * x + (1 - a) * this.prev;
    this.tPrev = t;
    return this.prev;
  }
  reset() {
    this.prev = null;
    this.dPrev = 0;
  }
}

const HAND_LINKS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

// 3D distance: MediaPipe z is on roughly the same scale as x, and using it keeps a finger
// that points toward the camera from looking "short" (folded).
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

// A finger is extended when it is nearly straight (tip-to-knuckle distance close to the
// sum of its bone lengths) and its tip is farther from the wrist than its middle joint.
function fingersUp(lm) {
  const up = (mcp, pip, dip, tip) => {
    const bones = dist(lm[mcp], lm[pip]) + dist(lm[pip], lm[dip]) + dist(lm[dip], lm[tip]);
    const straight = dist(lm[mcp], lm[tip]) / bones;
    return straight > 0.8 && dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]);
  };
  return { index: up(5, 6, 7, 8), middle: up(9, 10, 11, 12), ring: up(13, 14, 15, 16), pinky: up(17, 18, 19, 20) };
}

// Thumb-tip to index-tip gap divided by palm length, so it does not depend on how far
// the hand is from the camera.
function pinchRatio(lm) {
  return dist(lm[4], lm[8]) / dist(lm[0], lm[9]);
}

// Wide gap between the two thresholds (hysteresis): fingers drift apart a little while
// moving, and that must not lift the pen in the middle of a stroke.
const PINCH_CLOSE = 0.3; // below this the pen goes down
const PINCH_OPEN = 0.55; // above this the pen goes up

const DELETE_MS = 700; // hold an open palm this long to cancel / delete one character…
const DELETE_REPEAT_MS = 500; // …then one more every this often while still held
const GUIDE_SIZE = 0.7; // dashed guide box, as a fraction of the frame height
const MIN_CHAR_SIZE = 0.15; // smaller characters are ignored (accidental moves, too jittery)

// 5-point moving average, only for drawing on screen (the server smooths its own copy).
function smoothForDisplay(points) {
  return points.map((_, i) => {
    const win = points.slice(Math.max(0, i - 2), i + 3);
    return [win.reduce((a, p) => a + p[0], 0) / win.length, win.reduce((a, p) => a + p[1], 0) / win.length];
  });
}

export class AirPen {
  constructor({ video, canvas, onCharDone, onStatus, onGesture, pauseMs = 800 }) {
    Object.assign(this, { video, canvas, onCharDone, onStatus, onGesture, pauseMs });
    this.choices = []; // word suggestions drawn as chips across the top; pinch one to pick it
    this.hold = null; // the held palm gesture, see checkHold()
    this.ctx = canvas.getContext("2d");
    this.strokes = [];
    this.current = null;
    this.ghost = null; // last submitted character, drawn faded for a moment
    this.pose = "none";
    this.penDown = false;
    this.lastPenDown = 0;
    this.lastDrawSeen = 0;
    this.drawFrames = 0;
    this.upFrames = 0;
    this.fx = new OneEuro();
    this.fy = new OneEuro();
    this.enabled = true;
    this.fps = 0;
  }

  async start(deviceId) {
    if (!this.landmarker) {
      const fileset = await FilesetResolver.forVisionTasks("/static/vendor/mediapipe/wasm");
      const options = (delegate) => ({
        baseOptions: { modelAssetPath: "/static/vendor/hand_landmarker.task", delegate },
        runningMode: "VIDEO",
        numHands: 1,
      });
      this.landmarker = await HandLandmarker.createFromOptions(fileset, options("GPU")).catch(() =>
        HandLandmarker.createFromOptions(fileset, options("CPU")),
      );
    }
    await this.openCamera(deviceId);
    if (!this.running) {
      this.running = true;
      requestAnimationFrame(() => this.loop());
    }
  }

  async openCamera(deviceId) {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    // 640x480 is plenty for hand tracking, and laptop webcams usually only reach 30 fps
    // at this size — more frames per second means smoother, less jagged strokes.
    const video = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
    if (deviceId) video.deviceId = { exact: deviceId };
    this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
  }

  static async listCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === "videoinput");
  }

  clear() {
    this.strokes = [];
    this.current = null;
  }

  // What the hand is doing this frame: "draw" (pinched), "hover" (pen up), "palm" or "none".
  classify(lm) {
    const f = fingersUp(lm);
    const ratio = pinchRatio(lm);
    if (ratio < (this.penDown ? PINCH_OPEN : PINCH_CLOSE)) return "draw";
    if (f.index && f.middle && f.ring && f.pinky && ratio > PINCH_OPEN) return "palm";
    return "hover";
  }

  // The pen is the centre of the pinch: thumb tip/joint and index tip/joint averaged.
  // Pinched fingertips overlap and their landmarks jitter; averaging four points steadies it.
  penPoint(lm) {
    const pts = [lm[3], lm[4], lm[7], lm[8]];
    const x = pts.reduce((a, p) => a + p.x, 0) / pts.length;
    const y = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    // MediaPipe gives camera-image coordinates; flip x so it matches what the writer sees.
    return [(1 - x) * this.canvas.width, y * this.canvas.height];
  }

  loop() {
    const now = performance.now();
    // The screen refreshes faster than the camera; only run the hand model on new frames.
    if (this.video.readyState >= 2 && this.video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = this.video.currentTime;
      if (this.lastFrameAt) this.fps = 0.9 * this.fps + 0.1 * (1000 / (now - this.lastFrameAt));
      this.lastFrameAt = now;

      const res = this.landmarker.detectForVideo(this.video, now);
      this.landmarks = res.landmarks && res.landmarks[0];
      const raw = this.landmarks ? this.classify(this.landmarks) : "none";
      this.pose = raw;
      this.tip = null;
      if (this.landmarks) {
        const [x, y] = this.penPoint(this.landmarks);
        this.tip = [this.fx.filter(x, now / 1000), this.fy.filter(y, now / 1000)];
      } else {
        this.fx.reset();
        this.fy.reset();
      }
      if (this.enabled) this.updateStrokes(raw, this.tip, now);
    }
    this.draw();
    requestAnimationFrame(() => this.loop());
  }

  // Pen state with hysteresis so the line does not break when the pose flickers:
  //  - pen goes down after 2 frames of "draw"
  //  - pen goes up after 3 frames of "hover"/"palm", or ~300 ms without "draw" (hand lost)
  updatePenState(raw, now) {
    if (raw === "draw") {
      this.drawFrames++;
      this.upFrames = 0;
      this.lastDrawSeen = now;
    } else {
      this.drawFrames = 0;
      if (raw !== "none") this.upFrames++; // hover or palm: clearly not writing
    }
    if (!this.penDown && this.drawFrames >= 2) this.penDown = true;
    else if (this.penDown && (this.upFrames >= 3 || now - this.lastDrawSeen > 300)) this.penDown = false;
    return this.penDown;
  }

  updateStrokes(raw, tip, now) {
    const penDown = this.updatePenState(raw, now);
    this.checkHold(raw, now);

    // After a character was finished by holding still, wait for the pen to lift;
    // otherwise the move to the next character's start would be drawn too.
    if (this.waitForLift) {
      if (penDown) return;
      this.waitForLift = false;
    }

    if (penDown) {
      // Pinching on a word chip with nothing half-written is a click, not the start of a stroke.
      if (!this.current && this.strokes.length === 0) {
        const i = this.chipAt(tip);
        if (i >= 0) {
          this.waitForLift = true;
          this.onGesture?.("select", i + 1);
          return;
        }
      }
      if (!this.current) {
        const prev = this.strokes[this.strokes.length - 1];
        const [px, py] = prev ? prev[prev.length - 1] : [0, 0];
        if (prev && tip && now - this.lastPenUpAt < 250 && Math.hypot(tip[0] - px, tip[1] - py) < 0.08 * this.canvas.height) {
          // The pen only lifted for a moment near where it stopped: a flicker, not a new stroke.
          this.current = prev;
        } else {
          this.current = [];
          this.strokes.push(this.current);
        }
        this.lastSureLength = this.current.length;
      }
      if (tip) this.current.push([tip[0], tip[1], now]);
      if (raw === "draw") this.lastSureLength = this.current.length;
      this.lastPenDown = now;
      const still = this.stillSince(this.current);
      if (still !== null && now - this.current[still][2] > this.pauseMs) {
        // Pen held still: the character is done. Keep the stroke up to where it stopped.
        this.current.length = still + 1;
        this.current = null;
        this.waitForLift = true;
        this.finish(now);
      }
      return;
    }

    // Pen is up: drop points recorded while lifting.
    if (this.current) {
      this.current.length = this.lastSureLength;
      if (this.current.length < 3) this.strokes.pop(); // ignore accidental taps
      this.lastPenUpAt = now;
    }
    this.current = null;
    // Holding a palm over a half-written character means "cancel it", so do not send it yet.
    if (raw !== "palm" && now - this.lastPenDown > this.pauseMs) this.finish(now);
  }

  // ✋ palm held for DELETE_MS (pen up): if a character is half-written, cancel it (and stop
  // for this hold); otherwise delete one character, then another every DELETE_REPEAT_MS.
  checkHold(raw, now) {
    if (raw !== "palm" || this.penDown) {
      this.hold = null;
      return;
    }
    if (!this.hold) this.hold = { pose: raw, since: now, fired: 0, nextAt: 0 };
    const h = this.hold;
    if (h.cancelled || now - h.since < DELETE_MS || now < h.nextAt) return;
    if (this.strokes.length > 0) {
      this.clear();
      h.cancelled = true; // holding on does not go on to delete typed text
      this.onStatus?.("✋ ยกเลิกตัวที่กำลังเขียนแล้ว");
      return;
    }
    h.fired++;
    h.nextAt = now + DELETE_REPEAT_MS;
    this.onGesture?.("delete", h.fired); // h.fired = 1 for the first delete of this hold
  }

  // Word chips across the top of the frame, above the guide box (the hand stays fully in view
  // when reaching up, unlike at the bottom edge where the wrist leaves the frame).
  chipRects() {
    const { width: w, height: h } = this.canvas;
    const gap = 0.015 * w;
    const cw = (w - 4 * gap) / 3;
    return this.choices.map((_, i) => ({ x: gap + i * (cw + gap), y: 0.02 * h, w: cw, h: 0.11 * h }));
  }

  chipAt(tip) {
    if (!tip) return -1;
    return this.chipRects().findIndex((r) => tip[0] >= r.x && tip[0] <= r.x + r.w && tip[1] >= r.y && tip[1] <= r.y + r.h);
  }

  // Progress (0..1) of the palm being held, for the UI: { pose, progress } or null.
  holdProgress() {
    const h = this.hold;
    if (!h || h.cancelled) return null;
    return { pose: "palm", progress: h.fired ? 1 : Math.min(1, (performance.now() - h.since) / DELETE_MS) };
  }

  // Index of the first point of the trailing run where the pen has barely moved, or null.
  stillSince(points) {
    if (points.length < 10) return null;
    const radius = 0.025 * this.canvas.height;
    const [lx, ly] = points[points.length - 1];
    let k = points.length - 1;
    while (k > 0 && Math.hypot(points[k - 1][0] - lx, points[k - 1][1] - ly) < radius) k--;
    return k < points.length - 1 ? k : null;
  }

  finish(now) {
    const strokes = this.strokes.filter((s) => s.length >= 3);
    if (strokes.length === 0) return;
    const xs = strokes.flat().map((p) => p[0]);
    const ys = strokes.flat().map((p) => p[1]);
    const size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    this.ghost = { strokes: this.strokes, until: now + 1200 };
    this.clear();
    if (size < MIN_CHAR_SIZE * this.canvas.height) {
      this.onStatus?.("เส้นสั้นเกินไป ไม่ส่งทาย — เขียนให้ตัวใหญ่ขึ้น");
      return;
    }
    this.onCharDone?.(strokes.map((s) => s.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10])));
  }

  draw() {
    const { ctx, canvas, landmarks, tip } = this;
    const w = canvas.width;
    const h = canvas.height;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(this.video, 0, 0, w, h);
    ctx.restore();

    if (landmarks) {
      ctx.strokeStyle = "rgba(255,255,255,0.45)";
      ctx.lineWidth = Math.max(1, w / 400);
      for (const [a, b] of HAND_LINKS) {
        ctx.beginPath();
        ctx.moveTo((1 - landmarks[a].x) * w, landmarks[a].y * h);
        ctx.lineTo((1 - landmarks[b].x) * w, landmarks[b].y * h);
        ctx.stroke();
      }
    }

    // Guide box: characters written this big track far more accurately than small ones.
    const box = GUIDE_SIZE * h;
    ctx.save();
    ctx.setLineDash([w / 80, w / 80]);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = Math.max(1, w / 320);
    ctx.strokeRect((w - box) / 2, (h - box) / 2, box, box);
    ctx.restore();

    this.drawChips(tip);

    const now = performance.now();
    if (this.ghost && now < this.ghost.until) this.drawStrokes(this.ghost.strokes, "rgba(56,189,248,0.25)");
    this.drawStrokes(this.strokes, "#38bdf8");

    if (tip) {
      ctx.fillStyle = this.penDown ? "#22c55e" : this.pose === "palm" ? "#f87171" : "#facc15";
      ctx.beginPath();
      ctx.arc(tip[0], tip[1], w / 90, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawChips(tip) {
    const { ctx } = this;
    const hover = this.strokes.length === 0 ? this.chipAt(tip) : -1;
    this.chipRects().forEach((r, i) => {
      ctx.fillStyle = "rgba(11, 18, 32, 0.78)";
      ctx.strokeStyle = i === hover ? (this.penDown ? "#22c55e" : "#38bdf8") : "rgba(255,255,255,0.25)";
      ctx.lineWidth = i === hover ? r.h * 0.07 : 1;
      ctx.beginPath();
      ctx.roundRect(r.x, r.y, r.w, r.h, r.h * 0.2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#e5ecf6";
      ctx.font = `700 ${Math.round(r.h * 0.5)}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(this.choices[i], r.x + r.w / 2, r.y + r.h / 2, r.w * 0.9);
    });
  }

  // Draw through the midpoints with quadratic curves so the line looks smooth on screen.
  drawStrokes(strokes, color) {
    const { ctx } = this;
    ctx.strokeStyle = color;
    ctx.lineWidth = this.canvas.width / 110;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const raw of strokes) {
      if (raw.length < 2) continue;
      const s = smoothForDisplay(raw);
      ctx.beginPath();
      ctx.moveTo(s[0][0], s[0][1]);
      for (let i = 1; i < s.length - 1; i++) {
        const mx = (s[i][0] + s[i + 1][0]) / 2;
        const my = (s[i][1] + s[i + 1][1]) / 2;
        ctx.quadraticCurveTo(s[i][0], s[i][1], mx, my);
      }
      ctx.lineTo(s[s.length - 1][0], s[s.length - 1][1]);
      ctx.stroke();
    }
  }
}

export const POSE_LABEL = {
  draw: "🤏 กำลังเขียน",
  hover: "ปล่อยนิ้ว = ยกปากกา",
  palm: "✋ ค้างไว้ = ยกเลิกตัวที่เขียน / ลบตัวอักษร",
  none: "ไม่พบมือ",
};
