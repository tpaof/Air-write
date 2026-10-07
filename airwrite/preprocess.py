"""Turn a raw fingertip trajectory into the inputs each model expects.

A trajectory is a list of strokes; each stroke is a list of (x, y) points in
pixel coordinates (y grows downward, x as seen by the writer — i.e. already
un-mirrored). The public datasets (RTD/RTC) are written as one continuous
stroke, so we join our strokes end-to-end before anything else.
"""

import numpy as np

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_POINTS = 64  # length of the resampled sequence fed to GRU / KNN-DTW
IMAGE_SIZE = 28  # side of the image fed to the CNN


def join_strokes(strokes) -> np.ndarray:
    """Concatenate strokes into one (N, 2) polyline.

    The jump between strokes becomes a straight segment, which mimics the
    continuous single-stroke style of the RTD/RTC datasets.
    """
    parts = [np.asarray(s, dtype=np.float64)[:, :2] for s in strokes if len(s) > 0]
    if not parts:
        return np.zeros((0, 2))
    return np.concatenate(parts, axis=0)


def resample(points: np.ndarray, n: int = NUM_POINTS) -> np.ndarray:
    """Resample a polyline to n points spaced equally along its length.

    Removes the effect of writing speed and camera frame rate.
    """
    if len(points) == 0:
        return np.zeros((n, 2))
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] < 1e-9:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, dist[-1], n)
    return np.stack([np.interp(targets, dist, points[:, k]) for k in range(2)], axis=1)


def smooth(points: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing that keeps the first and last points fixed."""
    if len(points) < window:
        return points
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)], axis=1)
    out[0], out[-1] = points[0], points[-1]
    return out


def normalize(points: np.ndarray) -> np.ndarray:
    """Center at the bounding-box middle and scale the longer side to 1.

    Keeps the aspect ratio, so a tall '1' stays tall and does not turn into a blob.
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = max(float((hi - lo).max()), 1e-6)
    return (points - (lo + hi) / 2) / size


def to_sequence(strokes) -> np.ndarray:
    """Full pipeline for sequence models: (NUM_POINTS, 4) of [x, y, dx, dy]."""
    pts = normalize(smooth(resample(join_strokes(strokes))))
    deltas = np.diff(pts, axis=0, prepend=pts[:1])
    return np.concatenate([pts, deltas], axis=1).astype(np.float32)


def to_image(strokes, size: int = IMAGE_SIZE) -> np.ndarray:
    """Draw the trajectory as a (size, size) grayscale image in [0, 1] for the CNN."""
    pts = normalize(resample(join_strokes(strokes), n=256))
    margin = 3
    px = (pts + 0.5) * (size - 1 - 2 * margin) + margin
    img = np.zeros((size, size), dtype=np.float32)
    cols = np.clip(np.round(px[:, 0]).astype(int), 0, size - 1)
    rows = np.clip(np.round(px[:, 1]).astype(int), 0, size - 1)
    img[rows, cols] = 1.0
    # Thicken the 1-pixel line with a 3x3 max filter so small shifts still overlap.
    padded = np.pad(img, 1)
    thick = np.max(
        [padded[r : r + size, c : c + size] for r in range(3) for c in range(3)], axis=0
    )
    return thick


def augment(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random geometric distortions so the models tolerate different writers.

    Applied to the raw polyline during training only.
    """
    pts = points - points.mean(axis=0)
    angle = np.deg2rad(rng.uniform(-15, 15))
    shear = rng.uniform(-0.25, 0.25)
    sx, sy = rng.uniform(0.8, 1.2, size=2)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mat = rot @ np.array([[sx, shear], [0.0, sy]])
    pts = pts @ mat.T
    scale = np.ptp(pts, axis=0).max()
    pts = pts + rng.normal(0, 0.01 * scale, size=pts.shape)
    # Drop a few points at the ends — people start/stop the pen slightly early or late.
    trim = len(pts) // 20
    if trim > 0:
        pts = pts[rng.integers(0, trim + 1) : len(pts) - rng.integers(0, trim + 1)]
    return pts
