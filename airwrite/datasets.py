"""Load the public RTD/RTC air-writing datasets and our own webcam recordings.

Every loader returns a list of samples ``(strokes, label_index, writer)`` where
``strokes`` is a list of (N, 2) arrays (see preprocess.py).
"""

import json
import pickle
from pathlib import Path

import numpy as np

from airwrite.preprocess import CLASSES

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data" / "external"
COLLECTED = ROOT / "data" / "collected"


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """The datasets ship as pickles; only allow numpy arrays so loading cannot run code."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module, name):
        try:
            return self._ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}") from None


def _load_pickle(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f, encoding="latin1").load()


def _rows_to_points(row: np.ndarray, dim: int) -> np.ndarray:
    """A dataset row is a flat, zero-padded list of (x, y[, z]); keep (x, y) up to the last real point."""
    n = len(row) // dim
    pts = row[: n * dim].reshape(n, dim)
    real = np.flatnonzero(np.any(pts != 0, axis=1))
    return pts[: real[-1] + 1, :2] if len(real) else pts[:0, :2]


def _load_flat(features: Path, labels: Path, dim: int, first_class: int, source: str):
    X, Y = _load_pickle(features), _load_pickle(labels)
    samples = []
    for row, onehot in zip(X, Y):
        pts = _rows_to_points(row, dim)
        if len(pts) >= 5:
            samples.append(([pts], first_class + int(onehot.argmax()), source))
    return samples


def load_rtd():
    """RTD: digits 0-9 (Alam et al., 2020). Train rows are (x, y); test rows are (x, y, z)."""
    base = EXTERNAL / "rtd" / "RTD Dataset"
    train = _load_flat(base / "features", base / "labels", 2, 0, "rtd")
    test = _load_flat(base / "featuresTest", base / "labelsTest", 3, 0, "rtd")
    return train, test


def load_rtc():
    """RTC: uppercase A-Z, rows are (x, y, z)."""
    base = EXTERNAL / "rtc"
    return _load_flat(base / "features", base / "labels", 3, 10, "rtc")


def load_public(seed: int = 42):
    """Train/val/test split of RTD+RTC.

    RTD ships its own test set; RTC does not, so we hold out 10% + 10% of it
    stratified by class. RTD train is split 90/10 into train/val.
    """
    rng = np.random.default_rng(seed)
    rtd_train, rtd_test = load_rtd()
    rtc = load_rtc()

    def split(samples, fractions):
        by_class = {}
        for s in samples:
            by_class.setdefault(s[1], []).append(s)
        parts = [[] for _ in fractions]
        for items in by_class.values():
            order = rng.permutation(len(items))
            bounds = np.cumsum([0] + [int(round(f * len(items))) for f in fractions])
            bounds[-1] = len(items)
            for p in range(len(fractions)):
                parts[p].extend(items[i] for i in order[bounds[p] : bounds[p + 1]])
        return parts

    rtd_tr, rtd_va = split(rtd_train, [0.9, 0.1])
    rtc_tr, rtc_va, rtc_te = split(rtc, [0.8, 0.1, 0.1])
    return rtd_tr + rtc_tr, rtd_va + rtc_va, rtd_test + rtc_te


def split_collected(samples, test_per_class: int):
    """Hold out each character's last `test_per_class` recordings (in the order they were
    written) for testing; the rest may be used for training. Returns (train, test)."""
    by_class = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    train, test = [], []
    for items in by_class.values():
        cut = max(0, len(items) - test_per_class)
        train += items[:cut]
        test += items[cut:]
    return train, test


def load_collected(writer: str | None = None):
    """Samples recorded with our own webcam via the /collect page."""
    samples = []
    for path in sorted(COLLECTED.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if writer and data["writer"] != writer:
            continue
        if data["label"] not in CLASSES:
            continue
        strokes = [np.array(s, dtype=np.float64) for s in data["strokes"] if len(s) > 0]
        if strokes:
            samples.append((strokes, CLASSES.index(data["label"]), data["writer"]))
    return samples
