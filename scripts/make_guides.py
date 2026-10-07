"""Build the tracing guides shown in practice mode -> checkpoints/guides.json.

The strokes come from airwrite/stroke_order.py (Zaner-Bloser manuscript order). The guide
is drawn by airpen.js: numbered start dots, arrows, and dashed pen lifts.

Usage:  python scripts/make_guides.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from airwrite.preprocess import CLASSES, resample  # noqa: E402
from airwrite.stroke_order import STROKES  # noqa: E402


def guide_strokes(strokes, total_points=160):
    """Centre the character, scale its longest side to 1, and space points evenly so the
    arrows in airpen.js (placed by point index) land a third and two thirds of the way."""
    pts = np.concatenate(strokes)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    lengths = [np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in strokes]
    out = []
    for s, length in zip(strokes, lengths):
        n = max(16, int(round(total_points * length / sum(lengths))))
        out.append(((resample(s, n) - centre) / size).round(4).tolist())
    return out


def main():
    assert set(STROKES) == set(CLASSES), set(CLASSES) ^ set(STROKES)
    guides = {c: guide_strokes(STROKES[c]) for c in CLASSES}
    out = ROOT / "checkpoints" / "guides.json"
    out.write_text(json.dumps(guides))
    print(f"{len(guides)} guides -> {out}")


if __name__ == "__main__":
    main()
