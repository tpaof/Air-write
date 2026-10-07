"""Standard stroke order for 0-9 and A-Z, after the Zaner-Bloser manuscript stroke
descriptions (e.g. E = pull down, lift, slide right, lift, slide right & stop short, lift,
slide right). Numerals follow the same style.

Used for the tracing guides in practice mode (scripts/make_guides.py) and to synthesise
training samples written in this order (scripts/train.py --standard), so a child who
follows the guide is read correctly.

Coordinates: x right, y down, the character spans y -0.5 (top) .. 0.5 (bottom), midline 0.
Arc angles are in degrees in that y-down space: -90 top, 0 right, 90 bottom, 180 left;
a decreasing angle runs counter-clockwise on screen ("circle back"), increasing runs
clockwise ("circle forward").
"""

import numpy as np

from airwrite.preprocess import CLASSES

TOP, MID, BOT = -0.5, 0.0, 0.5


def line(*points):
    return np.array(points, dtype=float)


def arc(cx, cy, rx, ry, a0, a1):
    a = np.radians(np.linspace(a0, a1, max(8, int(abs(a1 - a0) / 3))))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)


def path(*parts):
    """Join pieces drawn without lifting the pen into one stroke."""
    return np.concatenate(parts)


def circle_back(cx, cy, rx, ry, start=-90.0, sweep=360.0):
    return arc(cx, cy, rx, ry, start, start - sweep)


STROKES = {
    # numerals
    "0": [circle_back(0, MID, 0.32, 0.5)],
    "1": [line((0, TOP), (0, BOT))],
    "2": [path(arc(0, -0.22, 0.3, 0.28, -165, 35), line((0.23, -0.04), (-0.32, BOT), (0.32, BOT)))],
    "3": [path(arc(0, -0.25, 0.28, 0.25, -155, 90), arc(0, 0.25, 0.32, 0.25, -90, 150))],
    # closed 4, as printed in children's books: slant left and slide right, lift, pull down
    "4": [line((0.15, TOP), (-0.32, 0.18), (0.32, 0.18)), line((0.15, TOP), (0.15, BOT))],
    "5": [path(line((-0.17, TOP), (-0.17, -0.05)), arc(0, 0.2, 0.3, 0.3, -125, 150)), line((-0.17, TOP), (0.28, TOP))],
    "6": [path(arc(0.22, 0.22, 0.5, 0.67, -90, -180), circle_back(0, 0.22, 0.28, 0.28, start=180))],
    "7": [line((-0.3, TOP), (0.3, TOP), (-0.1, BOT))],
    "8": [path(arc(0, -0.27, 0.25, 0.23, -30, -270), arc(0, 0.23, 0.3, 0.27, -90, 200), line((-0.28, 0.13), (0.22, -0.39)))],
    "9": [path(circle_back(0, -0.22, 0.28, 0.28, start=0), line((0.28, -0.22), (0.28, BOT)))],
    # capitals
    "A": [line((0, TOP), (-0.4, BOT)), line((0, TOP), (0.4, BOT)), line((-0.26, 0.15), (0.26, 0.15))],
    "B": [
        line((-0.3, TOP), (-0.3, BOT)),
        path(
            line((-0.3, TOP), (0.05, TOP)), arc(0.05, -0.25, 0.25, 0.25, -90, 90), line((0.05, MID), (-0.3, MID)),
            line((-0.3, MID), (0.08, MID)), arc(0.08, 0.25, 0.27, 0.25, -90, 90), line((0.08, BOT), (-0.3, BOT)),
        ),
    ],
    "C": [arc(0.05, MID, 0.45, 0.5, -40, -320)],
    "D": [line((-0.35, TOP), (-0.35, BOT)), path(line((-0.35, TOP), (-0.05, TOP)), arc(-0.05, MID, 0.4, 0.5, -90, 90), line((-0.05, BOT), (-0.35, BOT)))],
    "E": [line((-0.3, TOP), (-0.3, BOT)), line((-0.3, TOP), (0.3, TOP)), line((-0.3, MID), (0.18, MID)), line((-0.3, BOT), (0.3, BOT))],
    "F": [line((-0.3, TOP), (-0.3, BOT)), line((-0.3, TOP), (0.3, TOP)), line((-0.3, MID), (0.18, MID))],
    "G": [path(arc(0.05, MID, 0.45, 0.5, -40, -360), line((0.5, MID), (0.15, MID)))],
    "H": [line((-0.35, TOP), (-0.35, BOT)), line((0.35, TOP), (0.35, BOT)), line((-0.35, MID), (0.35, MID))],
    "I": [line((0, TOP), (0, BOT)), line((-0.25, TOP), (0.25, TOP)), line((-0.25, BOT), (0.25, BOT))],
    "J": [path(line((0.2, TOP), (0.2, 0.25)), arc(-0.05, 0.25, 0.25, 0.25, 0, 180)), line((-0.05, TOP), (0.45, TOP))],
    "K": [line((-0.3, TOP), (-0.3, BOT)), line((0.3, TOP), (-0.3, MID), (0.3, BOT))],
    "L": [line((-0.3, TOP), (-0.3, BOT), (0.3, BOT))],
    "M": [line((-0.4, TOP), (-0.4, BOT)), line((-0.4, TOP), (0, 0.3), (0.4, TOP), (0.4, BOT))],
    "N": [line((-0.35, TOP), (-0.35, BOT)), line((-0.35, TOP), (0.35, BOT), (0.35, TOP))],
    "O": [circle_back(0, MID, 0.45, 0.5)],
    "P": [line((-0.3, TOP), (-0.3, BOT)), path(line((-0.3, TOP), (0.05, TOP)), arc(0.05, -0.22, 0.25, 0.28, -90, 90), line((0.05, 0.06), (-0.3, 0.06)))],
    "Q": [circle_back(0, MID, 0.45, 0.5), line((0.1, 0.2), (0.45, 0.55))],
    "R": [
        line((-0.3, TOP), (-0.3, BOT)),
        path(line((-0.3, TOP), (0.05, TOP)), arc(0.05, -0.22, 0.25, 0.28, -90, 90), line((0.05, 0.06), (-0.3, 0.06), (0.35, BOT))),
    ],
    "S": [path(arc(0, -0.25, 0.38, 0.25, -30, -270), arc(0, 0.25, 0.38, 0.25, -90, 150))],
    "T": [line((0, TOP), (0, BOT)), line((-0.35, TOP), (0.35, TOP))],
    "U": [path(line((-0.35, TOP), (-0.35, 0.15)), arc(0, 0.15, 0.35, 0.35, 180, 0), line((0.35, 0.15), (0.35, TOP)))],
    "V": [line((-0.4, TOP), (0, BOT), (0.4, TOP))],
    "W": [line((-0.5, TOP), (-0.25, BOT), (0, TOP), (0.25, BOT), (0.5, TOP))],
    "X": [line((-0.35, TOP), (0.35, BOT)), line((0.35, TOP), (-0.35, BOT))],
    "Y": [line((-0.38, TOP), (0, MID)), line((0.38, TOP), (0, MID), (0, BOT))],
    "Z": [line((-0.35, TOP), (0.35, TOP), (-0.35, BOT), (0.35, BOT))],
}


def synthesize(label: int, rng: np.random.Generator):
    """One imitation of a person writing CLASSES[label] in the standard order: each stroke
    shifted and stretched a little on its own, the whole character warped gently. Returns
    (strokes, label, "std") like the dataset loaders. train.py adds its usual augmentation."""
    strokes = []
    for s in STROKES[CLASSES[label]]:
        centre = s.mean(axis=0)
        stretch = rng.uniform(0.9, 1.1, size=2)
        strokes.append((s - centre) * stretch + centre + rng.normal(0, 0.03, size=2))
    phase, freq, amp = rng.uniform(0, 2 * np.pi, 2), rng.uniform(1, 3, 2), rng.uniform(0, 0.03, 2)
    warped = []
    for s in strokes:
        s = s.copy()
        s[:, 0] += amp[0] * np.sin(freq[0] * np.pi * s[:, 1] + phase[0])
        s[:, 1] += amp[1] * np.sin(freq[1] * np.pi * s[:, 0] + phase[1])
        warped.append(s * 300)  # pixel-like scale, as recorded by the web page
    return warped, label, "std"


def synthesize_set(per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    return [synthesize(c, rng) for c in range(len(CLASSES)) for _ in range(per_class)]
