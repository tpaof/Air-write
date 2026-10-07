"""Draw the most recent demo attempts with what each model predicted.

Usage:  python scripts/show_log.py [--last 40]   -> reports/demo_log.png
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from airwrite.preprocess import to_image  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=40)
    args = parser.parse_args()

    files = sorted((ROOT / "data" / "demo_log").glob("*.json"))[-args.last :]
    if not files:
        print("no demo attempts logged yet")
        return
    cols = 8
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 3, rows * 1.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, path in enumerate(files):
        log = json.loads(path.read_text())
        raw_ax, img_ax = axes[k // cols][2 * (k % cols)], axes[k // cols][2 * (k % cols) + 1]
        for stroke in log["strokes"]:
            xs, ys = zip(*stroke)
            raw_ax.plot(xs, ys, "-", lw=1.5)
            raw_ax.plot(xs[0], ys[0], "go", ms=3)
        raw_ax.invert_yaxis()
        raw_ax.set_aspect("equal")
        raw_ax.set_title(f"{len(log['strokes'])} strokes", fontsize=7)
        img_ax.imshow(to_image(log["strokes"]), cmap="gray")
        guess = " ".join(f"{m[:3]}:{r['top'][0]['char']}" for m, r in log["models"].items())
        img_ax.set_title(guess, fontsize=7)
    fig.tight_layout()
    out = ROOT / "reports" / "demo_log.png"
    fig.savefig(out, dpi=90)
    print(f"{len(files)} attempts -> {out}")


if __name__ == "__main__":
    main()
