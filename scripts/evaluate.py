"""Evaluate the three models and write figures/tables for the report.

Usage:
    python scripts/evaluate.py                    # public test set (RTD test + 10% RTC)
    python scripts/evaluate.py --writer tpaof     # also our own webcam samples

Outputs go to reports/: results.md, results.json, confusion_*.png, learning_curves.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from airwrite.datasets import load_collected, load_public, split_collected  # noqa: E402
from airwrite.predictor import MODEL_DIR, MODEL_NAMES, Predictor  # noqa: E402
from airwrite.preprocess import CLASSES  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
PRETTY = {"knn_dtw": "KNN + DTW", "cnn": "CNN", "gru": "GRU", "ensemble": "Ensemble (เฉลี่ย 3 ตัว)"}


def run(predictor, samples):
    """Return predicted labels, top-3 hits and per-sample latency (ms) for every model."""
    y_true = np.array([s[1] for s in samples])
    pred = {m: [] for m in MODEL_NAMES}
    top3 = {m: [] for m in MODEL_NAMES}
    ms = {m: [] for m in MODEL_NAMES}
    t0 = time.time()
    for i, (strokes, label, _) in enumerate(samples):
        probs, timing, _ = predictor.scores(strokes)
        for m in MODEL_NAMES:
            order = np.argsort(probs[m])[::-1]
            pred[m].append(order[0])
            top3[m].append(label in order[:3])
            ms[m].append(timing[m])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(samples)}  ({time.time() - t0:.0f}s)")
    return y_true, {m: np.array(v) for m, v in pred.items()}, top3, ms


def summarize(name, y_true, pred, top3, ms):
    rows = {}
    for m in MODEL_NAMES:
        rows[m] = {
            "accuracy": float((pred[m] == y_true).mean()),
            "top3_accuracy": float(np.mean(top3[m])),
            "macro_f1": float(f1_score(y_true, pred[m], average="macro", labels=np.unique(y_true))),
            "latency_ms_median": float(np.median(ms[m])),
            "n": int(len(y_true)),
        }
        plot_confusion(y_true, pred[m], f"{name} — {PRETTY[m]}", REPORT_DIR / f"confusion_{name}_{m}.png")
    return rows


def plot_confusion(y_true, y_pred, title, path):
    labels = np.arange(len(CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(np.log1p(cm), cmap="Blues")
    ax.set_xticks(labels, list(CLASSES), fontsize=8)
    ax.set_yticks(labels, list(CLASSES), fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if i == j else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def top_confusions(y_true, y_pred, n=8):
    pairs = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(CLASSES[t], CLASSES[p])] = pairs.get((CLASSES[t], CLASSES[p]), 0) + 1
    return sorted(pairs.items(), key=lambda kv: -kv[1])[:n]


def plot_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind in ("cnn", "gru"):
        path = REPORT_DIR / f"history_{kind}.json"
        if not path.exists():
            continue
        h = json.loads(path.read_text())
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], label=f"{PRETTY[kind]} train")
        axes[0].plot(ep, [r["val_loss"] for r in h], "--", label=f"{PRETTY[kind]} val")
        axes[1].plot(ep, [r["train_acc"] for r in h], label=f"{PRETTY[kind]} train")
        axes[1].plot(ep, [r["val_acc"] for r in h], "--", label=f"{PRETTY[kind]} val")
    axes[0].set_title("loss")
    axes[1].set_title("accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "learning_curves.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer", help="also evaluate on webcam samples from this writer")
    parser.add_argument("--skip-public", action="store_true")
    parser.add_argument("--test-per-class", type=int, default=0,
                        help="evaluate only the writer's held-out samples (same split as train.py)")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="checkpoints to evaluate")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. 'base' or 'tuned'")
    args = parser.parse_args()
    REPORT_DIR.mkdir(exist_ok=True)

    predictor = Predictor(Path(args.model_dir))
    sets = {}
    if not args.skip_public:
        sets["public_test"] = load_public()[2]
    if args.writer:
        samples = load_collected(args.writer)
        if args.test_per_class:
            samples = split_collected(samples, args.test_per_class)[1]
        sets[f"webcam_{args.writer}{'_test' if args.test_per_class else ''}"] = samples

    results, lines = {}, ["# Evaluation results", ""]
    for name, samples in sets.items():
        print(f"evaluating {name} ({len(samples)} samples)")
        y_true, pred, top3, ms = run(predictor, samples)
        name = f"{name}_{args.tag}" if args.tag else name
        results[name] = summarize(name, y_true, pred, top3, ms)
        lines += [f"## {name} (n={len(samples)})", "",
                  "| model | accuracy | top-3 accuracy | macro-F1 | latency (median ms, CPU) |",
                  "|---|---|---|---|---|"]
        for m, r in results[name].items():
            lines.append(f"| {PRETTY[m]} | {r['accuracy']:.2%} | {r['top3_accuracy']:.2%} | "
                         f"{r['macro_f1']:.3f} | {r['latency_ms_median']:.1f} |")
        lines += ["", "Most common mistakes (true → predicted: count):", ""]
        for m in MODEL_NAMES:
            conf = ", ".join(f"{t}→{p}: {c}" for (t, p), c in top_confusions(y_true, pred[m]))
            lines.append(f"- **{PRETTY[m]}**: {conf or 'none'}")
        lines.append("")

    plot_learning_curves()
    suffix = f"_{args.tag}" if args.tag else ""
    (REPORT_DIR / f"results{suffix}.json").write_text(json.dumps(results, indent=1))
    (REPORT_DIR / f"results{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
