"""Train the three models on RTD+RTC (optionally plus our own webcam samples).

Usage:
    python scripts/train.py                 # all three models
    python scripts/train.py --model cnn     # only one
    python scripts/train.py --extra tpaof   # also train on samples collected by writer "tpaof"
    python scripts/train.py --standard 100  # also train on 100 synthetic standard-order samples per character
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airwrite.datasets import load_collected, load_public, split_collected  # noqa: E402
from airwrite.models import CharCNN, CharGRU, KnnDtw  # noqa: E402
from airwrite.predictor import MODEL_DIR  # noqa: E402
from airwrite.preprocess import CLASSES, augment, join_strokes, to_image, to_sequence  # noqa: E402
from airwrite.stroke_order import synthesize_set  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


class TrajectoryDataset(Dataset):
    def __init__(self, samples, kind: str, train: bool, seed: int = 0):
        self.samples, self.kind, self.train = samples, kind, train
        self.rng = np.random.default_rng(seed)
        if not train:  # fixed inputs: compute once
            self.cache = [self._encode(s[0]) for s in samples]

    def _encode(self, strokes):
        if self.kind == "cnn":
            return torch.from_numpy(to_image(strokes))[None]
        return torch.from_numpy(to_sequence(strokes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        strokes, label, _ = self.samples[i]
        if not self.train:
            return self.cache[i], label
        return self._encode([augment(join_strokes(strokes), self.rng)]), label


def worker_init(worker_id):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(1000 + worker_id + int(time.time()))


def train_network(kind, train_set, val_set, epochs, device):
    model = (CharCNN() if kind == "cnn" else CharGRU()).to(device)
    train_loader = DataLoader(
        TrajectoryDataset(train_set, kind, train=True),
        batch_size=128, shuffle=True, num_workers=6, worker_init_fn=worker_init, persistent_workers=True,
    )
    val_loader = DataLoader(TrajectoryDataset(val_set, kind, train=False), batch_size=512)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(train_loader))
    # Digits have ~3x more samples than letters in RTD/RTC; weight classes so 0/O and 1/I
    # are not decided by which one is more common.
    counts = np.bincount([s[1] for s in train_set], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    history, best_acc, best_state = [], 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_loss += loss_fn(logits, y).item() * len(y)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += len(y)

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total, "train_acc": correct / total,
            "val_loss": v_loss / v_total, "val_acc": v_correct / v_total,
        }
        history.append(row)
        print(f"[{kind}] epoch {epoch:2d}  train acc {row['train_acc']:.4f}  val acc {row['val_acc']:.4f}")
        if row["val_acc"] > best_acc:
            best_acc = row["val_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, MODEL_DIR / f"{kind}.pt")
    (REPORT_DIR / f"history_{kind}.json").write_text(json.dumps(history, indent=1))
    print(f"[{kind}] best val acc {best_acc:.4f} -> {MODEL_DIR / f'{kind}.pt'}")


def build_knn(train_set, per_class, k, always=(), seed=0):
    """Pick `per_class` random reference samples per class, plus every sample in `always`.

    More references = more accurate but slower. New writers can be supported just by
    adding their samples to `always` — no retraining needed.
    """
    rng = np.random.default_rng(seed)
    refs = [to_sequence(s[0])[:, :2] for s in always]
    labels = [s[1] for s in always]
    for c in range(len(CLASSES)):
        idx = [i for i, s in enumerate(train_set) if s[1] == c]
        for i in rng.choice(idx, size=min(per_class, len(idx)), replace=False):
            refs.append(to_sequence(train_set[i][0])[:, :2])
            labels.append(c)
    knn = KnnDtw(np.stack(refs), np.array(labels), k=k)
    knn.save(MODEL_DIR / "knn_dtw.npz")
    print(f"[knn_dtw] {len(refs)} references (k={k}) -> {MODEL_DIR / 'knn_dtw.npz'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "knn", "cnn", "gru"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--knn-per-class", type=int, default=40)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--extra", help="also train on webcam samples from this writer")
    parser.add_argument("--extra-test-per-class", type=int, default=0,
                        help="keep this many of the writer's samples per character out of training (for evaluate.py)")
    parser.add_argument("--standard", type=int, default=0,
                        help="add this many synthetic samples per character written in the standard stroke "
                             "order taught by the practice guides (RTD/RTC writers often use other orders)")
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    train_set, val_set, _ = load_public()
    extra = []
    if args.extra:
        extra = load_collected(args.extra)
        if args.extra_test_per_class:
            extra, _ = split_collected(extra, args.extra_test_per_class)
        # Repeat our few samples so they are not drowned out by 40k public ones.
        train_set = train_set + extra * 20
        print(f"added {len(extra)} webcam samples from '{args.extra}' (x20)")
    standard = []
    if args.standard:
        standard = synthesize_set(args.standard, seed=0)
        train_set = train_set + standard
        print(f"added {len(standard)} synthetic standard-order samples")
    print(f"train {len(train_set)}  val {len(val_set)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model in ("all", "knn"):
        public_only = [s for s in train_set if s[2] in ("rtd", "rtc")]
        # a few standard-order references per character are enough for nearest-neighbour matching
        build_knn(public_only, args.knn_per_class, args.knn_k, always=extra + standard[:: max(1, args.standard // 5)])
    for kind in ("cnn", "gru"):
        if args.model in ("all", kind):
            train_network(kind, train_set, val_set, args.epochs, device)


if __name__ == "__main__":
    main()
