"""The three classifiers we compare.

1. KnnDtw — k-nearest neighbours, distance = Dynamic Time Warping on the
   resampled (x, y) sequence. No training beyond picking reference samples.
2. CharCNN — a small CNN on the trajectory drawn as a 28x28 image.
3. CharGRU — a bidirectional GRU on the [x, y, dx, dy] point sequence.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from airwrite.preprocess import CLASSES, IMAGE_SIZE

NUM_CLASSES = len(CLASSES)


class CharCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64))  # 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * (IMAGE_SIZE // 4) ** 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # x: (B, 1, 28, 28)
        return self.classifier(self.features(x))


class CharGRU(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, input_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, num_classes))

    def forward(self, x):  # x: (B, NUM_POINTS, 4)
        _, h = self.gru(x)
        # h: (layers * 2, B, hidden) -> last layer, forward and backward directions
        last = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(last)


def dtw_distances(query: np.ndarray, refs_t: np.ndarray, band: int = 10) -> np.ndarray:
    """DTW distance from one sequence (T, D) to M references stored as refs_t (T, D, M).

    Classic recurrence  acc[i, j] = cost(i, j) + min(acc[i-1, j-1], acc[i-1, j], acc[i, j-1]),
    computed one query row at a time and vectorised over the M references. The
    Sakoe-Chiba band limits |i - j| so the alignment stays near the diagonal.
    """
    T, M = query.shape[0], refs_t.shape[2]
    query = query.astype(np.float32)
    prev = np.full((T + 1, M), np.inf, dtype=np.float32)
    prev[0] = 0.0
    for i in range(1, T + 1):
        cur = np.full((T + 1, M), np.inf, dtype=np.float32)
        lo, hi = max(1, i - band), min(T, i + band)
        diff = refs_t[lo - 1 : hi] - query[i - 1][None, :, None]  # (W, D, M)
        cost = np.sqrt((diff * diff).sum(axis=1))  # (W, M)
        from_above = np.minimum(prev[lo - 1 : hi], prev[lo : hi + 1])  # acc[i-1, j-1], acc[i-1, j]
        for k, j in enumerate(range(lo, hi + 1)):
            cur[j] = cost[k] + np.minimum(from_above[k], cur[j - 1])
        prev = cur
    return prev[T] / T


class KnnDtw:
    def __init__(self, refs: np.ndarray, labels: np.ndarray, k: int = 5):
        self.refs = refs.astype(np.float32)  # (M, NUM_POINTS, 2)
        self.refs_t = np.ascontiguousarray(self.refs.transpose(1, 2, 0))  # (NUM_POINTS, 2, M)
        self.labels = labels.astype(np.int64)
        self.k = k

    def scores(self, seq_xy: np.ndarray) -> np.ndarray:
        """Per-class score in [0, 1]: share of the k nearest neighbours voting for it."""
        dist = dtw_distances(seq_xy, self.refs_t)
        nearest = np.argsort(dist)[: self.k]
        votes = np.bincount(self.labels[nearest], minlength=NUM_CLASSES).astype(np.float64)
        # Break ties toward the class of the single closest neighbour.
        votes[self.labels[nearest[0]]] += 1e-3
        return votes / votes.sum()

    def save(self, path: Path):
        np.savez_compressed(path, refs=self.refs, labels=self.labels, k=self.k)

    @classmethod
    def load(cls, path: Path):
        d = np.load(path)
        return cls(d["refs"], d["labels"], int(d["k"]))
