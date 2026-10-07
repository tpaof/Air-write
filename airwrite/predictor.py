"""Load the trained models and run all three on one trajectory."""

import time
from pathlib import Path

import numpy as np
import torch

from airwrite.models import CharCNN, CharGRU, KnnDtw
from airwrite.preprocess import CLASSES, to_image, to_sequence

MODEL_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
MODEL_NAMES = ("knn_dtw", "cnn", "gru", "ensemble")


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.cnn = CharCNN()
        self.cnn.load_state_dict(torch.load(model_dir / "cnn.pt", map_location="cpu"))
        self.cnn.eval()
        self.gru = CharGRU()
        self.gru.load_state_dict(torch.load(model_dir / "gru.pt", map_location="cpu"))
        self.gru.eval()
        self.knn = KnnDtw.load(model_dir / "knn_dtw.npz")

    @torch.no_grad()
    def scores(self, strokes) -> dict[str, np.ndarray]:
        """Class probabilities from each model, plus how long each took (ms)."""
        seq = to_sequence(strokes)
        img = to_image(strokes)
        out, timing = {}, {}

        t = time.perf_counter()
        out["knn_dtw"] = self.knn.scores(seq[:, :2])
        timing["knn_dtw"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        logits = self.cnn(torch.from_numpy(img)[None, None])
        out["cnn"] = torch.softmax(logits, dim=1)[0].numpy()
        timing["cnn"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        logits = self.gru(torch.from_numpy(seq)[None])
        out["gru"] = torch.softmax(logits, dim=1)[0].numpy()
        timing["gru"] = (time.perf_counter() - t) * 1000

        # Ensemble: average the three probability vectors. A model that is unsure spreads
        # its probability thin, so it pulls the vote less than a confident one.
        out["ensemble"] = (out["knn_dtw"] + out["cnn"] + out["gru"]) / 3
        timing["ensemble"] = timing["knn_dtw"] + timing["cnn"] + timing["gru"]

        return out, timing, img

    def predict(self, strokes, top: int = 3) -> dict:
        """JSON-friendly result for the web UI."""
        probs, timing, img = self.scores(strokes)
        result = {
            "models": {},
            "image": img.round(3).tolist(),
            # Full probability vectors, for word prediction on the client's request.
            "probs": {name: probs[name].round(5).tolist() for name in MODEL_NAMES},
        }
        for name in MODEL_NAMES:
            order = np.argsort(probs[name])[::-1][:top]
            result["models"][name] = {
                "top": [{"char": CLASSES[i], "prob": float(probs[name][i])} for i in order],
                "ms": round(timing[name], 1),
            }
        return result
