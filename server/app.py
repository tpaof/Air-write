"""Web server: serves the demo/collect pages and runs the models.

Run:  uvicorn server.app:app --port 8001
"""

import json
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from airwrite.datasets import COLLECTED  # noqa: E402
from airwrite.lexicon import WordPredictor  # noqa: E402
from airwrite.predictor import Predictor  # noqa: E402
from airwrite.preprocess import CLASSES  # noqa: E402

WEB = ROOT / "web"
DEMO_LOG = ROOT / "data" / "demo_log"
app = FastAPI(title="AirWrite")
app.mount("/static", StaticFiles(directory=WEB), name="static")

try:
    predictor = Predictor()
except FileNotFoundError:
    predictor = None  # models not trained yet; /collect still works
words = WordPredictor()


class Trajectory(BaseModel):
    # strokes -> points -> [x, y] in pixels, as seen by the writer
    strokes: list[list[list[float]]] = Field(min_length=1)


class WordSoFar(BaseModel):
    # one 36-way probability vector per character written so far (empty = predict next word)
    probs: list[list[float]] = Field(default_factory=list, max_length=40)
    prev: str | None = None  # the word before, for the bigram language model


class Sample(Trajectory):
    writer: str
    label: str


def _writer_dir(writer: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", writer):
        raise HTTPException(400, "writer name: letters, digits, _ or - only")
    return COLLECTED / writer


@app.get("/")
def demo_page():
    return FileResponse(WEB / "index.html")


@app.get("/collect")
def collect_page():
    return FileResponse(WEB / "collect.html")


@app.post("/api/predict")
def predict(traj: Trajectory):
    if predictor is None:
        raise HTTPException(503, "models not trained yet — run scripts/train.py")
    if sum(len(s) for s in traj.strokes) < 5:
        raise HTTPException(400, "trajectory too short")
    result = predictor.predict(traj.strokes)
    # Keep every demo attempt so mistakes can be inspected later (scripts/show_log.py).
    DEMO_LOG.mkdir(parents=True, exist_ok=True)
    log = {"strokes": traj.strokes, "models": result["models"]}
    (DEMO_LOG / f"{int(time.time() * 1000)}.json").write_text(json.dumps(log), encoding="utf-8")
    return result


@app.post("/api/suggest")
def suggest(word: WordSoFar):
    if any(len(p) != len(CLASSES) for p in word.probs):
        raise HTTPException(400, f"each probability vector must have {len(CLASSES)} values")
    return {"suggestions": words.suggest(word.probs, word.prev)}


@app.post("/api/samples")
def save_sample(sample: Sample):
    if sample.label not in CLASSES:
        raise HTTPException(400, f"label must be one of {CLASSES}")
    folder = _writer_dir(sample.writer)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sample.label}_{int(time.time() * 1000)}.json"
    path.write_text(json.dumps(sample.model_dump()), encoding="utf-8")
    return {"saved": path.name}


@app.post("/api/samples/{writer}/undo")
def undo_last_sample(writer: str):
    files = sorted(_writer_dir(writer).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise HTTPException(404, "nothing to undo")
    files[-1].unlink()
    return {"removed": files[-1].name}


@app.get("/api/samples/{writer}/stats")
def sample_stats(writer: str):
    counts = {c: 0 for c in CLASSES}
    for path in _writer_dir(writer).glob("*.json"):
        label = path.name.split("_")[0]
        if label in counts:
            counts[label] += 1
    return counts
