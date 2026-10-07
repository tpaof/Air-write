"""Predict the word being written (autocomplete) with Bayes' rule.

Each written character i gives a probability vector p_i(c) from the classifier. For every
dictionary word w that is at least as long as what has been written so far:

    P(w | strokes, previous word)  ∝  P(w | previous word) · Π_i p_i(w_i)
                                       └ language model ┘   └ classifier ┘

The language model is a bigram model backed off to word frequency:

    P(w | prev) = 0.7 · count(prev, w) / count(prev, ·) + 0.3 · count(w) / total

It only gets λ = 90% of the prior mass. The other 10% goes to "any string" (spread evenly
over all 36^L strings), so codes and numbers that are not words can still be typed when the
classifier is confident. Before any character is written the same formula with no
classifier term predicts the next word from the previous one.
"""

from pathlib import Path

import numpy as np

from airwrite.preprocess import CLASSES

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicon"
DICT_WEIGHT = 0.9  # λ: prior probability that the writer meant a dictionary word
BIGRAM_WEIGHT = 0.7  # share of the language model taken from the bigram when available
PROB_FLOOR = 1e-3  # never trust a classifier's 0 completely (KNN votes are often exactly 0)
MIN_SUGGESTION_PROB = 0.01  # suggestions less likely than this are not shown


class WordPredictor:
    def __init__(self, lexicon_dir: Path = LEXICON_DIR):
        counts: dict[str, float] = {}
        for line in (lexicon_dir / "words_en.tsv").read_text(encoding="utf-8").splitlines():
            word, count = line.split("\t")
            counts[word] = float(count)
        # Custom words (names, course terms) get the frequency of a fairly common word.
        common = sorted(counts.values())[-200]
        custom = lexicon_dir / "custom_words.txt"
        if custom.exists():
            for line in custom.read_text(encoding="utf-8").splitlines():
                word = line.strip().upper()
                if word and not word.startswith("#") and all(c in CLASSES for c in word):
                    counts[word] = max(counts.get(word, 0.0), common)

        self.words = list(counts)
        self.index = {w: i for i, w in enumerate(self.words)}
        freq = np.array([counts[w] for w in self.words])
        self.unigram = freq / freq.sum()
        self.lengths = np.array([len(w) for w in self.words])
        # Character matrix, padded with -1 past the end of each word.
        width = self.lengths.max()
        self.chars = np.full((len(self.words), width), -1, dtype=np.int64)
        for i, w in enumerate(self.words):
            self.chars[i, : len(w)] = [CLASSES.index(c) for c in w]

        # Bigram counts: prev word -> (next word indices, next word probabilities)
        raw: dict[str, list[tuple[int, float]]] = {}
        bigram_file = lexicon_dir / "bigrams_en.tsv"
        if bigram_file.exists():
            for line in bigram_file.read_text(encoding="utf-8").splitlines():
                prev, nxt, count = line.split("\t")
                if nxt in self.index:
                    raw.setdefault(prev, []).append((self.index[nxt], float(count)))
        self.bigram = {}
        for prev, pairs in raw.items():
            idx = np.array([p[0] for p in pairs])
            c = np.array([p[1] for p in pairs])
            self.bigram[prev] = (idx, c / c.sum())

    def language_model(self, prev: str | None) -> np.ndarray:
        """P(w | prev) for every dictionary word."""
        prev = (prev or "").upper()
        if prev not in self.bigram:
            return self.unigram
        idx, p = self.bigram[prev]
        lm = (1 - BIGRAM_WEIGHT) * self.unigram
        lm[idx] += BIGRAM_WEIGHT * p
        return lm

    def suggest(self, probs, prev: str | None = None, top: int = 3) -> list[dict]:
        """probs: (L, 36) classifier output for the characters written so far (L may be 0).
        Returns the `top` most probable words, each with its posterior probability."""
        probs = np.asarray(probs, dtype=np.float64).reshape(-1, len(CLASSES))
        length = len(probs)
        log_prior = np.log(DICT_WEIGHT * self.language_model(prev))

        # Words long enough to start with what was written; score the written prefix.
        ok = self.lengths >= length
        scores = np.full(len(self.words), -np.inf)
        if length == 0:
            scores = log_prior.copy()
            # Web text has plenty of "the the" typos; never suggest repeating the last word.
            if prev and prev.upper() in self.index:
                scores[self.index[prev.upper()]] = -np.inf
        else:
            log_p = np.log(np.maximum(probs, PROB_FLOOR))  # (L, 36)
            prefix = self.chars[ok, :length]  # (N_ok, L)
            scores[ok] = log_prior[ok] + log_p[np.arange(length), prefix].sum(axis=1)

        cands: dict[str, float] = {}
        for k in np.argsort(scores)[::-1][: top * 6]:
            if np.isfinite(scores[k]):
                cands[self.words[k]] = scores[k]
        log_evidence = np.logaddexp.reduce(scores[np.isfinite(scores)]) if np.isfinite(scores).any() else -np.inf

        # "Any string" hypothesis: exactly the characters the classifier liked best.
        if length > 0:
            best = log_p.argmax(axis=1)
            raw = "".join(CLASSES[i] for i in best)
            raw_score = np.log(1 - DICT_WEIGHT) - length * np.log(len(CLASSES)) + log_p[np.arange(length), best].sum()
            cands[raw] = np.logaddexp(cands.get(raw, -np.inf), raw_score)
            log_evidence = np.logaddexp(log_evidence, raw_score)

        # Posterior = score / total over every hypothesis considered (all words + raw string).
        # Only real words are offered: the raw string is already on screen as typed, and the
        # single letters the web corpus counts as "words" (B, C, D…) are not useful picks.
        words = [(w, s) for w, s in cands.items() if w in self.index and (len(w) > 1 or w in ("A", "I"))]
        ranked = sorted(words, key=lambda kv: -kv[1])[:top]
        out = [{"word": w, "prob": float(np.exp(s - log_evidence))} for w, s in ranked]
        # Hide hopeless guesses (e.g. while writing a number like 2026) rather than show noise.
        return [s for s in out if s["prob"] >= MIN_SUGGESTION_PROB]
