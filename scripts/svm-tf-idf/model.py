"""News Headline Classifier (FOX vs NBC).

Loads a pickled scikit-learn pipeline from model.pt and exposes a
NewsClassifier whose predict(batch) returns 0 (FOX) or 1 (NBC).
"""
from __future__ import annotations

import html
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# The grader may import this file under a synthetic module name (not "model").
# The pickled pipeline records its custom transformers as `model._ColumnPicker`
# / `model.StyleFeatures`, so we register this module under that name too.
sys.modules.setdefault("model", sys.modules[__name__])

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"

LABEL_TO_INT = {"FOX": 0, "NBC": 1}
QUOTE_CHARS = '"\u2018\u2019\u201c\u201d'


def normalize_raw(text):
    """Light cleaning that preserves casing and punctuation."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", html.unescape(str(text)))
    return re.sub(r"\s+", " ", s).strip()


def clean_strict(text):
    """Lowercase, alphanumeric-only cleaning for the word TF-IDF branch."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", html.unescape(str(text))).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def url_to_text(url):
    """Pull a headline-like string from the slug of a URL."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        path = urlparse(url).path or ""
    except Exception:
        return url
    for seg in reversed([s for s in path.split("/") if s]):
        if any(c.isalpha() for c in seg):
            return seg.replace("-", " ").replace("_", " ")
    return ""


class _ColumnPicker(BaseEstimator, TransformerMixin):
    """Returns the named column of a DataFrame as a 1-D array."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].values


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Numeric style features per headline (length, casing, punctuation)."""

    feature_names = [
        "len_chars", "n_words", "mean_word_len",
        "caps_word_ratio", "allcaps_word_ratio",
        "digit_ratio", "punct_ratio",
        "has_question", "has_exclaim", "has_colon",
        "n_quotes", "n_emdash", "n_hyphen", "n_comma",
        "starts_with_quote", "ends_with_question",
    ]

    def fit(self, X, y=None):
        return self

    def _row(self, s):
        s = s or ""
        words = s.split()
        n = len(words) or 1
        cap = sum(1 for w in words if w[:1].isupper())
        allcap = sum(1 for w in words if len(w) >= 2 and w.isupper())
        digits = sum(1 for c in s if c.isdigit())
        punct = sum(1 for c in s if not c.isalnum() and not c.isspace())
        chars = len(s) or 1
        return [
            len(s), len(words),
            (sum(len(w) for w in words) / n) if words else 0.0,
            cap / n, allcap / n, digits / chars, punct / chars,
            float("?" in s), float("!" in s), float(":" in s),
            sum(1 for c in s if c in QUOTE_CHARS),
            s.count("\u2014"), s.count("-"), s.count(","),
            float(s[:1] in QUOTE_CHARS), float(s.endswith("?")),
        ]

    def transform(self, X):
        return sparse.csr_matrix(np.array([self._row(s) for s in X], dtype=np.float64))


def _to_input_frame(batch):
    """Turn an iterable of strings (or URLs) into the 2-column DataFrame
    that the feature pipeline expects: `text` (raw) and `text_clean`."""
    raw = []
    for item in batch:
        s = "" if item is None else str(item)
        if s.lower().startswith(("http://", "https://", "www.")):
            s = url_to_text(s)
        raw.append(normalize_raw(s))
    return pd.DataFrame({"text": raw, "text_clean": [clean_strict(s) for s in raw]})


class NewsClassifier:
    def __init__(self):
        self.pipeline = None
        if MODEL_PATH.exists():
            payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self.load_state_dict(payload)

    def load_state_dict(self, state, strict=True):
        if not (isinstance(state, dict) and "sk_pipeline_bytes" in state):
            raise ValueError("Expected a dict with key 'sk_pipeline_bytes'.")
        blob = state["sk_pipeline_bytes"]
        if isinstance(blob, torch.Tensor):
            blob = blob.cpu().numpy().tobytes()
        self.pipeline = pickle.loads(blob)

    def predict(self, batch):
        if self.pipeline is None:
            raise RuntimeError("Model not loaded; call load_state_dict() first.")
        df = _to_input_frame(batch)
        preds = self.pipeline.predict(df).tolist()
        return [LABEL_TO_INT.get(p, p) for p in preds]

    def __call__(self, batch):
        return self.predict(batch)


Model = NewsClassifier


def get_model():
    return NewsClassifier()
