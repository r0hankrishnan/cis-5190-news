

from __future__ import annotations

import html
import io
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"


sys.modules.setdefault("model", sys.modules[__name__])

_LABEL_TO_INT = {"FOX": 0, "fox": 0, "NBC": 1, "nbc": 1}
_QUOTE_CHARS = '"\u2018\u2019\u201c\u201d'

_SOURCE_TAG_PREFIX_RE = re.compile(
    r"^\s*(fox\s*news|nbc\s*news|foxnews|nbcnews|fox|nbc)(\.com)?"
    r"\s*[\-\|\u2013\u2014:]\s*",
    flags=re.IGNORECASE,
)
_SOURCE_TAG_SUFFIX_RE = re.compile(
    r"\s*[\-\|\u2013\u2014:]\s*(fox\s*news|nbc\s*news|foxnews|nbcnews)"
    r"(\.com)?\s*$",
    flags=re.IGNORECASE,
)
_SOURCE_TAG_LOOSE_SUFFIX_RE = re.compile(
    r"\s+(fox\s*news|nbc\s*news|foxnews|nbcnews)\s*$"
)


def clean_strict(text: Any) -> str:
    if text is None:
        return ""
    s = html.unescape(str(text)).lower()
    s = _SOURCE_TAG_PREFIX_RE.sub(" ", s)
    s = _SOURCE_TAG_SUFFIX_RE.sub(" ", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _SOURCE_TAG_LOOSE_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_raw(text: Any) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", html.unescape(str(text)))
    return re.sub(r"\s+", " ", s).strip()


def _url_to_text(value: str) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        path = urlparse(value).path or ""
    except Exception:
        return value
    for segment in reversed([s for s in path.split("/") if s]):
        if any(ch.isalpha() for ch in segment):
            return segment.replace("-", " ").replace("_", " ")
    return ""


class _ColumnPicker(BaseEstimator, TransformerMixin):

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].values


class StyleFeatures(BaseEstimator, TransformerMixin):

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

    def _row(self, s: str):
        s = s or ""
        words = s.split()
        n = len(words) or 1
        cap = sum(1 for w in words if w[:1].isupper())
        allcap = sum(1 for w in words if len(w) >= 2 and w.isupper())
        digits = sum(1 for ch in s if ch.isdigit())
        punct = sum(1 for ch in s if not ch.isalnum() and not ch.isspace())
        chars = len(s) or 1
        return [
            len(s),
            len(words),
            (sum(len(w) for w in words) / n) if words else 0.0,
            cap / n,
            allcap / n,
            digits / chars,
            punct / chars,
            float("?" in s),
            float("!" in s),
            float(":" in s),
            sum(1 for ch in s if ch in _QUOTE_CHARS),
            s.count("\u2014"),
            s.count("-"),
            s.count(","),
            float(s[:1] in _QUOTE_CHARS),
            float(s.endswith("?")),
        ]

    def transform(self, X):
        arr = np.array([self._row(s) for s in X], dtype=np.float64)
        return sparse.csr_matrix(arr)


def _coerce_str_item(item: Any) -> str:
    s = "" if item is None else str(item)
    if s.lower().startswith(("http://", "https://", "www.")):
        s = _url_to_text(s)
    return s


def _to_clean_strings(batch: Iterable[Any]) -> List[str]:
    return [clean_strict(_coerce_str_item(item)) for item in batch]


def _to_input_frame(batch: Iterable[Any]) -> pd.DataFrame:
    if isinstance(batch, pd.DataFrame):
        df = batch.copy()
        if "text" not in df.columns and "text_clean" not in df.columns:
            first_col = df.columns[0]
            df["text"] = df[first_col].astype(str).map(normalize_raw)
        elif "text" not in df.columns:
            df["text"] = df["text_clean"].astype(str).map(normalize_raw)
        if "text_clean" not in df.columns:
            df["text_clean"] = df["text"].astype(str).map(clean_strict)
        return df[["text", "text_clean"]]

    raw: List[str] = []
    for item in batch:
        raw.append(normalize_raw(_coerce_str_item(item)))
    return pd.DataFrame(
        {"text": raw, "text_clean": [clean_strict(s) for s in raw]}
    )


class _PortableUnpickler(pickle.Unpickler):

    _LOCAL_NAMES = {"_ColumnPicker", "StyleFeatures"}
    _ALIASED_MODULES = {"model", "__main__"}

    def find_class(self, module, name):
        if name in self._LOCAL_NAMES:
            return globals()[name]
        if module in self._ALIASED_MODULES:
            here = sys.modules[__name__]
            if hasattr(here, name):
                return getattr(here, name)
        return super().find_class(module, name)


def _portable_loads(blob: bytes):
    return _PortableUnpickler(io.BytesIO(blob)).load()


class NewsClassifier:
    """Inference wrapper around the trained sklearn pipeline."""

    def __init__(self) -> None:
        self.pipeline = None
        if MODEL_PATH.exists():
            payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self.load_state_dict(payload)

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> None:
        if isinstance(state_dict, dict) and "sk_pipeline_bytes" in state_dict:
            blob = state_dict["sk_pipeline_bytes"]
            if isinstance(blob, torch.Tensor):
                blob = blob.cpu().numpy().tobytes()
            self.pipeline = _portable_loads(blob)
            return
        raise ValueError(
            "Unexpected state_dict for NewsClassifier; "
            "expected dict with key 'sk_pipeline_bytes'."
        )

    def predict(self, batch: Iterable[Any]) -> List[int]:
        if self.pipeline is None:
            raise RuntimeError(
                "Model not loaded. Place model.pt next to model.py or call "
                "load_state_dict(torch.load('model.pt', map_location='cpu'))."
            )

        if isinstance(batch, pd.DataFrame):
            try:
                raw_preds = self.pipeline.predict(_to_input_frame(batch)).tolist()
            except Exception:
                raw_preds = self.pipeline.predict(
                    [clean_strict(s) for s in batch.iloc[:, 0].astype(str)]
                ).tolist()
        else:
            cleaned = _to_clean_strings(batch)
            try:
                raw_preds = self.pipeline.predict(cleaned).tolist()
            except (KeyError, AttributeError, ValueError, TypeError):
                raw_preds = self.pipeline.predict(_to_input_frame(batch)).tolist()

        return [_LABEL_TO_INT.get(p, p) for p in raw_preds]

    def __call__(self, batch: Iterable[Any]) -> List[int]:
        return self.predict(batch)


Model = NewsClassifier


def get_model() -> NewsClassifier:
    return NewsClassifier()
