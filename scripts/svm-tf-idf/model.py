
from __future__ import annotations

import html
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List
from urllib.parse import urlparse

import torch


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"


_LABEL_TO_INT = {"FOX": 0, "fox": 0, "NBC": 1, "nbc": 1}


def clean_strict(text: Any) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", html.unescape(str(text))).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _url_to_text(value: str) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        path = urlparse(value).path or ""
    except Exception:
        return value
    last = ""
    for segment in reversed([s for s in path.split("/") if s]):
        if any(ch.isalpha() for ch in segment):
            last = segment
            break
    if not last:
        return ""
    return last.replace("-", " ").replace("_", " ")


def _normalize_input(text: Any) -> str:
    s = "" if text is None else str(text)
    if s.lower().startswith(("http://", "https://", "www.")):
        s = _url_to_text(s)
    return clean_strict(s)


class NewsClassifier:

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
            self.pipeline = pickle.loads(blob)
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
        texts = [_normalize_input(x) for x in batch]
        raw_preds = self.pipeline.predict(texts).tolist()
        return [_LABEL_TO_INT.get(p, p) for p in raw_preds]

    def __call__(self, batch: Iterable[Any]) -> List[int]:
        return self.predict(batch)


Model = NewsClassifier


def get_model() -> NewsClassifier:
    return NewsClassifier()
