
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd


_HEADLINE_NAME_CANDIDATES = (
    "headline",
    "title",
    "text",
    "news_headline",
    "article_title",
    "content",
)

_URL_NAME_CANDIDATES = (
    "url",
    "link",
    "article_url",
    "page_url",
)

_LABEL_NAME_CANDIDATES = (
    "label",
    "labels",
    "source",
    "news_source",
    "class",
    "target",
    "y",
)

_FOX_TOKENS = {"fox", "fox news", "foxnews", "0"}
_NBC_TOKENS = {"nbc", "nbc news", "nbcnews", "1"}


def _lower_cols(df: pd.DataFrame) -> dict:
    return {c.lower(): c for c in df.columns}


def _looks_like_urls(values: Sequence[str], threshold: float = 0.5) -> bool:
    if not values:
        return False
    sample = values[: min(50, len(values))]
    hits = sum(
        1
        for v in sample
        if isinstance(v, str)
        and v.lower().startswith(("http://", "https://", "www."))
    )
    return hits / max(len(sample), 1) >= threshold


def _find_url_column(df: pd.DataFrame) -> Optional[str]:
    lower = _lower_cols(df)
    for cand in _URL_NAME_CANDIDATES:
        if cand in lower:
            return lower[cand]
    for col in df.columns:
        try:
            if _looks_like_urls(df[col].fillna("").astype(str).tolist()):
                return col
        except Exception:
            continue
    return None


def _find_headline_column(
    df: pd.DataFrame, exclude: Sequence[str] = ()
) -> Optional[str]:
    excluded = {c for c in exclude if c is not None}
    lower = _lower_cols(df)
    for cand in _HEADLINE_NAME_CANDIDATES:
        if cand in lower and lower[cand] not in excluded:
            return lower[cand]
    for col in df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        if _looks_like_urls(df[col].fillna("").astype(str).tolist()):
            continue
        return col
    return None


def _find_label_column(
    df: pd.DataFrame, exclude: Sequence[str] = ()
) -> Optional[str]:
    excluded = {c for c in exclude if c is not None}
    lower = _lower_cols(df)
    for cand in _LABEL_NAME_CANDIDATES:
        if cand in lower and lower[cand] not in excluded:
            return lower[cand]
    return None


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


def _label_from_url(value: str) -> Optional[int]:
    if not isinstance(value, str):
        return None
    v = value.lower()
    if "foxnews" in v or "fox-news" in v or v.startswith("fox"):
        return 0
    if "nbcnews" in v or "nbc-news" in v or v.startswith("nbc"):
        return 1
    return None


def _coerce_binary_labels(series: pd.Series) -> List[int]:
    if pd.api.types.is_numeric_dtype(series):
        as_int = pd.Series(series).fillna(0).astype(int)
        unique = set(as_int.unique().tolist())
        if unique.issubset({0, 1}):
            return as_int.tolist()

    norm = (
        pd.Series(series)
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    def _map_one(val: str) -> Optional[int]:
        if val in _FOX_TOKENS:
            return 0
        if val in _NBC_TOKENS:
            return 1
        if "foxnews" in val or val.startswith("fox"):
            return 0
        if "nbcnews" in val or val.startswith("nbc"):
            return 1
        return None

    mapped = norm.map(_map_one)
    if mapped.isna().any():
        bad = norm[mapped.isna()].head(5).tolist()
        raise ValueError(
            f"Could not map label column to 0/1. Sample bad values: {bad}."
        )
    return mapped.astype(int).tolist()


def _labels_from_url_series(series: pd.Series) -> List[int]:
    urls = pd.Series(series).fillna("").astype(str).tolist()
    out: List[int] = []
    bad: List[str] = []
    for u in urls:
        lbl = _label_from_url(u)
        if lbl is None:
            bad.append(u)
        out.append(lbl if lbl is not None else -1)
    if bad:
        raise ValueError(
            f"Could not infer FOX/NBC label from {len(bad)} URL(s); "
            f"sample: {bad[:3]}."
        )
    return out


def prepare_data(csv_path: str) -> Tuple[Sequence[str], Sequence[int]]:
    df = pd.read_csv(csv_path)

    if df.empty or len(df.columns) == 0:
        raise ValueError("Empty CSV: no columns to read.")

    url_col = _find_url_column(df)
    label_col = _find_label_column(df, exclude=[url_col] if url_col else [])
    head_col = _find_headline_column(
        df, exclude=[c for c in (url_col, label_col) if c]
    )

    if head_col is None:
        head_col = url_col

    if head_col is None:
        raise ValueError(
            f"Could not find an input column in {list(df.columns)}."
        )

    raw_inputs = df[head_col].fillna("").astype(str).tolist()
    if _looks_like_urls(raw_inputs):
        X: List[str] = [_url_to_text(v) for v in raw_inputs]
    else:
        X = raw_inputs

    if label_col is not None and label_col != head_col:
        y = _coerce_binary_labels(df[label_col])
    elif url_col is not None:
        y = _labels_from_url_series(df[url_col])
    else:
        non_input_cols = [c for c in df.columns if c != head_col]
        last_err: Optional[Exception] = None
        y = None
        for c in reversed(non_input_cols):
            try:
                y = _coerce_binary_labels(df[c])
                break
            except Exception as err:
                last_err = err
                continue
        if y is None:
            raise ValueError(
                "Could not find a label column or infer labels from URLs. "
                f"Columns seen: {list(df.columns)}. "
                f"Last error: {last_err}"
            )

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")

    return X, y
