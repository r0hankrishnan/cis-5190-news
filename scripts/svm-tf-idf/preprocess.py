from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd

HEADLINE_NAMES = ("headline", "title", "text", "news_headline", "article_title", "content")
URL_NAMES = ("url", "link", "article_url", "page_url")
LABEL_NAMES = ("label", "labels", "source", "news_source", "class", "target", "y")

FOX_TOKENS = {"fox", "fox news", "foxnews", "0"}
NBC_TOKENS = {"nbc", "nbc news", "nbcnews", "1"}


def _looks_like_urls(values, threshold=0.5):
    if not values:
        return False
    sample = values[: min(50, len(values))]
    hits = sum(
        1 for v in sample
        if isinstance(v, str) and v.lower().startswith(("http://", "https://", "www."))
    )
    return hits / max(len(sample), 1) >= threshold


def _find_named(df, candidates, exclude=()):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower and lower[c] not in exclude:
            return lower[c]
    return None


def _find_url_column(df):
    col = _find_named(df, URL_NAMES)
    if col:
        return col
    for c in df.columns:
        try:
            if _looks_like_urls(df[c].fillna("").astype(str).tolist()):
                return c
        except Exception:
            continue
    return None


def _url_to_text(url):
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


def _label_from_url(url):
    if not isinstance(url, str):
        return None
    v = url.lower()
    if "foxnews" in v or "fox-news" in v or v.startswith("fox"):
        return 0
    if "nbcnews" in v or "nbc-news" in v or v.startswith("nbc"):
        return 1
    return None


def _coerce_labels(series):
    if pd.api.types.is_numeric_dtype(series):
        ints = pd.Series(series).fillna(0).astype(int)
        if set(ints.unique()).issubset({0, 1}):
            return ints.tolist()
    norm = pd.Series(series).fillna("").astype(str).str.strip().str.lower()
    out = []
    for v in norm:
        if v in FOX_TOKENS:
            out.append(0)
        elif v in NBC_TOKENS:
            out.append(1)
        elif "foxnews" in v or v.startswith("fox"):
            out.append(0)
        elif "nbcnews" in v or v.startswith("nbc"):
            out.append(1)
        else:
            raise ValueError(f"Could not map label '{v}' to 0/1.")
    return out


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Empty CSV.")

    url_col = _find_url_column(df)
    label_col = _find_named(df, LABEL_NAMES, exclude=[url_col] if url_col else [])
    head_col = _find_named(df, HEADLINE_NAMES, exclude=[c for c in (url_col, label_col) if c])

    if head_col is None:
        for c in df.columns:
            if c in (url_col, label_col):
                continue
            if _looks_like_urls(df[c].fillna("").astype(str).tolist()):
                continue
            head_col = c
            break
    if head_col is None:
        head_col = url_col
    if head_col is None:
        raise ValueError(f"No usable input column in {list(df.columns)}.")

    raw = df[head_col].fillna("").astype(str).tolist()
    X = [_url_to_text(v) for v in raw] if _looks_like_urls(raw) else raw

    if label_col is not None and label_col != head_col:
        y = _coerce_labels(df[label_col])
    elif url_col is not None:
        y = []
        for u in df[url_col].fillna("").astype(str):
            lbl = _label_from_url(u)
            if lbl is None:
                raise ValueError(f"Could not infer FOX/NBC from URL: {u!r}")
            y.append(lbl)
    else:
        y = None
        for c in reversed([c for c in df.columns if c != head_col]):
            try:
                y = _coerce_labels(df[c])
                break
            except ValueError:
                continue
        if y is None:
            raise ValueError(f"Could not derive labels from columns: {list(df.columns)}.")

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")
    return X, y
