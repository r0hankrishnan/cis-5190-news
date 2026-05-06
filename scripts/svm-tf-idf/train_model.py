"""Train the FOX-vs-NBC pipeline and save it to model.pt for submission."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import _ColumnPicker, StyleFeatures, clean_strict, normalize_raw  # noqa: E402

DATA_DIR = HERE.parents[1] / "data" / "raw"
OUT_PATH = HERE / "model.pt"
SEED = 0


def time_stratified_split(df, frac=0.2):
    """Hold out the most recent `frac` per source, ordered by datetime_posted."""
    train, test = [], []
    for _, sub in df.groupby("source"):
        dated = sub.dropna(subset=["datetime_posted"]).sort_values("datetime_posted")
        no_date = sub[sub["datetime_posted"].isna()]
        n_test = max(1, int(round(len(dated) * frac)))
        train.append(pd.concat([dated.iloc[:-n_test], no_date], axis=0))
        test.append(dated.iloc[-n_test:])
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def word_dropout(text, rate, rng):
    if not text:
        return text
    words = text.split()
    if len(words) <= 2:
        return text
    keep = rng.random(len(words)) >= rate
    out = [w for w, k in zip(words, keep) if k]
    return " ".join(out) if out else text


def main():
    fox = pd.read_csv(DATA_DIR / "fox_scraped_all.csv")
    nbc = pd.read_csv(DATA_DIR / "nbc_scraped_all.csv")
    fox["source"] = "FOX"
    nbc["source"] = "NBC"
    df = pd.concat([fox, nbc], ignore_index=True)

    df["text"] = df["title"].map(normalize_raw)
    df["text_clean"] = df["text"].map(clean_strict)
    df["datetime_posted"] = pd.to_datetime(
        df["datetime_posted"], utc=True, errors="coerce", format="mixed"
    )
    df = df[df["text_clean"] != ""].drop_duplicates(["text_clean", "source"]).reset_index(drop=True)
    print(f"rows: {len(df)}  fox/nbc: {df['source'].value_counts().to_dict()}")

    train_df, test_df = time_stratified_split(df, frac=0.2)

    # Word-dropout augmentation: 1 extra training copy per row, ~10% words dropped.
    rng = np.random.default_rng(SEED)
    aug_text = train_df["text"].map(lambda s: word_dropout(s, 0.10, rng))
    aug_df = pd.DataFrame({
        "text": aug_text,
        "text_clean": aug_text.map(clean_strict),
        "source": train_df["source"].values,
    })
    train_df = pd.concat(
        [train_df[["text", "text_clean", "source"]], aug_df], ignore_index=True
    )
    print(f"train (with aug): {len(train_df)}  test: {len(test_df)}")

    features = FeatureUnion([
        ("word", Pipeline([
            ("pick", _ColumnPicker("text_clean")),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
        ])),
        ("char", Pipeline([
            ("pick", _ColumnPicker("text")),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      min_df=2, max_df=0.98, sublinear_tf=True)),
        ])),
        ("style", Pipeline([
            ("pick", _ColumnPicker("text")),
            ("feat", StyleFeatures()),
            ("scale", StandardScaler(with_mean=False)),
        ])),
    ])
    pipe = Pipeline([
        ("feat", features),
        ("clf", LinearSVC(C=1.0, max_iter=10000, random_state=SEED)),
    ])

    cols = ["text", "text_clean"]
    pipe.fit(train_df[cols], train_df["source"].values)
    preds = pipe.predict(test_df[cols])
    print(f"\nholdout accuracy: {accuracy_score(test_df['source'], preds):.4f}")
    print(classification_report(test_df["source"], preds, digits=4))

    full = pd.concat([train_df, test_df[cols + ["source"]]], ignore_index=True)
    pipe.fit(full[cols], full["source"].values)
    print(f"refit on full: {len(full)} rows")

    torch.save({"sk_pipeline_bytes": pickle.dumps(pipe)}, OUT_PATH)
    print(f"saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
