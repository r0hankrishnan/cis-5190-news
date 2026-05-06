
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import clean_strict  

REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = HERE / "model.pt"
SEED = 0


def build_dataset(fox_csv: Path, nbc_csv: Path) -> pd.DataFrame:
    fox = pd.read_csv(fox_csv)
    nbc = pd.read_csv(nbc_csv)
    fox["source"] = "FOX"
    nbc["source"] = "NBC"
    df = pd.concat([fox, nbc], ignore_index=True)

    df["text_clean"] = df["title"].map(clean_strict)
    df["datetime_posted"] = pd.to_datetime(
        df["datetime_posted"], utc=True, errors="coerce", format="mixed"
    )
    df = (
        df[df["text_clean"] != ""]
        .drop_duplicates(["text_clean", "source"])
        .reset_index(drop=True)
    )
    return df


def time_stratified_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for _, sub in df.groupby("source"):
        with_date = (
            sub.dropna(subset=["datetime_posted"])
            .sort_values("datetime_posted", kind="stable")
        )
        no_date = sub[sub["datetime_posted"].isna()]
        n_test = max(1, int(round(len(with_date) * test_frac)))
        train_parts.append(
            pd.concat([with_date.iloc[:-n_test], no_date], axis=0)
        )
        test_parts.append(with_date.iloc[-n_test:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "vec",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LinearSVC(C=0.5, max_iter=5000, random_state=SEED),
            ),
        ]
    )


def main() -> None:
    df = build_dataset(
        DATA_DIR / "fox_scraped_all.csv",
        DATA_DIR / "nbc_scraped_all.csv",
    )
    print(
        f"rows: {len(df)}  "
        f"fox/nbc: {df['source'].value_counts().to_dict()}"
    )

    train_df, test_df = time_stratified_split(df, test_frac=0.2)
    print(
        f"train: {len(train_df)} rows  "
        f"test: {len(test_df)} rows (most recent 20% per source)"
    )

    pipe = make_pipeline()
    pipe.fit(train_df["text_clean"].values, train_df["source"].values)
    preds = pipe.predict(test_df["text_clean"].values)
    holdout = float(accuracy_score(test_df["source"].values, preds))
    print(f"\ntime-stratified holdout accuracy: {holdout:.4f}")
    print(classification_report(test_df["source"].values, preds, digits=4))

    pipe.fit(df["text_clean"].values, df["source"].values)
    print(
        f"refit on full data: {len(df)} rows; "
        f"classes={list(pipe.classes_)}"
    )

    payload = {"sk_pipeline_bytes": pickle.dumps(pipe)}
    torch.save(payload, OUT_PATH)
    print(f"saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
