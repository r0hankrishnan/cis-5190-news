
from __future__ import annotations

import html
import pickle
import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent / "model.pt"


def clean_strict(x):
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKC", html.unescape(str(x))).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_dataset(fox_csv: Path, nbc_csv: Path) -> pd.DataFrame:
    fox = pd.read_csv(fox_csv)
    nbc = pd.read_csv(nbc_csv)

    fox_text = fox["title"].fillna("").astype(str)
    nbc_text = nbc["title"].fillna("").astype(str)

    df = pd.concat(
        [
            pd.DataFrame({"text": fox_text, "source": "FOX"}),
            pd.DataFrame({"text": nbc_text, "source": "NBC"}),
        ],
        ignore_index=True,
    )
    df["text_clean"] = df["text"].map(clean_strict)
    df = df[df["text_clean"] != ""].drop_duplicates(["text_clean", "source"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build_dataset(DATA_DIR / "fox_scraped_all.csv", DATA_DIR / "nbc_scraped_all.csv")
    print(f"rows: {len(df)}  fox/nbc: {df['source'].value_counts().to_dict()}")

    X, y = df["text_clean"], df["source"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(C=1.0, random_state=0)),
        ]
    )

    pipe.fit(X_train, y_train)
    holdout = float(accuracy_score(y_test, pipe.predict(X_test)))
    print(f"holdout accuracy (20% split, seed=0): {holdout:.4f}")

    pipe.fit(X, y)
    print(f"refit on full data: {len(X)} rows; classes={list(pipe.classes_)}")

    payload = {"sk_pipeline_bytes": pickle.dumps(pipe)}
    torch.save(payload, OUT_PATH)
    print(f"saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
