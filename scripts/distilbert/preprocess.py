import pandas as pd

def prepare_data(csv_path: str) -> tuple: 
    url_data = pd.read_csv(csv_path)
    df = url_data.copy()
    df["is_fox"] = df["url"].apply(lambda x: 1 if "foxnews.com" in x else 0)
    X = df["headline"].fillna("").astype(str).tolist()
    y = df["is_fox"].tolist()
    return X, y