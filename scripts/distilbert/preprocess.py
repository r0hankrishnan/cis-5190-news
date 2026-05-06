import pandas as pd

def prepare_data(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    df["is_fox"] = df["url"].apply(lambda x: 1 if "foxnews.com" in x else 0)
    
    for col in ("headline", "scraped_headline", "alternative_headline", "title"):
        if col in df.columns:
            X = df[col].fillna("").astype(str).tolist()
            break
    else:
        raise ValueError(f"No headline column found. Columns: {df.columns.tolist()}")
    
    y = df["is_fox"].tolist()
    return X, y