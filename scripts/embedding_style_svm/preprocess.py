import pandas as pd
import numpy as np
from typing import Any
import re

# ---------------------------------------------------------
# Lazy load big models + define globals
# ---------------------------------------------------------
_nlp = None
_embedder = None

def _get_nlp():
    """Lazy-load spaCy model (NER only, fast pipeline)."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
        except OSError:
            try:
                from spacy.cli.download import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm", disable = ["tagger", "parser", "lemmatizer"])
            
            except OSError as e:
                raise OSError(
                    "spaCy model 'en_core_web_sm' failed to load even after download. "
                    f"Original error: {e}"
                )
    return _nlp

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Can't instantiate embedding model. "
                "Must pip install sentence-transformers"
            )
    return _embedder

# ---------------------------------------------------------
# Get headlines and labels
# ---------------------------------------------------------
def _get_headlines_and_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["is_fox"] = df["url"].apply(lambda x: 1 if "foxnews.com" in x else 0)
    
    for col in ("headline", "scraped_headline", "alternative_headline", "title"):
        if col in df.columns:
            df["headline"] = df[col].fillna("").astype(str)
            break
        
    else:
        raise ValueError(f"No headline column found. Columns: {df.columns.tolist()}")
    
    return df[["headline", "is_fox"]].copy()

# ---------------------------------------------------------
# Extract style features
# ---------------------------------------------------------

def _extract_style_features(headline: str) -> dict[str, Any]:
    nlp = _get_nlp()
    text  = str(headline) if headline and not (isinstance(headline, float) and np.isnan(headline)) else ""
    words = text.split()
    n_words = max(len(words), 1)
    
    print("Running spaCy on headlines...")
    doc = nlp(text)
    print("spaCy complete.")
    
    ent_types = [ent.label_ for ent in doc.ents]
    ent_counts = {t: ent_types.count(t) for t in ['PERSON', 'ORG', 'GPE', 'NORP']}
    n_ents = max(len(doc.ents), 1)
    
    return {
         # Punctuation / formatting
        'scare_quote_count': len(re.findall(r"'[^']{2,30}'", text)),
        'has_colon': int(':' in text),
        # Structure
        'n_words': n_words,
        # spaCy NER counts
        'person_count': ent_counts['PERSON'],
        # NER ratios
        'person_to_ent_ratio': ent_counts['PERSON'] / n_ents,
    }
    
def _build_style_matrix(headlines: list[str]) -> np.ndarray:
    rows = [_extract_style_features(h) for h in headlines]
    return np.array([list(r.values()) for r in rows], dtype = float)

# ---------------------------------------------------------
# Embed headlines
# ---------------------------------------------------------

def _build_embedding_matrix(headlines: list[str]) -> np.ndarray:
    _embedder = _get_embedder()
    return _embedder.encode(headlines, batch_size = 64, show_progress_bar = False)

# ---------------------------------------------------------
# Preprocess data - get labels + headlines, get style features, 
# embed headlines, concat together, separate into X and y
# ---------------------------------------------------------

def prepare_data(csv_path: str) -> tuple:
    # Get basic df
    df = _get_headlines_and_labels(csv_path = csv_path)
    
    headlines = df["headline"].tolist()
    y = df["is_fox"].tolist()
    
    emb = _build_embedding_matrix(headlines) # (N, 384)
    style = _build_style_matrix(headlines) # (N, 5)
    X = np.concatenate([emb, style], axis = 1) # (N, 389)
    
    return X, y

