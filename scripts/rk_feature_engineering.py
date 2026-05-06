"""This script performs stylistic feature extraction on the base cleaned data.
The selected features come from notebook 03_rk_style_eda.ipynb. The main function is build_feature_df,
which applies extract_features to each headline and returns a DataFrame of features.

Assumes you are working with cleaned and combined headline data with the following columns:
        - "title": raw headline text
        - "datetime_posted": timestamp of article
        - "is_fox": binary label (1 for Fox, 0 for NBC)

Usage:
    from feature_engineering import extract_features, build_feature_df
    feature_df = build_feature_df(df["title"])  # df is your headlines DataFrame
"""
import re
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

# Cleaned and combined headline data path and output path for final feature dataframe
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "combined_base_data.csv"
WRITE_TO_CSV_PATH = Path(__file__).parent.parent / "data" / "processed" / "style_features.csv"

# Feature col choices are justified in notebook 03_rk_style_eda.ipynb
FEATURE_COLS = ["n_words", "scare_quote_count", "has_colon", "person_to_ent_ratio", "person_count"]

# VADER and spaCy are only loaded when needed (slow to import)
_vader = None
_nlp = None

def _get_vader():
    """Lazy-load VADER sentiment analyzer."""
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError("VADER sentiment analyzer not found. Run: pip install vaderSentiment")
    return _vader

def _get_nlp():
    """Lazy-load spaCy model (NER only, fast pipeline)."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp

# Load base data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load cleaned and combined headline data."""
    df = pd.read_csv(path)
    
    assert "title" in df.columns, "Expected 'title' column in data"
    assert "datetime_posted" in df.columns, "Expected 'datetime_posted' column in data"
    assert "is_fox" in df.columns, "Expected 'is_fox' column in data"
    
    df['datetime_posted'] = pd.to_datetime(df['datetime_posted'], utc=True, format='mixed')
    df = df.dropna(subset=['title']).reset_index(drop=True)
    return df

# Extract all features
def extract_all_features(headline: str) -> dict[str, Any]:
    """
    Extract the all stylistic features from a single headline string.

    Parameters
    ----------
    headline : str
        Raw headline text.

    Returns
    -------
    dict of all stylistic features (24 total)
    """
    vader = _get_vader()
    nlp = _get_nlp()
    text  = str(headline) if headline and not (isinstance(headline, float) and np.isnan(headline)) else ""
    words = text.split()
    n_words = max(len(words), 1)
    vader_score = vader.polarity_scores(text)
    doc = nlp(text)
    ent_types = [ent.label_ for ent in doc.ents]
    ent_counts = {t: ent_types.count(t) for t in ['PERSON', 'ORG', 'GPE', 'NORP']}
    n_ents = max(len(doc.ents), 1)
    
    return {
         # Punctuation / formatting
        'has_question':        int(text.endswith('?')),
        'scare_quote_count':   len(re.findall(r"'[^']{2,30}'", text)),
        'ellipsis_count':      text.count('...'),
        'exclamation_count':   text.count('!'),
        'has_colon':           int(':' in text),
        # Capitalization
        'allcaps_word_count':  sum(1 for w in words if w.isupper() and len(w) > 1),
        'title_case_ratio':    sum(1 for w in words if w.istitle()) / n_words,
        # VADER sentiment
        'vader_compound':      vader_score['compound'],
        'vader_neg':           vader_score['neg'],
        'vader_pos':           vader_score['pos'],
        'vader_neu':           vader_score['neu'],
        # Attribution / hedge language
        'has_says':            int(bool(re.search(r'\bsays?\b|\bsaid\b', text, re.I))),
        'has_report':          int(bool(re.search(r'\breports?\b|\breported\b', text, re.I))),
        'has_sources':         int(bool(re.search(r'\bsources?\b', text, re.I))),
        'has_allegedly':       int(bool(re.search(r'\ballegedly\b|\baccused\b|\bclaims?\b', text, re.I))),
        # Structure
        'n_words':             n_words,
        'starts_with_number':  int(bool(re.match(r'^\d', text))),
        'type_token_ratio':    len(set(words)) / n_words,
        # spaCy NER counts
        'person_count':        ent_counts['PERSON'],
        'org_count':           ent_counts['ORG'],
        'gpe_count':           ent_counts['GPE'],
        'norp_count':          ent_counts['NORP'],
        'total_ents':          len(doc.ents),
        # NER ratios
        'person_to_ent_ratio': ent_counts['PERSON'] / n_ents,
        'org_to_ent_ratio':    ent_counts['ORG'] / n_ents,
    }

# Build full stylistic feature dataframe 
def build_full_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build dataframe containing all 24 stylistic features and 'title', 'datetime_posted', and 'is_fox'.

    Args:
        df (pd.DataFrame): Cleaned and combined headline dataframe. Get with load_data().

    Returns:
        pd.DataFrame: A dataframe with all 24 stylistic features and 'title', 'datetime_posted', and 'is_fox'.
    """
    df_feat = df["title"].apply(extract_all_features).apply(pd.Series)
    
    df_feat['is_fox'] = df['is_fox'].values
    df_feat['datetime_posted'] = df['datetime_posted'].values
    df_feat['title'] = df['title'].values
    
    return df_feat

def summarize_features(full_feature_df: pd.DataFrame, features: list = FEATURE_COLS) -> pd.DataFrame:
    """
    Print a comparison table of feature means by Fox (1) vs NBC (0).
    Used for EDA / sanity checks.
    """
    df = full_feature_df.copy()[[c for c in full_feature_df.columns if c not in ["datetime_posted", "title"]]]
    
    summary = df.groupby("is_fox")[features].mean().T
    summary = summary.rename(columns={0: "NBC", 1: "Fox"})
    
    # Pooled std dev and Cohen's d -> from googling about how to compare two groups on multiple features
    n_fox, n_nbc = len(df[df['is_fox'] == 1]), len(df[df['is_fox'] == 0])
    pooled_std = np.sqrt(
        ((n_fox - 1) * df[df['is_fox'] == 1][features].std()**2 + (n_nbc - 1) * df[df['is_fox'] == 0][features].std()**2)
        / (n_fox + n_nbc - 2)
    )

    summary['abs_diff'] = (summary['Fox'] - summary['NBC']).abs()
    summary['cohens_d'] = (summary['Fox'] - summary['NBC']) / pooled_std
    summary['abs_cohens_d'] = summary['cohens_d'].abs()

    summary = summary.sort_values('abs_cohens_d', ascending=False).round(4)
    
    return summary
    
def build_final_feature_df(full_feature_df: pd.DataFrame, features = FEATURE_COLS):
    """Takes full feature df and filters for relevant columns (defined above or can be overwritten when called).

    Args:
        full_feature_df (pd.DataFrame): Output of build_full_feature_df, containing all 24 stylistic features + 
            'title', "datetime_posted", and 'is_fox'.
        features (_type_, optional): Selected stylistic features. Defaults to FEATURE_COLS.
        
    Returns:
        pd.DataFrame: A dataframe with 'title', 'datetime_posted', 'is_fox', and selected stylistic feature columns.
    """
    final_df = full_feature_df.copy()[['datetime_posted', 'title'] + features + ['is_fox']]
    return final_df

def run_full_style_feature_pipeline(write_to_csv: bool = True, path: Path | str = WRITE_TO_CSV_PATH):
    """Run the full feature engineering pipeline: load data, extract features, summarize, and save."""
    df = load_data()
    full_feature_df = build_full_feature_df(df)
    summary_df = summarize_features(full_feature_df)
    
    print("Feature comparison summary (sorted by absolute Cohen's d):")
    print(summary_df.to_string())
    
    final_feature_df = build_final_feature_df(full_feature_df)
    
    # Save final feature dataframe if True
    if write_to_csv:
        os.make_dirs(path.parent, exist_ok = True)
        final_feature_df.to_csv(path, index=False)
        print(f"Saved style features to {path} with shape {final_feature_df.shape}")
        return final_feature_df
    
    else:
        return final_feature_df