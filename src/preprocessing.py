"""
preprocessing.py — Text cleaning and DataFrame preprocessing.

Handles:
  - Lowercasing, URL removal, special character stripping
  - Tokenization and stopword removal (NLTK)
  - Missing-value imputation for contextual columns
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (one-time, silent after first run)
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Text-Level Cleaning ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip URLs, special chars, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"[^a-z\s]", "", text)                # non-alpha
    text = re.sub(r"\s+", " ", text).strip()             # collapse spaces
    return text


def tokenize_and_filter(text: str) -> str:
    """Tokenize, remove stopwords, lemmatize, and rejoin."""
    tokens = word_tokenize(text)
    filtered = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 1
    ]
    return " ".join(filtered)


def preprocess_text(text: str) -> str:
    """Full text pipeline: clean → tokenize → filter."""
    return tokenize_and_filter(clean_text(text))


# ── DataFrame-Level Preprocessing ───────────────────────────────────────────

CATEGORICAL_COLS = [
    "ambience_type",
    "time_of_day",
    "previous_day_mood",
    "face_emotion_hint",
    "reflection_quality",
]

NUMERIC_COLS = [
    "duration_min",
    "sleep_hours",
    "energy_level",
    "stress_level",
]


def preprocess_dataframe(df, is_training: bool = True):
    """
    Clean and prepare the full DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (training or test).
    is_training : bool
        If True, expects 'emotional_state' and 'intensity' columns.

    Returns
    -------
    df : pd.DataFrame  (modified copy)
    """
    df = df.copy()

    # ── Handle missing values ────────────────────────────────────────────
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
    df["previous_day_mood"] = df["previous_day_mood"].fillna("unknown")
    df["face_emotion_hint"] = df["face_emotion_hint"].fillna("none")

    # ── Clean text ───────────────────────────────────────────────────────
    df["clean_text"] = df["journal_text"].apply(preprocess_text)

    return df
