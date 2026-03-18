"""
feature_engineering.py — TF-IDF + categorical/numeric feature extraction.

Produces a combined sparse feature matrix from:
  1. TF-IDF on cleaned journal text (unigrams + bigrams)
  2. Label-encoded categorical columns
  3. Raw numeric columns (scaled)
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.preprocessing import CATEGORICAL_COLS, NUMERIC_COLS

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class FeatureExtractor:
    """Fit on training data, transform both train and test."""

    def __init__(self, max_tfidf_features: int = 5000):
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self._fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """Fit all transformers on training data."""
        # TF-IDF
        self.tfidf.fit(df["clean_text"])

        # Label encoders (add "unknown" class for unseen values at inference)
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            unique_vals = list(df[col].unique()) + ["unknown"]
            le.fit(unique_vals)
            self.label_encoders[col] = le

        # Scaler for numeric cols
        self.scaler.fit(df[NUMERIC_COLS].values)

        self._fitted = True
        return self

    # ── Transform ─────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame):
        """Return combined sparse feature matrix."""
        assert self._fitted, "Call fit() before transform()."

        # 1. TF-IDF features
        tfidf_matrix = self.tfidf.transform(df["clean_text"])

        # 2. Categorical features (encoded)
        cat_encoded = []
        for col in CATEGORICAL_COLS:
            le = self.label_encoders[col]
            # Map unseen labels to "unknown"
            safe_vals = df[col].apply(
                lambda v: v if v in le.classes_ else "unknown"
            )
            cat_encoded.append(le.transform(safe_vals).reshape(-1, 1))
        cat_matrix = csr_matrix(np.hstack(cat_encoded))

        # 3. Numeric features (scaled)
        num_matrix = csr_matrix(self.scaler.transform(df[NUMERIC_COLS].values))

        # Combine all
        combined = hstack([tfidf_matrix, cat_matrix, num_matrix])
        return combined

    def fit_transform(self, df: pd.DataFrame):
        """Convenience: fit + transform in one call."""
        return self.fit(df).transform(df)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or os.path.join(MODELS_DIR, "feature_extractor.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | None = None) -> "FeatureExtractor":
        path = path or os.path.join(MODELS_DIR, "feature_extractor.joblib")
        return joblib.load(path)
