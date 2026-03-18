"""
models.py — Emotion classifier and intensity predictor.

Emotion classifier : LinearSVC wrapped in CalibratedClassifierCV
Intensity predictor: Ridge regression (continuous 1–5)
"""

import os
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# ── Emotion Classifier ──────────────────────────────────────────────────────

class EmotionClassifier:
    """Multi-class emotion classification using LinearSVC + calibration."""

    def __init__(self):
        base_svc = LinearSVC(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        )
        self.model = CalibratedClassifierCV(base_svc, cv=3)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X, y_true) -> dict:
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        return {"accuracy": acc, "report": report, "predictions": y_pred}

    def save(self, path: str | None = None):
        path = path or os.path.join(MODELS_DIR, "emotion_classifier.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | None = None) -> "EmotionClassifier":
        path = path or os.path.join(MODELS_DIR, "emotion_classifier.joblib")
        self.model = joblib.load(path)
        return self


# ── Intensity Predictor ──────────────────────────────────────────────────────

class IntensityPredictor:
    """Regression model for intensity score (1–5)."""

    def __init__(self):
        self.model = Ridge(alpha=1.0, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(X)
        # Clamp to valid range
        return np.clip(np.round(preds, 2), 1.0, 5.0)

    def evaluate(self, X, y_true) -> dict:
        y_pred = self.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "predictions": y_pred}

    def save(self, path: str | None = None):
        path = path or os.path.join(MODELS_DIR, "intensity_predictor.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | None = None) -> "IntensityPredictor":
        path = path or os.path.join(MODELS_DIR, "intensity_predictor.joblib")
        self.model = joblib.load(path)
        return self


# ── Model Comparison Utility ─────────────────────────────────────────────────

def compare_classifiers(X_train, y_train, X_val, y_val):
    """
    Compare LinearSVC, Logistic Regression, and Naive Bayes.
    Returns a list of dicts with model name and accuracy.
    """
    models = {
        "LinearSVC (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42),
            cv=3,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        results.append({"model": name, "accuracy": round(acc, 4)})

    return results
