"""
pipeline.py — End-to-end prediction pipeline.

Usage:
    from src.pipeline import EmotionPipeline
    pipeline = EmotionPipeline.load()
    result = pipeline.predict("I feel so overwhelmed with everything")
"""

import os
import pandas as pd

from src.preprocessing import preprocess_text, preprocess_dataframe
from src.feature_engineering import FeatureExtractor
from src.models import EmotionClassifier, IntensityPredictor
from src.decision_engine import get_recommendation

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class EmotionPipeline:
    """Full pipeline: text → emotion + intensity + recommendation."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classifier: EmotionClassifier,
        intensity_model: IntensityPredictor,
    ):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.intensity_model = intensity_model

    # ── Single-text prediction ───────────────────────────────────────────

    def predict(
        self,
        text: str,
        time_of_day: str = "morning",
        stress_level: int = 3,
        energy_level: int = 3,
        sleep_hours: float = 7.0,
        ambience_type: str = "none",
        duration_min: int = 10,
        previous_day_mood: str = "unknown",
        face_emotion_hint: str = "none",
        reflection_quality: str = "clear",
    ) -> dict:
        """
        Predict emotion, intensity, and get recommendation.

        Parameters
        ----------
        text : str
            Raw journal text input.
        time_of_day, stress_level, energy_level, sleep_hours :
            Optional contextual features (use defaults if unknown).

        Returns
        -------
        dict with: emotion, intensity, intensity_label, recommendation,
                   timing_label, timing_detail
        """
        # Build a single-row DataFrame with all expected columns
        row = {
            "journal_text": text,
            "ambience_type": ambience_type,
            "duration_min": duration_min,
            "sleep_hours": sleep_hours,
            "energy_level": energy_level,
            "stress_level": stress_level,
            "time_of_day": time_of_day,
            "previous_day_mood": previous_day_mood,
            "face_emotion_hint": face_emotion_hint,
            "reflection_quality": reflection_quality,
        }
        df = pd.DataFrame([row])
        df = preprocess_dataframe(df, is_training=False)

        # Feature extraction
        X = self.feature_extractor.transform(df)

        # Predictions
        emotion = self.classifier.predict(X)[0]
        intensity = float(self.intensity_model.predict(X)[0])

        # Decision engine
        result = get_recommendation(
            emotion=emotion,
            intensity=intensity,
            time_of_day=time_of_day,
            stress_level=stress_level,
            energy_level=energy_level,
            sleep_hours=sleep_hours,
        )
        return result

    # ── Batch prediction on DataFrame ────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions on a full DataFrame (e.g., the test set).
        """
        df_processed = preprocess_dataframe(df, is_training=False)
        X = self.feature_extractor.transform(df_processed)

        emotions = self.classifier.predict(X)
        intensities = self.intensity_model.predict(X)

        results = []
        for i in range(len(df)):
            row = df.iloc[i]
            rec = get_recommendation(
                emotion=emotions[i],
                intensity=float(intensities[i]),
                time_of_day=row.get("time_of_day", "morning"),
                stress_level=int(row.get("stress_level", 3)),
                energy_level=int(row.get("energy_level", 3)),
                sleep_hours=float(row.get("sleep_hours", 7.0)),
            )
            results.append(rec)

        results_df = pd.DataFrame(results)
        return pd.concat(
            [df[["id", "journal_text"]].reset_index(drop=True), results_df],
            axis=1,
        )

    # ── Load saved models ────────────────────────────────────────────────

    @classmethod
    def load(cls, models_dir: str | None = None) -> "EmotionPipeline":
        """Load all saved models and return a ready pipeline."""
        models_dir = models_dir or MODELS_DIR

        feature_extractor = FeatureExtractor.load(
            os.path.join(models_dir, "feature_extractor.joblib")
        )
        classifier = EmotionClassifier()
        classifier.load(os.path.join(models_dir, "emotion_classifier.joblib"))

        intensity_model = IntensityPredictor()
        intensity_model.load(os.path.join(models_dir, "intensity_predictor.joblib"))

        return cls(feature_extractor, classifier, intensity_model)
