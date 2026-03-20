"""
main.py - Training, evaluation, and sample predictions.

Run:   python main.py
"""

import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_dataframe
from src.feature_engineering import FeatureExtractor
from src.models import EmotionClassifier, IntensityPredictor, compare_classifiers
from src.pipeline import EmotionPipeline

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "data", "Sample_arvyax_reflective_dataset.xlsx")
TEST_FILE = os.path.join(BASE_DIR, "data", "arvyax_test_inputs_120.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "test_predictions.csv")


def main():
    print("=" * 70)
    print("  EMOTION DETECTION AI - Training Pipeline")
    print("=" * 70)

    # -- 1. Load Training Data --
    print("\n[1/6] Loading training data...")
    train_df = pd.read_excel(TRAIN_FILE)
    print(f"       Loaded {len(train_df)} samples with columns: {list(train_df.columns)}")

    # -- 2. Preprocess --
    print("\n[2/6] Preprocessing text & features...")
    train_df = preprocess_dataframe(train_df, is_training=True)
    print(f"       Sample cleaned text: '{train_df['clean_text'].iloc[0]}'")

    # -- 3. Feature Engineering --
    print("\n[3/6] Extracting features (TF-IDF + categorical + numeric)...")
    fe = FeatureExtractor(max_tfidf_features=5000)
    X_all = fe.fit_transform(train_df)
    y_emotion = train_df["emotional_state"].values
    y_intensity = train_df["intensity"].values
    print(f"       Feature matrix shape: {X_all.shape}")

    # Train/Val split
    X_train, X_val, y_emo_train, y_emo_val, y_int_train, y_int_val = (
        train_test_split(
            X_all, y_emotion, y_intensity,
            test_size=0.2, random_state=42, stratify=y_emotion,
        )
    )
    print(f"       Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

    # -- 4. Train & Evaluate Models --
    print("\n[4/6] Training models...")

    # --- Emotion classifier ---
    print("\n  +-- Emotion Classifier (LinearSVC + Calibration)")
    emo_clf = EmotionClassifier()
    emo_clf.fit(X_train, y_emo_train)
    emo_results = emo_clf.evaluate(X_val, y_emo_val)
    print(f"  |   Accuracy: {emo_results['accuracy']:.4f}")
    print(f"  |\n  |   Classification Report:")
    for line in emo_results["report"].split("\n"):
        print(f"  |   {line}")
    print("  +--------------------------------------")

    # --- Model comparison ---
    print("\n  +-- Model Comparison")
    comparison = compare_classifiers(X_train, y_emo_train, X_val, y_emo_val)
    for m in comparison:
        print(f"  |   {m['model']}: {m['accuracy']:.4f}")
    print("  +--------------------------------------")

    # --- Intensity predictor ---
    print("\n  +-- Intensity Predictor (Ridge Regression)")
    int_model = IntensityPredictor()
    int_model.fit(X_train, y_int_train)
    int_results = int_model.evaluate(X_val, y_int_val)
    print(f"  |   RMSE: {int_results['rmse']:.4f}")
    print(f"  |   MAE : {int_results['mae']:.4f}")
    print("  +--------------------------------------")

    # -- 5. Save Models --
    print("\n[5/6] Saving models...")
    fe.save()
    emo_clf.save()
    int_model.save()
    print("       Models saved to models/ directory")

    # -- 6. Sample Predictions --
    print("\n[6/6] Running sample predictions...\n")
    pipeline = EmotionPipeline(fe, emo_clf, int_model)

    sample_texts = [
        ("I feel so overwhelmed, everything is piling up and I can't breathe.", "afternoon", 5, 2, 4.5),
        ("Today was peaceful. I meditated and feel at ease.", "morning", 1, 4, 8.0),
        ("I can't focus on anything, my mind keeps jumping everywhere.", "evening", 4, 2, 5.0),
        ("The meeting went fine, nothing special happened.", "afternoon", 2, 3, 7.0),
        ("I'm really happy today! Got great news about my project.", "morning", 1, 5, 8.0),
        ("I'm torn between feeling grateful and feeling guilty.", "night", 3, 3, 6.0),
        ("My anxiety won't let me sleep. Every noise startles me.", "night", 5, 1, 3.5),
        ("I finished all my tasks efficiently today. Feeling productive.", "evening", 1, 4, 7.5),
        ("I feel nothing. Just going through the motions.", "morning", 2, 2, 6.0),
        ("The rain session helped a little but I still feel restless inside.", "afternoon", 4, 3, 5.5),
    ]

    for i, (text, tod, stress, energy, sleep) in enumerate(sample_texts, 1):
        result = pipeline.predict(
            text,
            time_of_day=tod,
            stress_level=stress,
            energy_level=energy,
            sleep_hours=sleep,
        )
        print(f"  -- Sample {i} --")
        print(f"  Input : \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        print(f"  Emotion     : {result['emotion']}")
        print(f"  Intensity   : {result['intensity']} ({result['intensity_label']})")
        print(f"  Action      : {result['recommendation']}")
        print(f"  Timing      : {result['timing_label']} - {result['timing_detail']}")
        print()

    # -- 7. Predict on Test Set --
    print("=" * 70)
    print("  Running predictions on test dataset (120 samples)...")
    test_df = pd.read_excel(TEST_FILE)
    test_results = pipeline.predict_batch(test_df)
    test_results.to_csv(OUTPUT_FILE, index=False)
    print(f"  Results saved to: {OUTPUT_FILE}")
    print(f"\n  First 5 test predictions:")
    for _, row in test_results.head(5).iterrows():
        print(f"    [{row['emotion']}] Intensity={row['intensity']} | {row['recommendation'][:60]}...")
    print("=" * 70)
    print("  Pipeline complete! Run the Flask app: python app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
