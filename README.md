# 🧠 Emotion Detection AI

An end-to-end Machine Learning pipeline that analyzes human emotions from reflective journal text, predicts emotional intensity, and provides personalized, actionable recommendations with timing suggestions.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Model Choices](#model-choices)
- [Pipeline Design](#pipeline-design)
- [Decision Engine Logic](#decision-engine-logic)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Example Output](#example-output)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Emotion Classification** | Detects 6 emotional states: calm, focused, neutral, restless, mixed, overwhelmed |
| **Intensity Prediction** | Predicts intensity on a 1–5 scale (Low / Medium / High) |
| **Actionable Recommendations** | Context-aware, human-centric suggestions based on emotion + intensity |
| **Timing Suggestions** | Urgency-scored timing: Immediate / Soon / Later Today / Whenever Ready |
| **Streamlit UI** | Interactive web interface for single and batch analysis |
| **Fully Offline** | No cloud APIs, no LLMs — runs 100% locally |

---

## 📁 Project Structure

```
Emotion detection AI/
├── data/
│   ├── Sample_arvyax_reflective_dataset.xlsx   # Training data (1,200 samples)
│   ├── arvyax_test_inputs_120.xlsx             # Test data (120 samples)
│   └── test_predictions.csv                    # Generated predictions
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Text cleaning, tokenization, stopword removal
│   ├── feature_engineering.py    # TF-IDF + categorical + numeric features
│   ├── models.py                 # Emotion classifier & intensity predictor
│   ├── decision_engine.py        # Recommendation & timing engine
│   └── pipeline.py               # End-to-end predict() function
├── models/                       # Saved trained models (generated)
│   ├── emotion_classifier.joblib
│   ├── intensity_predictor.joblib
│   └── feature_extractor.joblib
├── main.py                       # Training, evaluation, sample predictions
├── app.py                        # Streamlit web UI
├── requirements.txt
└── README.md
```

---

## 🧪 Approach

### 1. Data Understanding
The training dataset contains **1,200 reflective journal entries** with:
- **Text**: Free-form journal entries about mindfulness/ambience sessions
- **Contextual features**: stress level, energy level, sleep hours, time of day, ambience type
- **Labels**: emotional state (6 classes) and intensity (1–5)

### 2. Text Preprocessing
- **Lowercasing** all text
- **URL removal** (regex)
- **Special character stripping** (keep only alphabetic chars)
- **Tokenization** using NLTK's `word_tokenize`
- **Stopword removal** using NLTK's English stopwords
- **Lemmatization** using WordNet lemmatizer
- **Missing value imputation** for sleep_hours, previous_day_mood, face_emotion_hint

### 3. Feature Engineering
- **TF-IDF vectorization**: unigrams + bigrams, max 5,000 features, sublinear TF
- **Categorical encoding**: LabelEncoder for ambience_type, time_of_day, previous_day_mood, face_emotion_hint, reflection_quality
- **Numeric features**: StandardScaler on duration_min, sleep_hours, energy_level, stress_level
- All combined into a single **sparse feature matrix**

---

## 🤖 Model Choices

### Emotion Classifier — LinearSVC + CalibratedClassifierCV
- **Why LinearSVC?** Excellent performance on high-dimensional sparse text data (TF-IDF), fast training, memory-efficient
- **Why Calibration?** Enables `predict_proba()` for confidence scores, uses 3-fold cross-validation for calibration
- **Class balancing**: `class_weight="balanced"` handles any class imbalance
- **Alternatives tested**: Logistic Regression (comparison printed during training)

### Intensity Predictor — Ridge Regression
- **Why Ridge?** Handles sparse features (TF-IDF) well with L2 regularization, treats intensity as continuous (1–5) for smoother predictions
- **Post-processing**: Predictions clamped to [1, 5] range

---

## ⚙️ Pipeline Design

```
                     ┌──────────────┐
                     │  Raw Text +  │
                     │   Context    │
                     └──────┬───────┘
                            │
                    ┌───────▼────────┐
                    │ Preprocessing  │  Clean text, handle missing values
                    └───────┬────────┘
                            │
                  ┌─────────▼──────────┐
                  │ Feature Extraction │  TF-IDF + categorical + numeric
                  └─────────┬──────────┘
                            │
               ┌────────────┼────────────┐
               │                         │
      ┌────────▼─────────┐    ┌──────────▼──────────┐
      │ Emotion Classifier│    │ Intensity Predictor │
      │   (LinearSVC)     │    │   (Ridge)           │
      └────────┬─────────┘    └──────────┬──────────┘
               │                         │
               └────────────┬────────────┘
                            │
                  ┌─────────▼──────────┐
                  │  Decision Engine   │  Urgency scoring → recommendation
                  └─────────┬──────────┘
                            │
                    ┌───────▼────────┐
                    │   Final Output │  Emotion + Intensity + Action + Timing
                    └────────────────┘
```

---

## 🎯 Decision Engine Logic

The recommendation engine is **not** simple if-else. It uses a **multi-factor urgency scoring system**:

### Urgency Score Computation

```
urgency = 0.30 × base_weight        (recommendation-specific)
        + 0.30 × intensity_norm     (1–5 → 0–1)
        + 0.20 × stress_norm        (1–5 → 0–1)
        + 0.20 × fatigue_factor     (energy + sleep quality)
```

### Selection Strategy
- **High intensity + negative emotion** → picks highest-urgency recommendation
- **Low intensity + positive emotion** → picks lowest-urgency (gentle suggestion)
- **Otherwise** → picks the most balanced (mid-urgency) option

### Timing Rules

| Urgency Score | Timing Label | Example |
|---------------|-------------|---------|
| ≥ 0.70 | **Immediate** | "Right now — take a pause before anything else" |
| ≥ 0.50 | **Soon** | "Within the next hour, before your day gets busier" |
| ≥ 0.30 | **Later Today** | "Schedule this for your lunch break" |
| < 0.30 | **Whenever Ready** | "Tomorrow morning could be lovely for this" |

### Contextual Adjustments
- **Poor sleep** (< 5h): Adds a rest reminder
- **High stress + low energy**: Adds a self-compassion note

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Navigate to the project directory
cd "Emotion detection AI"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models and run evaluation
python main.py

# 4. Launch the Streamlit web app
streamlit run app.py
```

---

## 💻 Usage

### Command Line — `predict()` Function

```python
from src.pipeline import EmotionPipeline

pipeline = EmotionPipeline.load()

result = pipeline.predict(
    text="I feel so overwhelmed, everything is piling up",
    time_of_day="afternoon",
    stress_level=5,
    energy_level=2,
    sleep_hours=4.5,
)

print(result)
# {
#     'emotion': 'Overwhelmed',
#     'intensity': 4.32,
#     'intensity_label': 'High',
#     'recommendation': 'Splash cold water on your face — ...',
#     'timing_label': 'Immediate',
#     'timing_detail': 'Right now — take a pause before doing anything else.',
#     'urgency_score': 0.825
# }
```

### Streamlit Web App

```bash
streamlit run app.py
```

- **Single Analysis tab**: Enter journal text + optional context → see results
- **Batch Prediction tab**: Upload CSV/Excel or use the built-in test set

---

## 📊 Example Output

| Input Text | Emotion | Intensity | Action | Timing |
|------------|---------|-----------|--------|--------|
| "I feel so overwhelmed, everything piling up..." | Overwhelmed | 4.3 (High) | Splash cold water on your face — calming reflex | Immediate |
| "Today was peaceful. I meditated and feel at ease." | Calm | 1.8 (Low) | Read something inspiring or listen to a saved podcast | Whenever Ready |
| "I can't focus, my mind keeps jumping everywhere." | Restless | 3.9 (High) | Go for a brisk 10-minute walk | Soon |
| "The meeting went fine, nothing special." | Neutral | 2.1 (Low) | Reflect on one thing that went well today | Whenever Ready |

---

## 📝 Notes

- All models run **locally** — no internet or API calls required
- The training dataset has **1,200 labeled samples** with 6 emotion classes
- The test set has **120 unlabeled samples** — predictions are saved to `data/test_predictions.csv`
- Model files are saved in `models/` directory after running `main.py`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| NLP | NLTK |
| Text Features | TF-IDF |
| UI | Streamlit |
| Data | pandas, openpyxl |
