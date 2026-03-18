"""
app.py — Streamlit UI for the Emotion Detection AI pipeline.

Run:   streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

# Ensure models exist before importing pipeline
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

from src.pipeline import EmotionPipeline


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0; font-size: 2rem; font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.05rem;
    }

    .result-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        border: 1px solid #e8eaf6;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }
    .result-card h3 {
        margin: 0 0 0.4rem 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9e9e9e;
        font-weight: 600;
    }
    .result-card .value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1a1a2e;
        line-height: 1.4;
    }

    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
    }

    .timing-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .intensity-bar {
        height: 10px;
        border-radius: 5px;
        background: #e0e0e0;
        margin-top: 0.5rem;
    }
    .intensity-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# ── Emotion color map ────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "Calm": ("#4caf50", "🌿"),
    "Focused": ("#2196f3", "🎯"),
    "Neutral": ("#9e9e9e", "😐"),
    "Restless": ("#ff9800", "🌀"),
    "Mixed": ("#9c27b0", "🎭"),
    "Overwhelmed": ("#f44336", "😰"),
}

TIMING_COLORS = {
    "Immediate": "#f44336",
    "Soon": "#ff9800",
    "Later Today": "#2196f3",
    "Whenever Ready": "#4caf50",
}

INTENSITY_COLORS = {
    "Low": "#4caf50",
    "Medium": "#ff9800",
    "High": "#f44336",
}


# ── Load Pipeline ────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load the trained pipeline (cached)."""
    return EmotionPipeline.load(MODELS_DIR)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Emotion Detection AI</h1>
    <p>Analyze emotions from journal text and get personalized, actionable recommendations</p>
</div>
""", unsafe_allow_html=True)

# Check if models exist
if not os.path.exists(os.path.join(MODELS_DIR, "emotion_classifier.joblib")):
    st.error("⚠️ Models not found! Please run `python main.py` first to train the models.")
    st.stop()

pipeline = load_pipeline()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["✍️ Single Analysis", "📊 Batch Prediction"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Single Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_context = st.columns([2, 1])

    with col_input:
        st.subheader("Journal Entry")
        text_input = st.text_area(
            "Write or paste your journal entry below:",
            height=180,
            placeholder="e.g., I feel so overwhelmed with everything piling up. My mind keeps racing and I can't seem to calm down...",
        )

    with col_context:
        st.subheader("Context (Optional)")
        time_of_day = st.selectbox(
            "Time of day",
            ["morning", "early_morning", "afternoon", "evening", "night"],
            index=0,
        )
        stress_level = st.slider("Stress level", 1, 5, 3)
        energy_level = st.slider("Energy level", 1, 5, 3)
        sleep_hours = st.slider("Sleep hours (last night)", 2.0, 12.0, 7.0, 0.5)

    # ── Analyze Button ───────────────────────────────────────────────────
    if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result = pipeline.predict(
                    text_input,
                    time_of_day=time_of_day,
                    stress_level=stress_level,
                    energy_level=energy_level,
                    sleep_hours=sleep_hours,
                )

            st.markdown("---")

            # ── Results Grid ─────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)

            emotion = result["emotion"]
            color, emoji = EMOTION_COLORS.get(emotion, ("#607d8b", "❓"))
            int_color = INTENSITY_COLORS.get(result["intensity_label"], "#9e9e9e")
            timing_color = TIMING_COLORS.get(result["timing_label"], "#607d8b")

            with c1:
                st.markdown(f"""
                <div class="result-card">
                    <h3>Detected Emotion</h3>
                    <div class="emotion-badge" style="background: {color};">
                        {emoji} {emotion}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                pct = (result["intensity"] - 1) / 4 * 100
                st.markdown(f"""
                <div class="result-card">
                    <h3>Intensity</h3>
                    <div class="value">{result['intensity']} / 5.0 — {result['intensity_label']}</div>
                    <div class="intensity-bar">
                        <div class="intensity-fill" style="width: {pct}%; background: {int_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="result-card">
                    <h3>Timing</h3>
                    <div class="timing-badge" style="background: {timing_color}; color: white;">
                        ⏰ {result['timing_label']}
                    </div>
                    <p style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
                        {result['timing_detail']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ── Recommendation Card ──────────────────────────────────────
            st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid {color}; margin-top: 1rem;">
                <h3>💡 Recommended Action</h3>
                <div class="value" style="font-size: 1.15rem;">
                    {result['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload a dataset for batch prediction")
    st.caption("The file should have a `journal_text` column (and optionally context columns).")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    use_test_set = st.checkbox("Or use the built-in test set (120 samples)")

    if st.button("🚀 Run Batch Prediction", type="primary"):
        df = None

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        elif use_test_set:
            test_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "arvyax_test_inputs_120.xlsx",
            )
            df = pd.read_excel(test_path)

        if df is None:
            st.warning("Please upload a file or select the built-in test set.")
        elif "journal_text" not in df.columns:
            st.error("The file must contain a `journal_text` column.")
        else:
            with st.spinner(f"Processing {len(df)} entries..."):
                results_df = pipeline.predict_batch(df)

            st.success(f"✅ Processed {len(results_df)} entries!")

            # Show summary stats
            st.markdown("### Emotion Distribution")
            emotion_counts = results_df["emotion"].value_counts()
            st.bar_chart(emotion_counts)

            # Show results table
            st.markdown("### Detailed Results")
            st.dataframe(
                results_df[["journal_text", "emotion", "intensity", "intensity_label",
                             "recommendation", "timing_label"]],
                use_container_width=True,
                height=400,
            )

            # Download button
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv_data,
                "emotion_predictions.csv",
                "text/csv",
            )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses machine learning to analyze emotions 
    from reflective journal entries.
    
    **Pipeline:**
    1. Text preprocessing (cleaning, lemmatization)
    2. Feature extraction (TF-IDF + context)
    3. Emotion classification (LinearSVC)
    4. Intensity prediction (Ridge Regression)
    5. Personalized recommendation engine
    
    **Emotions detected:**
    - 🌿 Calm
    - 🎯 Focused
    - 😐 Neutral
    - 🌀 Restless
    - 🎭 Mixed
    - 😰 Overwhelmed
    """)

    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    st.caption(f"Models directory: `{MODELS_DIR}`")
    model_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
    for f in model_files:
        size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
        st.caption(f"📦 `{f}` ({size_kb:.1f} KB)")
