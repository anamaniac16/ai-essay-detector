"""
app.py — Streamlit interface for AI Essay Detector.

Features:
  - Paste an essay, get sentence-level AI-likelihood highlighting
  - Click/hover flagged sentences to see driving feature values
  - ESL signal banner when appropriate
  - Not a black box: shows the actual features and reasoning
"""

import os
import sys
import json
import pickle

import streamlit as st
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector.features import (
    extract_features, extract_sentence_features, token_log_probs,
    split_sentences, FEATURE_NAMES
)
from detector.esl_signal import (
    extract_esl_features, esl_likelihood_score, esl_report,
    ESL_FEATURE_NAMES
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def load_model():
    """Load the trained classifier."""
    model_path = os.path.join(MODELS_DIR, "classifier.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_importances():
    """Load feature importances for explanation."""
    imp_path = os.path.join(MODELS_DIR, "feature_importances.json")
    if not os.path.exists(imp_path):
        return {}
    with open(imp_path, "r") as f:
        return json.load(f)


def get_sentence_ai_scores(text: str, model) -> list:
    """
    Compute per-sentence AI-likelihood contribution scores.
    Each sentence gets a score based on how its features deviate from
    what's typical for human writing.
    """
    sentence_feats = extract_sentence_features(text)

    if not sentence_feats:
        return []

    # Essay-level prediction
    essay_feats = extract_features(text)
    X_essay = np.array([[essay_feats[name] for name in FEATURE_NAMES]])
    X_essay = np.nan_to_num(X_essay, nan=0.0, posinf=100.0, neginf=-100.0)

    if hasattr(model, 'predict_proba'):
        essay_ai_prob = model.predict_proba(X_essay)[0][1]
    else:
        essay_ai_prob = float(model.predict(X_essay)[0])

    # Per-sentence scoring based on perplexity and feature deviation
    perplexities = [sf["perplexity"] for sf in sentence_feats]
    avg_perp = np.mean(perplexities) if perplexities else 50.0
    std_perp = np.std(perplexities) if len(perplexities) > 1 else 1.0

    results = []
    for sf in sentence_feats:
        # How much does this sentence's perplexity deviate from average?
        # Low perplexity (very predictable) = more AI-like
        if std_perp > 0:
            z_score = (sf["perplexity"] - avg_perp) / std_perp
        else:
            z_score = 0.0

        # Sentence-level AI-likelihood: combine essay score with sentence deviation
        # Low perplexity (negative z-score) pushes score higher (more AI-like)
        sentence_ai_score = essay_ai_prob * (1.0 - 0.2 * z_score)
        sentence_ai_score = max(0.0, min(1.0, sentence_ai_score))

        results.append({
            "sentence": sf["sentence"],
            "ai_score": sentence_ai_score,
            "perplexity": sf["perplexity"],
            "avg_log_prob": sf["avg_log_prob"],
            "log_prob_std": sf["log_prob_std"],
            "word_count": sf["word_count"],
            "function_word_ratio": sf["function_word_ratio"],
            "perplexity_z_score": z_score,
        })

    return results


def color_for_score(score: float) -> str:
    """Return a CSS color based on AI-likelihood score."""
    if score < 0.3:
        return "rgba(34, 197, 94, 0.15)"   # Green - likely human
    elif score < 0.5:
        return "rgba(250, 204, 21, 0.20)"   # Yellow - uncertain
    elif score < 0.7:
        return "rgba(251, 146, 60, 0.25)"   # Orange - suspicious
    else:
        return "rgba(239, 68, 68, 0.30)"    # Red - likely AI


def label_for_score(score: float) -> str:
    """Human-readable label for AI-likelihood score."""
    if score < 0.3:
        return "Likely Human"
    elif score < 0.5:
        return "Uncertain"
    elif score < 0.7:
        return "Suspicious"
    else:
        return "Likely AI"


def main():
    st.set_page_config(
        page_title="AI Essay Detector",
        page_icon="🔍",
        layout="wide",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .sentence-block {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        border-left: 4px solid;
        cursor: pointer;
        transition: all 0.2s;
    }
    .sentence-block:hover {
        transform: translateX(4px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .esl-banner {
        background: linear-gradient(135deg, #fef3c7, #fed7aa);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .feature-detail {
        background: #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85em;
        margin: 4px 0;
    }
    .methodology-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header"><h1>🔍 AI Essay Detector</h1></div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #64748b;'>"
        "Sentence-level AI text detection using GPT-2 perplexity analysis + statistical features. "
        "Not a chat-model wrapper — every signal is inspectable."
        "</p>",
        unsafe_allow_html=True
    )
    
    st.warning(
        "⚠️ **Dataset Limitation Warning**: The classifier is trained on a proxy dataset where the human-authored class "
        "is sourced from NLTK's Brown Corpus (academic prose) and the AI class is sourced from Gemini + GPT-2. "
        "It has NOT been validated on real student college admissions essays. Interpret scores accordingly."
    )

    # Load model
    model = load_model()
    importances = load_feature_importances()

    if model is None:
        st.error("⚠️ No trained model found. Run `train_classifier.py` first.")
        st.stop()

    # Methodology section (collapsible)
    with st.expander("📐 How This Works (Not a Black Box)", expanded=False):
        st.markdown("""
        **This detector does NOT ask a chat model "is this AI-written?"**

        Instead, it:
        1. **Runs your text through local GPT-2** (124M parameters) to get token-level log-probabilities
        2. **Extracts statistical features** from those probabilities:
           - Perplexity (how "surprised" GPT-2 is by the text)
           - Burstiness (variance of per-sentence perplexity — humans are spikier)
           - Sentence length patterns, function word ratio, POS-tag entropy
        3. **Feeds these features into a trained classifier** (Logistic Regression or XGBoost)
           trained on labeled human/AI essays
        4. **Shows you the evidence** so you can judge for yourself

        The ESL signal is a **separate, independent check** — it flags patterns common in
        non-native English writing that can cause false positives.
        """)

        if importances:
            st.markdown("**Feature Importances** (from trained model):")
            imp_df = pd.DataFrame([
                {"Feature": k, "Importance": v}
                for k, v in importances.items()
            ])
            st.dataframe(imp_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Input
    essay_text = st.text_area(
        "📝 Paste an essay below:",
        height=250,
        placeholder="Paste the essay you want to analyze here...",
    )

    analyze_btn = st.button("🔎 Analyze Essay", type="primary", use_container_width=True)

    if analyze_btn and essay_text.strip():
        with st.spinner("Analyzing... (running GPT-2 inference on CPU, this takes 10-30 seconds)"):
            # ── 1. Essay-level features ──
            essay_feats = extract_features(essay_text)
            X = np.array([[essay_feats[name] for name in FEATURE_NAMES]])
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

            if hasattr(model, 'predict_proba'):
                ai_probability = float(model.predict_proba(X)[0][1])
            else:
                ai_probability = float(model.predict(X)[0])

            prediction = "AI-Generated" if ai_probability >= 0.5 else "Human-Written"

            # ── 2. ESL check ──
            esl = esl_report(essay_text)

            # ── 3. Per-sentence analysis ──
            sentence_scores = get_sentence_ai_scores(essay_text, model)

        # ── Results Display ──
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            color = "#ef4444" if ai_probability >= 0.5 else "#22c55e"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {color};">{ai_probability:.1%}</div>
                <div class="metric-label">AI Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {'#ef4444' if prediction == 'AI-Generated' else '#22c55e'};">
                    {prediction.split('-')[0]}
                </div>
                <div class="metric-label">Prediction</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #3b82f6;">{essay_feats['essay_perplexity']:.1f}</div>
                <div class="metric-label">Perplexity</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #8b5cf6;">{essay_feats['burstiness_cv']:.3f}</div>
                <div class="metric-label">Burstiness (CV)</div>
            </div>
            """, unsafe_allow_html=True)

        # ESL Banner
        if esl["flag_active"]:
            st.markdown(f"""
            <div class="esl-banner">
                <strong>⚠️ ESL Signal Active (Score: {esl['esl_score']:.2f})</strong><br>
                {esl['message']}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 ESL Signal Details"):
                esl_df = pd.DataFrame([
                    {"Feature": k, "Value": f"{v:.4f}"}
                    for k, v in esl["features"].items()
                ])
                st.dataframe(esl_df, hide_index=True, use_container_width=True)

        # ── Sentence-level highlighting ──
        st.markdown("### 🔦 Sentence-Level Analysis")
        st.markdown(
            "<p style='color: #64748b; font-size: 0.9rem;'>"
            "Each sentence is colored by its contribution to the AI-likelihood score. "
            "Click a sentence to see the driving features."
            "</p>",
            unsafe_allow_html=True
        )

        # Color legend
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        lcol1.markdown("🟢 **< 30%** Likely Human")
        lcol2.markdown("🟡 **30-50%** Uncertain")
        lcol3.markdown("🟠 **50-70%** Suspicious")
        lcol4.markdown("🔴 **> 70%** Likely AI")

        # Render sentences
        for i, ss in enumerate(sentence_scores):
            bg_color = color_for_score(ss["ai_score"])
            border_color = (
                "#22c55e" if ss["ai_score"] < 0.3 else
                "#facc15" if ss["ai_score"] < 0.5 else
                "#fb923c" if ss["ai_score"] < 0.7 else
                "#ef4444"
            )

            with st.container():
                st.markdown(
                    f'<div class="sentence-block" '
                    f'style="background: {bg_color}; border-left-color: {border_color};">'
                    f'<strong>[{label_for_score(ss["ai_score"])} — {ss["ai_score"]:.0%}]</strong> '
                    f'{ss["sentence"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                with st.expander(f"📋 Sentence {i+1} — Feature Evidence", expanded=False):
                    detail_cols = st.columns(3)
                    with detail_cols[0]:
                        st.metric("Perplexity", f"{ss['perplexity']:.1f}",
                                  help="Lower = more predictable (AI-like)")
                    with detail_cols[1]:
                        st.metric("Avg Log-Prob", f"{ss['avg_log_prob']:.3f}",
                                  help="Higher (closer to 0) = more predictable")
                    with detail_cols[2]:
                        st.metric("Log-Prob StdDev", f"{ss['log_prob_std']:.3f}",
                                  help="Lower = more uniform (AI-like)")

                    detail_cols2 = st.columns(3)
                    with detail_cols2[0]:
                        st.metric("Word Count", ss["word_count"])
                    with detail_cols2[1]:
                        st.metric("Function Word %", f"{ss['function_word_ratio']:.1%}")
                    with detail_cols2[2]:
                        st.metric("Perplexity Z-Score", f"{ss['perplexity_z_score']:.2f}",
                                  help="Negative = below avg perplexity (more AI-like)")

                    # Interpretation
                    flags = []
                    if ss["perplexity"] < 30:
                        flags.append(f"🔴 Very low perplexity ({ss['perplexity']:.1f}) — text is highly predictable")
                    if ss["log_prob_std"] < 1.5:
                        flags.append(f"🟠 Low log-prob variance ({ss['log_prob_std']:.3f}) — unusually uniform token probabilities")
                    if ss["perplexity_z_score"] < -1.0:
                        flags.append(f"🟡 Below-average perplexity (z={ss['perplexity_z_score']:.2f}) — more predictable than the essay average")

                    if flags:
                        st.markdown("**Why this sentence is flagged:**")
                        for flag in flags:
                            st.markdown(f"- {flag}")
                    else:
                        st.markdown("✅ No strong AI signals in this sentence.")

        # ── Full Feature Table ──
        with st.expander("📊 Full Essay Feature Vector", expanded=False):
            feat_df = pd.DataFrame([
                {
                    "Feature": name,
                    "Value": f"{essay_feats[name]:.4f}",
                    "Importance": f"{importances.get(name, 0):.4f}" if importances else "N/A",
                }
                for name in FEATURE_NAMES
            ])
            st.dataframe(feat_df, hide_index=True, use_container_width=True)

    elif analyze_btn:
        st.warning("Please paste an essay to analyze.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>"
        "AI Essay Detector — Built with GPT-2 (local, CPU), scikit-learn, and Streamlit. "
        "Not a chat-model wrapper. All signals are inspectable."
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
