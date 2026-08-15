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


def get_sentence_ai_scores(text: str, model, essay_ai_prob: float = None) -> list:
    """
    Compute per-sentence AI-likelihood contribution scores.
    Each sentence gets a score based on how its features deviate from
    what's typical for human writing.
    """
    sentence_feats = extract_sentence_features(text)

    if not sentence_feats:
        return []

    # If overall essay_ai_prob is not provided, extract features from model
    if essay_ai_prob is None:
        essay_feats = extract_features(text)
        FEATURE_NAMES_NO_NUM = [f for f in FEATURE_NAMES if f != "num_sentences"]
        X_essay = np.array([[essay_feats[name] for name in FEATURE_NAMES_NO_NUM]])
        X_essay = np.nan_to_num(X_essay, nan=0.0, posinf=100.0, neginf=-100.0)

        if hasattr(model, 'predict_proba'):
            essay_ai_prob = float(model.predict_proba(X_essay)[0][1])
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
        sentence_ai_score = essay_ai_prob * (1.0 - 0.2 * z_score)
        sentence_ai_score = max(0.0, min(1.0, sentence_ai_score))

        # Guardrail: For human-classified text (essay_ai_prob < 0.50), individual sentences should never exceed overall essay_ai_prob
        if essay_ai_prob < 0.50:
            sentence_ai_score = min(sentence_ai_score, essay_ai_prob)

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


def color_for_score(score: float, threshold: float = 0.50) -> str:
    """Return a CSS color based on AI-likelihood score."""
    if score < 0.35:
        return "rgba(34, 197, 94, 0.15)"   # Green - Likely Human
    elif score < 0.55:
        return "rgba(250, 204, 21, 0.20)"   # Yellow - Uncertain
    elif score < 0.75:
        return "rgba(251, 146, 60, 0.25)"   # Orange - Likely AI
    else:
        return "rgba(239, 68, 68, 0.30)"    # Red - Highly Confirmed AI


def label_for_score(score: float, threshold: float = 0.50) -> str:
    """Human-readable label for AI-likelihood score."""
    if score < 0.35:
        return "Likely Human"
    elif score < 0.55:
        return "Uncertain"
    elif score < 0.75:
        return "Likely AI"
    else:
        return "Highly Confirmed AI"



def main():
    st.set_page_config(
        page_title="AI Essay Detector",
        page_icon=None,
        layout="wide",
    )

    # Custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Target fonts selectively to avoid breaking Material Icons/SVGs */
    .stApp, .main-header h1, .metric-box, .metric-value, .metric-label, .sentence-block, .esl-banner, .methodology-box, .document-editor {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    
    /* Premium Card Design for Metrics (Theme-Agnostic / Dark Mode Friendly) */
    .metric-box {
        background: rgba(128, 128, 128, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: inherit;
    }
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.15), 0 10px 10px -5px rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    
    /* Grammarly-style Document Editor Box */
    .document-editor {
        background: rgba(128, 128, 128, 0.02);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 20px;
        padding: 2.25rem;
        margin: 1.5rem 0;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.15);
        max-height: 550px;
        overflow-y: auto;
        color: inherit;
    }
    
    /* Inline highlighted sentences (Grammarly style) */
    .inline-sentence {
        padding: 3px 6px;
        margin: 0 1px;
        border-radius: 6px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        display: inline;
    }
    .inline-sentence:hover {
        background: rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    
    /* Custom Input Textarea styling (supporting light/dark modes) */
    .stTextArea textarea {
        border-radius: 16px !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
        padding: 1.25rem !important;
        transition: all 0.25s ease !important;
        background-color: rgba(128, 128, 128, 0.02) !important;
        color: inherit !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        background-color: rgba(128, 128, 128, 0.04) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Premium Gradient Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: -0.01em;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.45) !important;
        background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(0) !important;
    }
    
    /* ESL and Info Banners (styled for dark mode readability) */
    .esl-banner {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.05), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin: 1.5rem 0;
        color: #f59e0b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .feature-detail {
        background: rgba(0, 0, 0, 0.25);
        color: #cbd5e1;
        border-radius: 10px;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85em;
        margin: 6px 0;
        border-left: 3px solid #6366f1;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .methodology-box {
        background: rgba(34, 197, 94, 0.03);
        border: 1px solid rgba(34, 197, 94, 0.15);
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin: 1.5rem 0;
        color: #4ade80;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header"><h1>AI Essay Detector</h1></div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #64748b;'>"
        "Sentence-level AI text detection using GPT-2 perplexity analysis + statistical features. "
        "Not a chat-model wrapper — every signal is inspectable."
        "</p>",
        unsafe_allow_html=True
    )
    
    st.warning(
        "**Dataset Limitation Warning**: The classifier is trained on a proxy dataset where the human-authored class "
        "is sourced from NLTK's Brown Corpus (academic prose) and the AI class is sourced from Gemini + GPT-2. "
        "It has NOT been validated on real student college admissions essays. Interpret scores accordingly."
    )

    # Load model
    model = load_model()
    importances = load_feature_importances()

    if model is None:
        st.error("No trained model found. Run `train_classifier.py` first.")
        st.stop()

    # Sidebar settings
    st.sidebar.markdown("### Classifier Sensitivity")
    threshold = st.sidebar.slider(
        "AI Detection Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Higher values reduce false positives (flagging human text as AI) by requiring stronger statistical evidence to predict AI-Generated."
    )
    
    st.sidebar.markdown(f"""
    **Current Threshold: {threshold:.0%}**
    
    **Why do false positives happen?**
    Statistical detectors look for grammatical predictability (low perplexity) and uniform sentence length. 
    Formal, structured, or academic human writing naturally has lower perplexity and can mimic these patterns, causing false flags. 
    
    *Adjust this slider to **0.75 - 0.85** to protect formal human writers from false alarms.*
    """)

    # Methodology section (collapsible)
    with st.expander("How This Works (Not a Black Box)", expanded=False):
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
        "Paste an essay below:",
        height=250,
        placeholder="Paste the essay you want to analyze here...",
    )

    analyze_btn = st.button("Analyze Essay", type="primary", use_container_width=True)

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

            # ── 2. ESL check ──
            esl = esl_report(essay_text)

            # ── 3. Bias Compensation & Human Protection ──
            # Rule A: High perplexity (> 50.0) is a strong human signal (AI generators stay under 30-40)
            if essay_feats['essay_perplexity'] > 50.0:
                perp_discount = min(0.65, (essay_feats['essay_perplexity'] - 50.0) / 100.0)
                ai_probability = max(0.05, ai_probability - perp_discount)

            # Rule B: ESL non-native writing patterns mimic low perplexity to statistical classifiers
            if esl["flag_active"]:
                esl_discount = min(0.50, esl["esl_score"] * 1.2)
                ai_probability = max(0.05, ai_probability - esl_discount)

            # Rule C: First-person pronouns (I, me, my, we, our) indicate genuine human voice
            if essay_feats.get('first_person_pronoun_ratio', 0.0) > 0.015:
                pronoun_discount = min(0.40, essay_feats['first_person_pronoun_ratio'] * 5.0)
                ai_probability = max(0.05, ai_probability - pronoun_discount)

            # Rule D: Contractions (don't, it's, can't) indicate natural human conversational writing
            if essay_feats.get('contraction_ratio', 0.0) > 0.01:
                contraction_discount = min(0.30, essay_feats['contraction_ratio'] * 4.0)
                ai_probability = max(0.05, ai_probability - contraction_discount)

            # Rule E: Short-text uncertainty attenuation (< 80 words)
            word_count = len(essay_text.split())
            if word_count < 80 and ai_probability < 0.85:
                # Scale probability toward lower/neutral values for short inputs
                ai_probability = ai_probability * (word_count / 80.0)

            prediction = "AI-Generated" if ai_probability >= threshold else "Human-Written"


            # ── 4. Per-sentence analysis ──
            # Pass final adjusted ai_probability to ensure sentence highlights match essay score
            sentence_scores = get_sentence_ai_scores(essay_text, model, ai_probability)

        # ── Results Display ──
        st.markdown("---")
        st.markdown("## Analysis Results")

        # Clamp low human scores to 0.0% for clean UI display
        display_prob = 0.0 if ai_probability <= 0.08 else ai_probability

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            color = "#ef4444" if ai_probability >= threshold else "#22c55e"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {color};">{display_prob:.1%}</div>
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
                <strong>ESL Bias Compensated (Score: {esl['esl_score']:.2f})</strong><br>
                This essay shows non-native English writing patterns. The raw AI likelihood score was automatically 
                discounted to compensate for ESL syntax bias and prevent false accusations against human writers.
            </div>
            """, unsafe_allow_html=True)

            with st.expander("ESL Signal Details"):
                esl_df = pd.DataFrame([
                    {"Feature": k, "Value": f"{v:.4f}"}
                    for k, v in esl["features"].items()
                ])
                st.dataframe(esl_df, hide_index=True, use_container_width=True)

        # ── Sentence-level highlighting ──
        st.markdown("### Sentence-Level Analysis")
        st.markdown(
            "<p style='color: #64748b; font-size: 0.9rem;'>"
            "Each sentence is colored by its contribution to the AI-likelihood score. "
            "Click a sentence to see the driving features."
            "</p>",
            unsafe_allow_html=True
        )

        # Color legend
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        lcol1.markdown(f"**[< {0.5*threshold:.0%}]** Likely Human")
        lcol2.markdown(f"**[{0.5*threshold:.0%}-{0.8*threshold:.0%}]** Uncertain")
        lcol3.markdown(f"**[{0.8*threshold:.0%}-{threshold:.0%}]** Suspicious")
        lcol4.markdown(f"**[> {threshold:.0%}]** Likely AI")

        # Render inline paragraph document (Grammarly style)
        paragraphs = [p.strip() for p in essay_text.split("\n\n") if p.strip()]
        html_content = '<div class="document-editor">'
        
        for p in paragraphs:
            html_content += '<p style="margin-bottom: 1.25rem; line-height: 1.8; font-size: 1.05rem;">'
            p_sentences = split_sentences(p)
            for s in p_sentences:
                match_score = None
                for score_info in sentence_scores:
                    if score_info["sentence"].strip() == s.strip():
                        match_score = score_info
                        break
                
                if match_score:
                    score = match_score["ai_score"]
                    bg_color = color_for_score(score, threshold)
                    border_color = (
                        "#22c55e" if score < 0.5 * threshold else
                        "#facc15" if score < 0.8 * threshold else
                        "#fb923c" if score < threshold else
                        "#ef4444"
                    )
                    html_content += (
                        f'<span class="inline-sentence" '
                        f'style="background: {bg_color}; border-bottom: 2px solid {border_color};" '
                        f'title="{label_for_score(score, threshold)} ({score:.0%})">{s}</span> '
                    )
                else:
                    html_content += f'<span>{s}</span> '
            html_content += '</p>'
        html_content += '</div>'
        
        st.markdown(html_content, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed expandable list below
        st.markdown("### Detailed Sentence Evidence")
        st.markdown(
            "<p style='color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem;'>"
            "Click any sentence below to analyze its statistical features and perplexity metrics."
            "</p>",
            unsafe_allow_html=True
        )

        for i, ss in enumerate(sentence_scores):
            preview = ss["sentence"][:80] + "..." if len(ss["sentence"]) > 80 else ss["sentence"]
            ai_lbl = label_for_score(ss["ai_score"], threshold)
            prob_pct = f"{ss['ai_score']:.0%}"
            
            with st.expander(f"Sentence {i+1} [{ai_lbl} — {prob_pct}]: \"{preview}\"", expanded=False):
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
                    flags.append(f"Very low perplexity ({ss['perplexity']:.1f}) — text is highly predictable")
                if ss["log_prob_std"] < 1.5:
                    flags.append(f"Low log-prob variance ({ss['log_prob_std']:.3f}) — unusually uniform token probabilities")
                if ss["perplexity_z_score"] < -1.0:
                    flags.append(f"Below-average perplexity (z={ss['perplexity_z_score']:.2f}) — more predictable than the essay average")

                if flags:
                    st.markdown("**Why this sentence is flagged:**")
                    for flag in flags:
                        st.markdown(f"- {flag}")
                else:
                    st.markdown("No strong AI signals in this sentence.")

        # ── Full Feature Table ──
        with st.expander("Full Essay Feature Vector", expanded=False):
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
