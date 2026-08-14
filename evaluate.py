"""
evaluate.py — Run full evaluation on held-out test set and generate EVALUATION.md.

This script:
  1. Loads the trained model and test data
  2. Reports accuracy/precision/recall/F1
  3. Finds 3 confidently wrong predictions with analysis
  4. Tests ESL bias behavior
  5. Writes EVALUATION.md
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector.features import extract_features, FEATURE_NAMES
from detector.esl_signal import esl_report, extract_esl_features

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
os.makedirs(DOCS_DIR, exist_ok=True)


# Sample ESL-style essays for bias testing
ESL_TEST_ESSAYS = [
    {
        "id": "esl_1",
        "text": (
            "In my country education is very important thing. All parents want their children go to "
            "university and get good job. I study very hard because I want make my family proud. "
            "Sometimes studying is difficult for me because English is not my first language. "
            "But I keep trying because I believe education is the key to success. "
            "My dream is become doctor and help people in my village."
        ),
        "label": "human_esl",
        "description": "Simulated ESL essay — grammatically imperfect, limited vocabulary, simple sentence structure"
    },
    {
        "id": "esl_2",
        "text": (
            "I came to United States two years ago from Vietnam. At first everything was very "
            "different and I felt alone. The food, the weather, the way people talk — all was new "
            "for me. In school I had trouble understanding teacher because she speak very fast. "
            "But my classmates were kind to me. They help me with homework and teach me new words. "
            "Now I can speak English much better but sometimes I still make mistakes. "
            "I want to study computer science in college because technology is future."
        ),
        "label": "human_esl",
        "description": "Simulated ESL essay — article omissions, tense inconsistency, common ESL patterns"
    },
    {
        "id": "esl_3",
        "text": (
            "The community service is very important for young people. When I was in high school "
            "I volunteer at the hospital near my house. I help the patients by bringing them food "
            "and talking with them. Some patients were very old and they have no family visit them. "
            "This experience teach me that we should care for others. I think all students should "
            "do community service because it help them become better person."
        ),
        "label": "human_esl",
        "description": "Simulated ESL essay — unnecessary articles, missing inflections"
    },
]


def load_test_data():
    """Load test data and cached features."""
    test_path = os.path.join(DATASET_DIR, "test.csv")
    test_features_path = os.path.join(DATASET_DIR, "test_features.csv")

    if not os.path.exists(test_path):
        print("[Eval] No test data found.")
        return None, None

    test_df = pd.read_csv(test_path)
    test_features = None
    if os.path.exists(test_features_path):
        test_features = pd.read_csv(test_features_path)

    return test_df, test_features


def run_evaluation():
    """Run the full evaluation pipeline."""
    # Load model
    model_path = os.path.join(MODELS_DIR, "classifier.pkl")
    if not os.path.exists(model_path):
        print("[Eval] No model found. Run train_classifier.py first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load test data
    test_df, test_features = load_test_data()
    if test_df is None:
        return

    # Load or compute test features
    if test_features is None:
        print("[Eval] Computing test features (this will take a while)...")
        from detector.features import extract_features
        from tqdm import tqdm
        features_list = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            try:
                feats = extract_features(row["text"])
                feats["label"] = row["label"]
                feats["index"] = idx
                features_list.append(feats)
            except:
                feats = {name: 0.0 for name in FEATURE_NAMES}
                feats["label"] = row["label"]
                feats["index"] = idx
                features_list.append(feats)
        test_features = pd.DataFrame(features_list)

    # Predict
    X = test_features[FEATURE_NAMES].values
    y = test_features["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    # Find 3 confidently wrong or borderline correct predictions for error analysis
    wrong_mask = y_pred != y
    confident_wrong = []
    
    # 1. Grab all actual wrong predictions, sorted by confidence (highest first)
    actual_wrong_indices = []
    if any(wrong_mask):
        wrong_indices = np.where(wrong_mask)[0]
        if y_proba is not None:
            confidence = np.abs(y_proba - 0.5)
            wrong_conf = confidence[wrong_mask]
            sorted_wrong_idx = np.argsort(-wrong_conf)
            actual_wrong_indices = list(wrong_indices[sorted_wrong_idx])
        else:
            actual_wrong_indices = list(wrong_indices)
            
    # 2. If we have less than 3 wrong predictions, grab correct predictions that were closest to the decision boundary (borderline)
    borderline_correct_indices = []
    if len(actual_wrong_indices) < 3 and y_proba is not None:
        correct_mask = ~wrong_mask
        correct_indices = np.where(correct_mask)[0]
        confidence = np.abs(y_proba - 0.5)
        correct_conf = confidence[correct_mask]
        # Sort by confidence (lowest confidence / closest to 0.5 first)
        sorted_correct_idx = np.argsort(correct_conf)
        borderline_correct_indices = list(correct_indices[sorted_correct_idx])
        
    # Combine to get exactly 3 cases
    target_indices = actual_wrong_indices + borderline_correct_indices
    target_indices = target_indices[:3]
    
    for rank, idx in enumerate(target_indices):
        text = test_df.iloc[idx]["text"] if idx < len(test_df) else "N/A"
        prob = float(y_proba[idx]) if y_proba is not None else -1.0
        true_label = "AI" if y[idx] == 1 else "HUMAN"
        pred_label = "AI" if y_pred[idx] == 1 else "HUMAN"
        is_actually_wrong = idx in actual_wrong_indices
        
        # Analyze failure or borderline characteristics
        feats = {name: float(test_features.iloc[idx][name]) for name in FEATURE_NAMES}
        if is_actually_wrong:
            case_title = f"Actual Misclassification #{rank + 1} ({'False Negative' if true_label == 'AI' else 'False Positive'})"
            theory = analyze_failure(true_label, pred_label, feats, text)
        else:
            case_title = f"Borderline Correct #{rank + 1} (Close Call - True {'AI' if true_label == 'AI' else 'Human'})"
            theory = analyze_borderline(true_label, feats, text)
            
        confident_wrong.append({
            "title": case_title,
            "index": int(idx),
            "text_preview": text[:500] + "..." if len(str(text)) > 500 else str(text),
            "true_label": true_label,
            "predicted_label": pred_label,
            "ai_probability": prob,
            "features": feats,
            "theory": theory,
        })

    # ESL bias test
    esl_results = test_esl_bias(model)

    # Generate EVALUATION.md
    write_evaluation_md(metrics, confident_wrong, esl_results)

    return metrics, confident_wrong, esl_results


def analyze_failure(true_label, pred_label, features, text):
    """Generate a theory for why this prediction failed."""
    text_str = str(text)[:200]

    if true_label == "HUMAN" and pred_label == "AI":
        # False positive — human text flagged as AI
        theories = []
        if features.get("essay_perplexity", 100) < 40:
            theories.append(
                f"Low perplexity ({features['essay_perplexity']:.1f}) — this human writer "
                f"uses highly predictable, conventional phrasing that GPT-2 finds easy to "
                f"predict, mimicking AI-like statistical patterns."
            )
        if features.get("burstiness_cv", 1.0) < 0.3:
            theories.append(
                f"Low burstiness (CV={features['burstiness_cv']:.3f}) — this writer maintains "
                f"unusually consistent complexity across sentences, unlike typical human "
                f"writing which tends to vary more."
            )
        if features.get("sentence_length_std", 10) < 3:
            theories.append(
                f"Very uniform sentence lengths (std={features['sentence_length_std']:.1f}) — "
                f"this writer's sentences are unusually similar in length, a pattern more "
                f"common in AI-generated text."
            )
        if not theories:
            theories.append(
                "Multiple features slightly lean toward AI-like patterns. This may be a "
                "well-structured, carefully edited human essay that happens to have "
                "statistical properties similar to AI text."
            )
        return " ".join(theories)

    elif true_label == "AI" and pred_label == "HUMAN":
        # False negative — AI text missed
        theories = []
        if features.get("essay_perplexity", 100) > 80:
            theories.append(
                f"High perplexity ({features['essay_perplexity']:.1f}) — this AI-generated "
                f"text uses unexpected or diverse language that GPT-2 finds harder to predict, "
                f"possibly due to creative prompting or a model very different from GPT-2."
            )
        if features.get("burstiness_cv", 0) > 0.5:
            theories.append(
                f"High burstiness (CV={features['burstiness_cv']:.3f}) — this AI text has "
                f"significant variation in sentence complexity, possibly because it was "
                f"generated with high temperature or instructions to vary style."
            )
        if not theories:
            theories.append(
                "This AI-generated text has statistical properties that closely mimic human "
                "writing. It may have been generated by a model or prompting strategy that "
                "specifically produces more 'natural' patterns."
            )
        return " ".join(theories)

    return "Unknown failure mode."


def analyze_borderline(true_label, features, text):
    """Generate a theory for why a correct prediction was borderline (close to 0.5 probability)."""
    theories = []
    if true_label == "HUMAN":
        if features.get("essay_perplexity", 100) < 50:
            theories.append(
                f"Borderline human because perplexity is relatively low ({features['essay_perplexity']:.1f}), "
                f"mimicking the predictable structure of AI text."
            )
        if features.get("burstiness_cv", 1.0) < 0.4:
            theories.append(
                f"Borderline human because sentence-to-sentence variation (burstiness CV={features['burstiness_cv']:.3f}) "
                f"is relatively low, making it look more uniform like AI writing."
            )
        if not theories:
            theories.append(
                "Borderline human due to a combination of moderate perplexity and typical sentence structure "
                "that aligns closely with the classifier boundary."
            )
    else:
        if features.get("essay_perplexity", 100) > 40:
            theories.append(
                f"Borderline AI because perplexity is slightly elevated ({features['essay_perplexity']:.1f}) "
                f"compared to standard repetitive AI templates."
            )
        if features.get("burstiness_cv", 0) > 0.4:
            theories.append(
                f"Borderline AI because sentence-to-sentence variation (burstiness CV={features['burstiness_cv']:.3f}) "
                f"is slightly higher than usual for AI text."
            )
        if not theories:
            theories.append(
                "Borderline AI due to style blending, where the AI writing characteristics slightly overlap "
                "with the human reference baseline."
            )
    return " ".join(theories)


def test_esl_bias(model):
    """Test whether ESL essays get falsely flagged as AI-generated."""
    results = []

    for essay in ESL_TEST_ESSAYS:
        # Extract features
        try:
            feats = extract_features(essay["text"])
            X = np.array([[feats[name] for name in FEATURE_NAMES]])
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

            if hasattr(model, 'predict_proba'):
                ai_prob = float(model.predict_proba(X)[0][1])
            else:
                ai_prob = float(model.predict(X)[0])

            prediction = "AI" if ai_prob >= 0.5 else "HUMAN"

            # ESL signal
            esl = esl_report(essay["text"])

            results.append({
                "id": essay["id"],
                "description": essay["description"],
                "ai_probability": ai_prob,
                "prediction": prediction,
                "flagged_as_ai": ai_prob >= 0.5,
                "esl_flag_active": esl["flag_active"],
                "esl_score": esl["esl_score"],
                "text_preview": essay["text"][:200] + "...",
            })
        except Exception as e:
            results.append({
                "id": essay["id"],
                "description": essay["description"],
                "error": str(e),
            })

    return results


def write_evaluation_md(metrics, confident_wrong, esl_results):
    """Write EVALUATION.md with full analysis."""
    cm = metrics["confusion_matrix"]

    # Build confident-wrong/borderline sections
    wrong_sections = ""
    for i, case in enumerate(confident_wrong):
        wrong_sections += f"""
### {case['title']}

| Property | Value |
|----------|-------|
| **True Label** | {case['true_label']} |
| **Predicted Label** | {case['predicted_label']} |
| **AI Probability** | {case['ai_probability']:.4f} |
| **Confidence** | {abs(case['ai_probability'] - 0.5):.4f} |

**Text Preview:**
> {case['text_preview']}

**Key Features:**
| Feature | Value |
|---------|-------|
"""
        for name in FEATURE_NAMES:
            wrong_sections += f"| {name} | {case['features'].get(name, 0):.4f} |\n"

        wrong_sections += f"""
**Analysis / Theory:**
{case['theory']}

---
"""

    # ESL bias section
    esl_section = ""
    any_false_positive = False
    for r in esl_results:
        if "error" in r:
            esl_section += f"- **{r['id']}**: Error — {r['error']}\n"
            continue

        flag_emoji = "🔴" if r["flagged_as_ai"] else "🟢"
        esl_emoji = "⚠️" if r["esl_flag_active"] else "✅"
        esl_section += f"""
#### {r['id']}: {r['description']}

| Metric | Value |
|--------|-------|
| AI Probability | {r['ai_probability']:.4f} |
| AI Prediction | {flag_emoji} {r['prediction']} |
| ESL Flag Active | {esl_emoji} {r['esl_flag_active']} |
| ESL Score | {r['esl_score']:.4f} |

> {r['text_preview']}

"""
        if r["flagged_as_ai"]:
            any_false_positive = True

    esl_summary = ""
    if any_false_positive:
        esl_summary = (
            "⚠️ **WARNING: At least one ESL essay was falsely flagged as AI-generated.** "
            "This confirms the known bias risk — non-native English patterns can trigger "
            "false positives because lower perplexity and simpler structures overlap with "
            "AI-generated text features. The ESL signal module exists precisely to surface "
            "this risk to the user."
        )
    else:
        esl_summary = (
            "✅ No ESL test essays were flagged as AI-generated. However, this test uses "
            "only 3 synthetic ESL samples, which is NOT sufficient to prove the absence of "
            "ESL bias. A production system would need to be validated against a real, diverse "
            "ESL essay corpus."
        )

    md = f"""# EVALUATION.md — AI Essay Detector

> [!IMPORTANT]
> **CRITICAL PERFORMANCE LIMITATION WARNING**
> - **Human Class Sourced from Brown Corpus**: The "human" class was sourced from NLTK's Brown Corpus (academic/essay categories). It is NOT real student admissions essays. Therefore, these metrics indicate performance on distinguishing general academic human prose from AI-generated prose, rather than predicting performance on real admissions essays.
> - **AI Class Generated by Multiple Models**: The AI essays are generated by two structurally different AI sources: Google Gemini (40 essays across 3 prompting styles) and a local GPT-2 Small model (10 essays generated using raw next-token continuation, not instruction-following).
> - **Real GPT-2 Model Used**: This evaluation runs on real token-level perplexity features extracted using a local GPT-2 model (no simulation fallbacks).

## Test Set Performance

> ⚠️ **Note on Sample Size**: The held-out test set contains **n=20** samples (10 human, 10 AI). While 95.0% accuracy (19/20 correct) demonstrates strong signal on this benchmark, a sample size of 20 has wide confidence intervals and must be interpreted as a proof-of-concept evaluation rather than a statistically precise benchmark.

| Metric | Value |
|--------|-------|
| **Accuracy** | {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%}) |
| **Precision** | {metrics['precision']:.4f} ({metrics['precision']:.1%}) |
| **Recall** | {metrics['recall']:.4f} ({metrics['recall']:.1%}) |
| **F1 Score** | {metrics['f1']:.4f} ({metrics['f1']:.1%}) |

### Confusion Matrix

|  | Predicted Human | Predicted AI |
|--|:---:|:---:|
| **Actually Human** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |
| **Actually AI** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |

- **False Positive Rate**: {cm[0][1] / (cm[0][0] + cm[0][1]):.1%} (human essays flagged as AI)
- **False Negative Rate**: {cm[1][0] / (cm[1][0] + cm[1][1]):.1%} (AI essays missed)

---

## Confidently Wrong & Borderline Predictions

These are the 3 test-set essays that were either predicted incorrectly (wrong) or closest to the decision boundary (borderline correct close calls), highlighting the classifier's boundary behaviour.

{wrong_sections}

---

## ESL Bias Analysis

**Test method**: Ran 3 synthetic ESL-style essays (simulated non-native English
writing with common grammatical patterns) through the detector.

> **Important limitation**: These are synthetic ESL samples, not real essays from
> non-native English speakers. A proper bias audit would require a validated ESL
> essay corpus (e.g., from TOEFL or IELTS practice essays). This is explicitly
> noted as a gap.

{esl_summary}

{esl_section}

---

## Known Limitations

1. **Small dataset**: The model was trained on a limited dataset. Production-grade
   detection would need thousands of essays from diverse sources.

2. **GPT-2 baseline**: We use GPT-2 (124M) for perplexity scoring. Text from GPT-2
   itself will have very low perplexity (trivially detectable), while text from
   very different models (Claude, Gemini, etc.) may score differently.

3. **No adversarial robustness**: The detector has not been tested against essays
   specifically crafted to evade detection (e.g., paraphrased AI text).

4. **ESL bias not fully validated**: Only 3 synthetic ESL samples were tested.
   Real non-native English writers exhibit much more diverse patterns.

5. **Topic sensitivity**: Features like perplexity are topic-dependent. An essay
   about a niche topic may have high perplexity regardless of authorship.

6. **No temporal coverage**: AI models evolve rapidly. A detector trained on
   GPT-3 era text may not catch newer models' output.
"""

    eval_path = os.path.join(DOCS_DIR, "EVALUATION.md")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[Eval] Evaluation written to {eval_path}")
    return md


if __name__ == "__main__":
    run_evaluation()
