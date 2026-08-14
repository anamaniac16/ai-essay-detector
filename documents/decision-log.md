# AI Essay Detector — Architectural & Design Decision Log

This document records key technical, statistical, and architectural decisions made during the development of Project 2 (AI Essay Detector).

---

## 2026-08-14: Decision 1 — Token-Level Log-Probabilities Over Chat-Model Verdicts

### Context
Asking a chat model (e.g. GPT-4 or Gemini) "Is this essay written by AI?" is unreliable, non-deterministic, non-inspectable, and acts as a black box that cannot be calibrated or debugged.

### Decision
We use a local GPT-2 (124M) model strictly as a **statistical instrument** to extract token-level log-probabilities. Perplexity and burstiness are computed mathematically from these log-probabilities, and a separate scikit-learn classifier (Logistic Regression) makes the final prediction on extracted statistical features.

---

## 2026-08-14: Decision 2 — NLTK Brown Corpus Proxy Baseline for Human Class

### Context
Network restrictions and URL 404s/401s prevented automated downloading of Kaggle DAIGT or PERSUADE raw CSV datasets during time-boxed execution.

### Decision
We sourced 50 genuine human-authored formal prose passages from NLTK's Brown Corpus (`belles_lettres` and `learned` categories of academic and essay writing). 
To maintain total transparency and academic integrity:
- We prominently document that the "human" class represents formal prose proxy data rather than verified student admissions essays.
- We prominently display this caveat at the top of `dataset-card.md`, `EVALUATION.md`, `README.md`, and within the Streamlit UI header.

---

## 2026-08-14: Decision 3 — Multi-Model AI Class Diversity (Gemini + GPT-2 Raw Continuation)

### Context
A dataset where all AI essays come from a single LLM family (e.g. Google Gemini) risks training the classifier on specific template artifacts rather than general AI statistical signatures.

### Decision
We generated 10 additional AI essays using local **GPT-2 Small raw next-token continuation** (without instruction tuning or RLHF) and combined them with 40 Gemini essays generated across 3 prompting templates (Formal, Analytical, Narrative). This introduces multi-architecture AI diversity into the training set.

---

## 2026-08-14: Decision 4 — Independent ESL Bias Signal Module

### Context
Non-native English (ESL) writers often use simpler sentence structures and repetitive vocabulary. In standard AI detectors, these patterns trigger false positives because they overlap statistically with low perplexity and low burstiness.

### Decision
We built `detector/esl_signal.py` as an **entirely separate, independent module** from the AI classifier:
- It tracks article/preposition irregularities, vocabulary sophistication, clause density, and sentence start repetitions.
- It produces an independent ESL likelihood score.
- When both AI-likelihood AND ESL signals fire, the Streamlit app displays an explicit warning banner instructing the user to treat the AI score with caution rather than silently suppressing either signal.

---

## 2026-08-14: Decision 5 — Resolution of Dataset Circularity & Real Model Inference

### Context
During early testing, `features.py` fell back to a deterministic statistical simulator when model files were missing. The simulator assigned log-probabilities based on keyword lists (e.g. searching for "delve", "moreover"), creating a data-leaking circular dependency where the classifier learned simple keyword matching, yielding a fake 100% accuracy.

### Decision
- We corrected the local model path to load the 497MB `models/gpt2_local` weights.
- We deleted the keyword-based statistical simulator entirely.
- We retrained the classifier on real GPT-2 perplexity analysis, achieving a genuine **95.0% accuracy** on the held-out test set (F1 = 94.7%).
