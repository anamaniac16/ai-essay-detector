# AI Essay Detector — Beginner-Friendly Breakdown

> Everything explained simply: what this app does, how it works under the hood, how circular data leakage was caught and fixed, and why evaluators will be impressed.

---

## 🎯 What Is This Project?

Think of **AI Essay Detector** as a **lie detector for text** — but instead of judging an essay based on a single opaque black-box percentage, it acts like a **transparent statistical magnifying glass**:
- It takes an essay and **breaks it down sentence-by-sentence**.
- It highlights each sentence in green (likely human), yellow (uncertain), orange (suspicious), or red (likely AI).
- When you click or hover over a flagged sentence, it shows you the **exact numerical evidence** (perplexity, burstiness, function-word ratio, log-prob variance) that drove the score.
- It includes a **standalone ESL (English as a Second Language) bias checker** that flags non-native writing markers so reviewers don't accidentally penalize non-native English students.

It's a **complete Python data science & web application** (GPT-2 NLP + Feature Engineering + Scikit-Learn Machine Learning + Streamlit Interactive UI).

---

## 📦 What Stack Is Used?

| Component | Technology | Details |
|-----------|------------|---------|
| **Language** | Python 3.13 | Core logic, feature extraction, ML training |
| **NLP Instrument** | PyTorch + HuggingFace Transformers | Local GPT-2 Small (124M parameters) running on CPU for token-level log-probabilities |
| **Machine Learning** | Scikit-Learn + XGBoost | Logistic Regression & XGBoost trained on 11 statistical features |
| **Web Interface** | Streamlit | Responsive dashboard with sentence-level CSS highlighting & inspectable expandable cards |
| **POS Tagging** | NLTK | Tagging parts-of-speech for POS-bigram entropy calculations |
| **Version Control** | Git | Sole contributor setup under `anamaniac16` |

---

## 🔧 What Was Done (Phases A through G)

### Phase A: Dataset Sourcing & Proxy Fix
- **The Problem**: Network firewall sandboxing blocked downloading raw Kaggle/HuggingFace datasets.
- **The Solution**: We sourced **50 genuine human formal prose passages** from NLTK's Brown Corpus (`belles_lettres` and `learned` academic categories).
- **Multi-Model AI Diversity**: We combined 40 Gemini essays across 3 prompting styles (Formal, Analytical, Narrative) with **10 local GPT-2 raw next-token continuation essays** (Style 4) to ensure the AI class isn't limited to a single instruction-tuned model family.
- **Honest Caveat**: Prominently documented that the human baseline is a formal prose proxy, not actual admissions essays.

---

### Phase B: Feature Extraction (Not a Black Box)
Instead of asking an AI chat model *"Is this written by AI?"* (which is unreliable and uninspectable), we run the text through GPT-2 to compute **11 named statistical features**:

1. `essay_perplexity`: How "surprised" GPT-2 is by the text (AI text is very predictable → low perplexity).
2. `burstiness_std`: Standard deviation of sentence perplexity (humans vary complexity → high burstiness; AI is uniform → low burstiness).
3. `burstiness_cv`: Coefficient of variation (normalized burstiness).
4. `avg_sentence_length` & `sentence_length_std`: Measures sentence length uniformity.
5. `function_word_ratio`: Percentage of common structural words (`the`, `is`, `with`, `for`).
6. `pos_bigram_entropy`: Variety of grammatical patterns (nouns, verbs, adjectives).
7. `type_token_ratio`: Vocabulary richness (unique words / total words).
8. `avg_log_prob` & `log_prob_std`: Average token confidence and variance.

---

### Phase C: Independent ESL Bias Signal
- **The Problem**: Non-native English (ESL) writers often use simpler vocabulary and uniform sentence structures. Standard AI detectors falsely flag ESL essays as AI because low perplexity overlaps with AI features.
- **The Solution**: Built `detector/esl_signal.py` as an **entirely separate score** measuring article/preposition irregularities, syllable sophistication, simple sentence ratios, and repeated sentence starters.
- **UI Integration**: If the ESL signal fires alongside a high AI score, an explicit warning banner appears: *"This essay shows patterns common in non-native English writing — treat this score with added caution."*

---

### Phase D: Statistical Classifier Training
- Trained a **Logistic Regression model** on the extracted 11 numerical features (80 train / 20 test split).
- **Strict Rule Enforced**: The model is trained purely on extracted numerical feature columns, NOT on raw text.
- Saved `models/classifier.pkl` and `models/feature_importances.json`.

---

### Phase F: Evaluation & Error Analysis
- **Test Set Accuracy**: **95.0%** (Precision: 100.0%, Recall: 90.0%, F1: 94.7%).
- **Sample Size Disclaimer**: Explicitly noted that $n=20$ is a proof-of-concept evaluation.
- **Categorized Boundary Cases**: Analyzed the 1 actual misclassification (False Negative on GPT-2 raw text continuation due to high temperature burstiness) and 2 borderline close calls.
- **ESL Audit**: Verified that simulated ESL essays trigger high AI likelihood (92–96%), proving the real-world bias risk and the necessity of our ESL warning banner.

---

### Phase G: 1-Command Startup & Clean GitHub Push
- Created cross-platform launcher scripts: `run.bat` (Windows) and `run.sh` (Linux/Mac).
- Pushed cleanly to GitHub under `anamaniac16` with single-command instructions at the top of `README.md`.

---

## 🐛 The Dataset Circularity Catch (The "Aha!" Moment)

### What Went Wrong Initially?
During early setup, `features.py` had a fallback "statistical simulator" when local model weights failed to load. But that simulator used **keyword matching** (checking for words like `"delve"`, `"moreover"`, `"tapestry"`) to assign fake log-probabilities.

### Why Was That Bad?
The classifier was simply learning the keyword list! It produced a **fake 100% accuracy** that was completely circular — if an essay contained "moreover", the simulator called it AI, and the classifier agreed.

### How Was It Fixed?
1. We corrected the local path to load the real 497MB `models/gpt2_local` safetensors weights.
2. We **deleted the statistical simulator entirely**.
3. We retrained the model on **100% real GPT-2 perplexity analysis**, yielding an honest, statistically valid **95.0% accuracy**.

---

## 📊 The Final Numbers

| Metric | Value |
|--------|-------|
| **Total Dataset Size** | 100 essays (50 Human, 50 AI) |
| **Train / Test Split** | 80 / 20 stratified |
| **Test Set Accuracy** | 95.0% (19 / 20 correct) |
| **Test Set Precision** | 100.0% (0 False Positives) |
| **Test Set Recall** | 90.0% (1 False Negative) |
| **Test Set F1 Score** | 94.7% |
| **Extracted Features** | 11 named numerical features |
| **1-Command Run** | `.\run.bat` (Win) / `./run.sh` (Mac/Linux) |

---

## 🏆 How This Helps You Get Selected

Here's what evaluators look for and how your work proves it:

1. **You Don't Build Black Boxes**: You didn't just wrap a chat model; you used GPT-2 as a statistical instrument and built inspectable feature vectors.
2. **You Fix Hidden Data Leakage**: You caught the circular keyword simulator, deleted it, and forced real model inference.
3. **You Care About Algorithmic Bias**: You built an independent ESL signal to protect non-native English writers from false accusations.
4. **You Are Honest About Limitations**: You documented the Brown Corpus proxy baseline and the $n=20$ sample size caveat prominently.
5. **You Build User-Friendly Apps**: The Streamlit interface provides sentence-level highlighting with 1-click feature evidence inspection.
6. **You Deliver 1-Click Reproducibility**: Anyone can clone the repo and launch the app in 1 command (`run.bat` / `run.sh`).

---

## 🗝️ Key Vocabulary to Know

| Term | What It Means (Simply) |
|------|----------------------|
| **Perplexity (PPL)** | How "surprised" a language model is by a sequence of words. AI text is predictable → low perplexity. |
| **Burstiness** | The variance in perplexity across sentences. Humans alternate simple and complex sentences (high burstiness); AI text is uniform (low burstiness). |
| **POS-Bigram Entropy** | Measure of grammatical pattern diversity. Lower entropy means formulaic sentence structure. |
| **Log-Probability** | The log of the probability GPT-2 assigned to predicting the next word given previous words. |
| **Type-Token Ratio (TTR)** | Unique words divided by total words — measures vocabulary richness. |
| **ESL Bias** | The tendency of AI detectors to falsely flag non-native English writing due to simpler vocabulary and uniform sentence length. |
| **Proxy Baseline** | Using academic prose (Brown Corpus) as a stand-in for student essays when direct essay datasets are unavailable. |

---

> [!TIP]
> **If you're asked to explain in an interview:**
> 
> *"I built an inspectable AI essay detector using local GPT-2 for token log-probabilities and scikit-learn for classification. Instead of a flat score or a black-box LLM verdict, it extracts 11 features like perplexity, sentence burstiness, and POS-bigram entropy, providing sentence-level highlighting in a Streamlit UI. I also built a separate ESL bias signal to surface false-positive risks for non-native writers. During development, I caught a data-leakage issue where an offline fallback was using keyword matching, fixed it by loading the real 124M GPT-2 weights, and achieved an honest 95.0% accuracy on the test set."*
