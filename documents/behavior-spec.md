# AI Essay Detector — Behavior & System Specification

This document provides a comprehensive behavioral specification for the AI Essay Detector, detailing feature extraction formulas, ESL bias indicators, classifier pipeline specifications, and Streamlit user interface interactions.

---

## 1. Feature Engineering Specifications (`detector/features.py`)

Every feature function is independently callable and inspectable.

| Feature Name | Description | Mathematical / Heuristic Formula | AI Direction |
|--------------|-------------|----------------------------------|--------------|
| `essay_perplexity` | Exponential of negative mean log-prob across all tokens | $\exp(-\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid w_{<i}))$ | **Lower** = more predictable (AI-like) |
| `burstiness_std` | Standard deviation of per-sentence perplexities | $\sigma(PPL_{s_1}, PPL_{s_2}, \dots, PPL_{s_k})$ | **Lower** = more uniform (AI-like) |
| `burstiness_cv` | Coefficient of variation (normalized burstiness) | $\frac{\sigma(PPL)}{\mu(PPL)}$ | **Lower** = uniform complexity (AI-like) |
| `avg_sentence_length` | Mean word count per sentence | $\frac{1}{K} \sum_{j=1}^K \text{words}(s_j)$ | Evaluates stylistic consistency |
| `sentence_length_std` | Standard deviation of sentence lengths | $\sigma(\text{words}(s_1), \dots, \text{words}(s_k))$ | **Lower** = uniform length (AI-like) |
| `function_word_ratio` | Ratio of 50 common English function words to total words | $\frac{\text{count}(\text{FunctionWords})}{\text{TotalWords}}$ | Evaluates structural balance |
| `pos_bigram_entropy` | Shannon entropy of POS-tag bigram distribution | $-\sum p(b) \log_2 p(b)$ | **Lower** = formulaic structure (AI-like) |
| `type_token_ratio` | Lexical diversity (unique words / total words) | $\frac{\text{unique}(\text{words})}{\text{total}(\text{words})}$ | Evaluates vocabulary richness |
| `num_sentences` | Total sentence count | Count of sentence splits | Contextual scaling |
| `avg_log_prob` | Mean log probability across tokens | $\frac{1}{N} \sum \log P(w_i)$ | **Higher (less negative)** = AI-like |
| `log_prob_std` | Standard deviation of token log-probabilities | $\sigma(\log P(w_1), \dots, \log P(w_n))$ | **Lower** = uniform uncertainty (AI-like) |

---

## 2. ESL Signal Specifications (`detector/esl_signal.py`)

The ESL signal operates as an **independent score** [0, 1] to surface potential false-positive bias.

### Heuristics & Rules
1. **Article & Preposition Irregularities**: Checks patterns such as missing articles before adjectives/nouns (`is very good thing`), double prepositions (`in on`), and confused preposition collocations (`depend of`, `interested for`, `married with`).
2. **Vocabulary Sophistication**: Ratio of words with $\ge 3$ syllables to total word count.
3. **Simple Sentence Ratio**: Proportion of sentences without commas or subordinating conjunctions (`because`, `although`, `while`, `if`, etc.).
4. **Sentence Start Repetition**: Proportion of sentences starting with the same initial word (e.g. repeated `I`, `The`, `It`).

### Decision Logic
If `esl_likelihood_score(text) >= 0.35`, the ESL flag is marked **Active** and a prominent warning banner is triggered in the user interface.

---

## 3. Classifier Pipeline (`train_classifier.py`)

```
Raw Input Text → Tokenizer & GPT-2 (124M) → Feature Vector (11 features)
                        ↓
               StandardScaler (Mean 0, Variance 1)
                        ↓
               Logistic Regression (balanced class weight, C=1.0)
                        ↓
               Prediction & Probability [0, 1]
```

### Performance Metrics (Held-Out Test Set, n=20)
- **Accuracy**: 95.0%
- **Precision**: 100.0%
- **Recall**: 90.0%
- **F1 Score**: 94.7%

---

## 4. User Interface Specification (`app.py`)

### Sentence-Level Highlighting Thresholds

| AI Probability Score | CSS Highlight Color | Label |
|----------------------|---------------------|-------|
| $< 30\%$ | `rgba(34, 197, 94, 0.15)` (Soft Green) | **Likely Human** |
| $30\% - 50\%$ | `rgba(250, 204, 21, 0.20)` (Soft Yellow) | **Uncertain** |
| $50\% - 70\%$ | `rgba(251, 146, 60, 0.25)` (Soft Orange) | **Suspicious** |
| $> 70\%$ | `rgba(239, 68, 68, 0.30)` (Soft Red) | **Likely AI** |

### Sentence Evidence Expander
Clicking any flagged sentence displays:
- Per-sentence Perplexity & Z-Score (how much it deviates from essay average)
- Average Log-Probability & Log-Prob StdDev
- Function Word Ratio & Word Count
- Human-readable interpretation explaining why the sentence was flagged.
