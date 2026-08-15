# Video Demo Script — AI Essay Detector (Project 2)

**Presenter**: Anamika Dutta  
**Project**: AI Essay Detector (Project 2)  
**Target Duration**: 3:30 – 4:00 Minutes  
**Format**: Screencast Demo with Voiceover & Camera Overlay  
**GitHub Repository**: [https://github.com/anamaniac16/ai-essay-detector](https://github.com/anamaniac16/ai-essay-detector)

---

## Overview & Video Flow

| Scene | Time | Focus | Visual / Screen Action |
|---|---|---|---|
| **Scene 1** | 0:00 – 0:45 | **Introduction & Problem Statement** | Camera ON / Streamlit Homepage Header |
| **Scene 2** | 0:45 – 1:30 | **Architecture & Feature Engineering** | VS Code view of `detector/features.py` & feature table |
| **Scene 3** | 1:30 – 2:30 | **Live Demo: Human Writing Protection** | Live Streamlit UI analyzing human essays & short paragraphs |
| **Scene 4** | 2:30 – 3:15 | **ESL Bias Compensation & Sentence Highlighting** | ESL banner demo, sentence highlighting & evidence expander |
| **Scene 5** | 3:15 – 3:45 | **Model Calibration & Benchmark Results** | `EVALUATION.md` test metrics & 0% false positive results |
| **Scene 6** | 3:45 – 4:00 | **Conclusion & Call to Action** | GitHub repo overview & wrap-up |

---

## Detailed Script & Scene Breakdown

### Scene 1: Introduction & Problem Statement (0:00 – 0:45)

**[VISUAL]**: Camera overlay on bottom right. Main screen shows the Streamlit app header for **AI Essay Detector**.

**[AUDIO / NARRATION]**:
> "Hi everyone, I'm Anamika Dutta, and welcome to the walk-through of my project: the **AI Essay Detector**.
>
> One of the biggest challenges with AI text detection today is that commercial detectors act as black boxes—frequently falsely accusing human writers, non-native English speakers, and students of using AI simply because their writing is clear or structured.
>
> In this project, I built a transparent, statistical AI detection engine powered by a local GPT-2 model and custom feature engineering. Unlike black-box wrappers, every signal here is inspectable, explainable, and calibrated with strict guardrails to **eliminate false positives on human-authored text**."

---

### Scene 2: Core Architecture & Feature Engineering (0:45 – 1:30)

**[VISUAL]**: Switch screen to VS Code highlighting `detector/features.py` and `calibrate_model.py`.

**[AUDIO / NARRATION]**:
> "Let's take a look under the hood. 
> 
> Rather than relying on chat-model verdicts, our backend extracts token-level log-probabilities using a local GPT-2 124M model. From this, we compute 13 statistical features:
>
> 1. **Perplexity and Burstiness**: Measuring sentence-to-sentence complexity variance—since human writing is naturally 'bursty' while AI text is unnaturally uniform.
> 2. **POS-Bigram Entropy & Lexical Diversity**: Evaluating grammatical structure and vocabulary richness (`type_token_ratio`).
> 3. **Human Voice Indicators**: We recently engineered three explicit human protection features: `first_person_pronoun_ratio` tracking narrative markers like *I, me, my, we, our*; `contraction_ratio` tracking natural conversational phrasing like *don't, can't, it's*; and `punctuation_variety`.
>
> During model calibration in `calibrate_model.py`, we enforced domain-sign constraints so that these human traits consistently **reduce** AI probability, protecting human authors."

---

### Scene 3: Live Demo — Human Writing Protection (1:30 – 2:30)

**[VISUAL]**: Return to Streamlit UI at `http://localhost:8502`. Paste a human-written student essay and click **Analyze Essay**.

**[AUDIO / NARRATION]**:
> "Now let's see the application in action. 
>
> I'll paste a genuine student essay into the text editor and click **Analyze Essay**.
>
> Notice the result: the AI probability is computed at **3.1%**, confidently classified as **Human-Written**.
>
> Even when we analyze short human paragraphs, formal academic excerpts, or personal reflections, our short-text uncertainty scaling and human-voice guardrails ensure these texts remain safely below 20% AI probability. Across a benchmark suite of 40 diverse human writing samples, our calibrated model achieved a **0% false positive rate**."

---

### Scene 4: ESL Bias Compensation & Sentence Evidence (2:30 – 3:15)

**[VISUAL]**: Paste an ESL essay into the editor. Highlight the **ESL Bias Compensated** banner and click a sentence to show the Grammarly-style sentence inspector.

**[AUDIO / NARRATION]**:
> "A major feature of this project is our independent **ESL Signal Detector** (`esl_signal.py`).
>
> Non-native English writers often use simpler sentence structures or article patterns that statistical classifiers confuse with AI text. When ESL patterns are detected, the system displays a dedicated warning banner and automatically applies a bias discount to prevent false accusations.
>
> Furthermore, the **Sentence-Level Analysis** highlights each sentence using Grammarly-style color coding. Clicking on any sentence reveals the exact perplexity, Z-score, and token log-probability driving the score—giving educators and students full transparency."

---

### Scene 5: Benchmark Evaluation & Results (3:15 – 3:45)

**[VISUAL]**: Display `documents/EVALUATION.md` showing test metrics and confusion matrix.

**[AUDIO / NARRATION]**:
> "Let's review the evaluation metrics.
>
> On our held-out test dataset, the model achieves:
> - **95.0% Overall Accuracy**
> - **100.0% Precision for Human Protection**
> - **0.0% False Positive Rate** on human essays.
>
> By capping individual sentence scores for human-classified documents, we guarantee that genuine human writing never displays false red or orange 'Likely AI' highlights."

---

### Scene 6: Conclusion & Wrap-Up (3:45 – 4:00)

**[VISUAL]**: Camera overlay full screen or GitHub repository homepage (`https://github.com/anamaniac16/ai-essay-detector`).

**[AUDIO / NARRATION]**:
> "In summary, the AI Essay Detector delivers an accurate, transparent, and fair detection system that respects human authorship.
>
> All code, dataset cards, and evaluation benchmarks are available on my GitHub repository. Thank you for watching!"

---

## Presenter Delivery Checklist & Tips

- [x] **Lighting & Audio**: Ensure clear microphone audio and good lighting if using camera overlay.
- [x] **App Readiness**: Streamlit app running at `http://localhost:8502`.
- [x] **Sample Texts Ready**: Have 3 test texts copied on clipboard:
  1. Formal Human Student Essay
  2. Casual / Personal Human Reflection
  3. ESL Essay Sample
- [x] **Repository Verification**: Ensure commits are authored as `Anamika Dutta` on GitHub `main` branch.
