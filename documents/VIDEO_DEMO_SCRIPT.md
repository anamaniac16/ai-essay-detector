# Video Demo Script — AI Essay Detector (Project 2)

**Presenter**: Anamika Dutta  
**Project**: AI Essay Detector (Project 2)  
**Target Duration**: Under 2 Minutes (1:45 – 1:55)  
**Format**: Fast-Paced Screencast Demo with Voiceover & Camera Overlay  
**GitHub Repository**: [https://github.com/anamaniac16/ai-essay-detector](https://github.com/anamaniac16/ai-essay-detector)

---

## Overview & Timing Flow (Total: 1:50)

| Scene | Time | Focus | Visual / Screen Action |
|---|---|---|---|
| **Scene 1** | 0:00 – 0:25 | **Hook & Problem Statement** | Camera ON / Streamlit Homepage Header |
| **Scene 2** | 0:25 – 0:55 | **Architecture & Feature Engineering** | VS Code view of `detector/features.py` |
| **Scene 3** | 0:55 – 1:30 | **Live Demo: Zero False-Positives & ESL Check** | Live Streamlit UI analyzing human essays & sentence inspector |
| **Scene 4** | 1:30 – 1:50 | **Benchmark Results & Conclusion** | `EVALUATION.md` metrics table & GitHub repo link |

---

## Detailed Script & Scene Breakdown

### Scene 1: Hook & Problem Statement (0:00 – 0:25)

**[VISUAL]**: Camera overlay in corner. Main screen displays Streamlit UI at `http://localhost:8502`.

**[AUDIO / NARRATION]**:
> "Hi everyone, I'm Anamika Dutta. 
> 
> Commercial AI detectors act as black boxes—frequently falsely accusing human writers, non-native English speakers, and students simply because their writing is structured. 
> 
> To solve this, I built the **AI Essay Detector**—a transparent, statistical detection engine powered by a local GPT-2 model and custom feature engineering, calibrated to **eliminate false positives on human text**."

---

### Scene 2: Architecture & Human-Voice Engineering (0:25 – 0:55)

**[VISUAL]**: Switch screen to VS Code highlighting `detector/features.py` and `calibrate_model.py`.

**[AUDIO / NARRATION]**:
> "Instead of chat-model guesses, our backend computes token-level log-probabilities using a local GPT-2 124M model. 
> 
> We extract 13 statistical features—including perplexity, sentence burstiness, POS-bigram entropy, and three explicit human protection signals: `first_person_pronoun_ratio` (*I, my, we*), `contraction_ratio` (*don't, it's*), and punctuation variety. 
> 
> Domain sign constraints in `calibrate_model.py` ensure these human markers consistently **reduce** AI probability."

---

### Scene 3: Live Demo — Zero False Positives & ESL Check (0:55 – 1:30)

**[VISUAL]**: Return to Streamlit UI. Paste a human student essay, click **Analyze Essay**, then click a sentence to show Grammarly-style sentence evidence and the ESL banner.

**[AUDIO / NARRATION]**:
> "Watch what happens when we paste a genuine human student essay. 
> 
> The system calculates an AI probability of **3.1%**, confidently marking it **Human-Written**.
> 
> When non-native English writing is entered, our independent **ESL Signal Detector** triggers an automatic bias discount so ESL writers aren't penalized. Plus, Grammarly-style sentence highlighting lets you inspect exact perplexities per sentence—with guardrails ensuring human text **never** displays false red or orange flags."

---

### Scene 4: Benchmark Results & Conclusion (1:30 – 1:50)

**[VISUAL]**: Show `EVALUATION.md` metrics table, then transition to GitHub repository homepage.

**[AUDIO / NARRATION]**:
> "On our held-out test benchmark, the system achieves **95.0% accuracy**, **100% precision for human protection**, and a **0% false positive rate** across 40+ human writing samples.
> 
> All code, dataset cards, and evaluation metrics are available on my GitHub repository. Thank you for watching!"

---

## Delivery Checklist

- [x] **Strict Time Limit**: Keep speech pace crisp (~150 words/min) to complete under 1 minute 50 seconds.
- [x] **Clipboard Prepped**: Have 1 human student essay pre-copied.
- [x] **App Running**: Streamlit live at `http://localhost:8502`.
- [x] **Commit Author**: Verified as `Anamika Dutta`.
