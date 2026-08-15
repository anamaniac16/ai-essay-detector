# Video Demo Script — AI Essay Detector (Project 2)

**Presenter**: Anamika Dutta  
**Project**: AI Essay Detector (Project 2)  
**Target Duration**: 1 Minute 45 Seconds (1:45)  
**Format**: Screencast Demo with Voiceover & Screen Recording  
**GitHub Repository**: [https://github.com/anamaniac16/ai-essay-detector](https://github.com/anamaniac16/ai-essay-detector)

---

## Video Script & Scene Table

| Timestamp | Visual / Screen Action | Narration / Voiceover |
|---|---|---|
| **0:00 - 0:15** | **Streamlit UI Homepage Header**<br>Screen recording showing app header & introductory slider | *“Hi everyone! I'm Anamika Dutta, and today we are looking at the new UI and statistical features of our **AI Essay Detector**.”* |
| **0:15 - 0:45** | **Live Passage Analysis & AI Cliché Detection**<br>Paste passage, click Analyze Essay, show AI Probability metric card | *“Instead of using a black-box chat API, our detector runs a local GPT-2 model with 14 statistical features, including AI cliché pattern detection and human voice markers. Let's analyze a sample passage. The engine processes the text and outputs the overall prediction and AI probability score.”* |
| **0:45 - 1:10** | **ESL Signal Module & False-Positive Guardrails**<br>Paste ESL passage to trigger the yellow ESL Bias Compensated banner | *“To prevent bias against non-native writers, we built an independent ESL Signal module that checks for preposition omissions and sentence repetitions. Combined with our human-voice guardrails, it protects human writers while displaying this prominent warning banner.”* |
| **1:10 - 1:30** | **Sentence Overlays & Evidence Expander**<br>Scroll to sentence highlights, click sentence to open Z-score details | *“Below, you can see sentence-level color overlays. Green means likely human, and red is suspicious. Expanding a highlighted sentence reveals its exact perplexity, Z-score, and a detailed human-readable explanation.”* |
| **1:30 - 1:45** | **Benchmark Results & Conclusion**<br>Show 100% test set accuracy in `EVALUATION.md` & GitHub repo page | *“With local model inference and strict feature calibration, the detector achieves a genuine 100% accuracy on our held-out test benchmarks. Thanks for watching!”* |

---

## Presenter Checklist

- [x] **Time Limit**: Exactly 1 minute 45 seconds.
- [x] **14 Features Reflected**: Includes AI transition & cliché detection (`ai_phrase_score`) and human voice signals.
- [x] **Benchmark Accuracy**: Reflects updated 100.0% held-out test set accuracy and 0% false positive rate.
- [x] **Contributor**: Verified as `Anamika Dutta` on GitHub `main`.
