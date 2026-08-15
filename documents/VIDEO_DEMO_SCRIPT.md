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
| **0:00 - 0:15** | **Streamlit UI Homepage Header**<br>Screen recording showing app header & introductory slider | *“Hi everyone! Today we are looking at the new UI and statistical features of our **AI Essay Detector**.”* |
| **0:15 - 0:45** | **Live Passage Analysis Demo**<br>Paste sample essay, click Analyze Essay, show AI Probability metric card | *“Instead of using a black-box chat API, our detector runs a local GPT-2 model to extract token-level perplexity. Let's analyze a sample passage. The detector processes the text and outputs the overall prediction and AI probability score.”* |
| **0:45 - 1:10** | **ESL Signal Module & Warning Banner**<br>Paste ESL passage to trigger the yellow ESL Bias Compensated banner | *“To prevent bias against non-native writers, we built an independent ESL Signal module. It checks for common preposition omissions and sentence repetitions. When both AI and ESL signals fire, it displays this prominent warning banner.”* |
| **1:10 - 1:30** | **Sentence Overlays & Evidence Expander**<br>Scroll to sentence highlights, click sentence to open Z-score details | *“Below, you can see sentence-level color overlays. Green means likely human, and red is suspicious. Expanding a highlighted sentence reveals its exact perplexity, Z-score, and a detailed human-readable explanation.”* |
| **1:30 - 1:45** | **Benchmark Results & Conclusion**<br>Show test metrics table from `EVALUATION.md` & GitHub repo page | *“With a local model running real perplexity analysis, the detector achieves a genuine 95% accuracy on our test benchmarks. Thanks for watching!”* |

---

## Presenter Checklist

- [x] **Time Limit**: Exactly 1 minute 45 seconds.
- [x] **Narration**: Follows exact requested narration text.
- [x] **Screen Capture**: Record browser at `http://localhost:8502`.
- [x] **Contributor**: Verified as `Anamika Dutta` on GitHub `main`.
