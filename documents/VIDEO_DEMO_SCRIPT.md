# Video Demo Script — AI Essay Detector (Project 2)

**Presenter**: Anamika Dutta  
**Project**: AI Essay Detector (Project 2)  
**Target Duration**: Under 2 Minutes (1:30 – 1:45)  
**Format**: 100% Pure App Screen Demo (Streamlit Interface Walkthrough)  
**GitHub Repository**: [https://github.com/anamaniac16/ai-essay-detector](https://github.com/anamaniac16/ai-essay-detector)

---

## Overview & Timing Flow (Total: 1:40)

| Scene | Time | Focus | Visual / Screen Action (Streamlit UI Only) |
|---|---|---|---|
| **Scene 1** | 0:00 – 0:25 | **Welcome & Problem Statement** | Main Streamlit App Header & Sensitivity Slider |
| **Scene 2** | 0:25 – 1:00 | **Live Demo: Human Writing Protection** | Paste human essay -> Click "Analyze Essay" -> Show 3.1% Human result |
| **Scene 3** | 1:00 – 1:25 | **Sentence-Level Evidence & Inspector** | Scroll to Grammarly-style highlights & expand sentence evidence details |
| **Scene 4** | 1:25 – 1:40 | **ESL Bias Compensation & Wrap-Up** | Show ESL banner, 0% false-positive benchmark, & concluding call to action |

---

## Detailed Script & Scene Breakdown

### Scene 1: Welcome & App Overview (0:00 – 0:25)

**[VISUAL]**: Screen recording of the live Streamlit web application running at `http://localhost:8502`. Hover mouse over the title **AI Essay Detector** and the sidebar slider.

**[AUDIO / NARRATION]**:
> "Hi everyone, I'm Anamika Dutta, and welcome to the live demo of my **AI Essay Detector**.
>
> Commercial detectors act as black boxes—frequently falsely accusing human writers, students, and non-native English speakers simply because their writing is clear. 
> 
> I built this statistical detector to provide 100% inspectable signals and strict guardrails calibrated to **eliminate false positives on human text**."

---

### Scene 2: Live Demo — Human Writing Protection (0:25 – 1:00)

**[VISUAL]**: Paste a genuine human student essay into the text box. Click the gradient **Analyze Essay** button. Watch the spinner, then highlight the metric cards.

**[AUDIO / NARRATION]**:
> "Let's test a genuine human student essay. 
> 
> I'll paste the text and click **Analyze Essay**. 
> 
> As you can see, the engine calculates an AI probability of **3.1%**, confidently classifying it as **Human-Written**. 
> 
> Whether you paste formal academic essays, short paragraphs, or personal reflections, our short-text uncertainty scaling and human-voice guardrails ensure human text remains safely classified as Human."

---

### Scene 3: Sentence-Level Evidence & Inspector (1:00 – 1:25)

**[VISUAL]**: Scroll down to **Sentence-Level Analysis**. Hover over green highlighted sentences, then click a sentence to open the **Detailed Sentence Evidence** expander.

**[AUDIO / NARRATION]**:
> "Scrolling down to **Sentence-Level Analysis**, every sentence is highlighted using a Grammarly-style visual editor. 
> 
> Guardrails ensure human text **never** triggers false red or orange 'Likely AI' highlights. 
> 
> Clicking on any sentence opens the evidence inspector, revealing the exact perplexity, Z-score, and token log-probabilities driving the analysis."

---

### Scene 4: ESL Bias Compensation & Wrap-Up (1:25 – 1:40)

**[VISUAL]**: Paste an ESL essay to show the yellow **ESL Bias Compensated** warning banner, then point out the 0% false positive rate.

**[AUDIO / NARRATION]**:
> "When non-native English writing is entered, our independent ESL detector surfaces a dedicated warning banner and applies a bias discount to protect ESL students.
> 
> Across a benchmark of 40 diverse human samples, our model achieves a **0% false positive rate**. 
> 
> You can try the live app and check out the full code on my GitHub repository. Thank you!"

---

## Recording Instructions for Presenter

- [x] **Screen Capture**: Record browser window showing `http://localhost:8502` only.
- [x] **No Code/IDE Switching**: Keep the entire recording inside the Streamlit web application.
- [x] **Pre-copied Texts**:
  1. Human Student Essay (for Scene 2 & 3)
  2. ESL Writing Sample (for Scene 4)
- [x] **Pace**: Crisp, energetic delivery completed in 1 minute 40 seconds.
