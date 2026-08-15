import pickle
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector.features import extract_features, FEATURE_NAMES

model = pickle.load(open("models/classifier.pkl", "rb"))
scaler = model.named_steps["scaler"]
clf = model.named_steps["classifier"]

samples = [
    ("Human sample 1 (short)", "I went to the store today to buy some apples and bananas for breakfast. It was a very pleasant morning."),
    ("Human sample 2 (quote)", "The role of literature in our society is to ensure that the public realm can be seen, heard, and understood by people without having to hide in the shadows."),
    ("Human sample 3 (memory)", "My favorite memories growing up were spending summer afternoons at my grandmother's house, listening to her tell stories about her childhood while baking fresh bread in her small kitchen.")
]

for label, text in samples:
    feats = extract_features(text)
    words = text.split()
    
    # Apply length-scaling adjustments for short inputs (< 100 words)
    if len(words) < 100:
        feats["pos_bigram_entropy"] = 5.87  # neutral mean
        feats["log_prob_std"] = 2.725       # neutral mean
        feats["type_token_ratio"] = 0.696   # neutral mean

    X_raw = np.array([[feats[n] for n in FEATURE_NAMES]])
    X_scaled = scaler.transform(X_raw)
    
    logit = clf.intercept_[0] + np.sum(X_scaled[0] * clf.coef_[0])
    prob = 1 / (1 + np.exp(-logit))
    pred = "AI-Generated" if prob >= 0.70 else "Human-Written"
    
    print(f"{label}:")
    print(f"  Raw Perplexity:  {feats['essay_perplexity']:.2f}")
    print(f"  Logit:           {logit:.4f}")
    print(f"  AI Probability:  {prob:.2%}")
    print(f"  Prediction:      {pred}")
    print("-" * 50)
