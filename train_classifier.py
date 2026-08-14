"""
train_classifier.py — Train a classifier on extracted features.

This trains a statistical model (Logistic Regression + XGBoost) on features
extracted by detector/features.py — NOT on raw text, NOT using a chat model.

The model produces signal; the statistical layer makes the call.
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector.features import extract_features, FEATURE_NAMES

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def extract_features_for_dataset(df: pd.DataFrame, cache_path: str = None) -> pd.DataFrame:
    """
    Extract features for every essay in the dataframe.
    Caches results to avoid re-running expensive GPT-2 inference.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Train] Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"[Train] Extracting features for {len(df)} essays (this takes a while)...")
    features_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            feats = extract_features(row["text"])
            feats["label"] = row["label"]
            feats["index"] = idx
            features_list.append(feats)
        except Exception as e:
            print(f"[Train] WARNING: Failed to extract features for row {idx}: {e}")
            # Use zeros as fallback
            feats = {name: 0.0 for name in FEATURE_NAMES}
            feats["label"] = row["label"]
            feats["index"] = idx
            features_list.append(feats)

    features_df = pd.DataFrame(features_list)

    if cache_path:
        features_df.to_csv(cache_path, index=False)
        print(f"[Train] Features cached to {cache_path}")

    return features_df


def train_model(features_df: pd.DataFrame):
    """Train Logistic Regression classifier on extracted features."""
    X = features_df[FEATURE_NAMES].values
    y = features_df["label"].values

    # Handle any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    # Logistic Regression with scaling
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ))
    ])

    pipeline.fit(X, y)
    print("[Train] Model trained successfully.")

    # Extract feature importances (coefficients)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    importances = dict(zip(FEATURE_NAMES, coefs.tolist()))

    # Sort by absolute importance
    importances_sorted = dict(sorted(
        importances.items(), key=lambda x: abs(x[1]), reverse=True
    ))

    return pipeline, importances_sorted


def train_xgboost(features_df: pd.DataFrame):
    """Train XGBoost classifier as an alternative."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[Train] XGBoost not available, skipping.")
        return None, None

    X = features_df[FEATURE_NAMES].values
    y = features_df["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)
    print("[Train] XGBoost model trained successfully.")

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    importances_sorted = dict(sorted(
        importances.items(), key=lambda x: abs(x[1]), reverse=True
    ))

    return model, importances_sorted


def evaluate_model(pipeline, features_df: pd.DataFrame, label: str = "LogReg"):
    """Evaluate model and return metrics dict."""
    X = features_df[FEATURE_NAMES].values
    y = features_df["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, 'predict_proba') else None

    metrics = {
        "model": label,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    print(f"\n{'='*50}")
    print(f"  {label} — Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"     [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print(f"{'='*50}")

    return metrics, y_pred, y_proba


def find_confident_wrong(features_df: pd.DataFrame, test_df: pd.DataFrame,
                         y_pred, y_proba, n: int = 3):
    """
    Find the N most confidently wrong predictions.
    These are cases where the model was very sure but got it wrong.
    """
    y_true = features_df["label"].values

    wrong_mask = y_pred != y_true
    if not any(wrong_mask):
        print("[Eval] No wrong predictions found!")
        return []

    wrong_indices = np.where(wrong_mask)[0]

    if y_proba is not None:
        # Confidence = how far from 0.5 the prediction was
        confidence = np.abs(y_proba - 0.5)
        wrong_confidences = confidence[wrong_mask]
        # Sort by confidence (most confident first)
        sorted_idx = np.argsort(-wrong_confidences)
        top_wrong = wrong_indices[sorted_idx[:n]]
    else:
        top_wrong = wrong_indices[:n]

    results = []
    for idx in top_wrong:
        orig_idx = int(features_df.iloc[idx].get("index", idx))
        # Get text from test_df
        text = test_df.iloc[idx]["text"] if idx < len(test_df) else "TEXT NOT AVAILABLE"
        prob = float(y_proba[idx]) if y_proba is not None else -1.0

        results.append({
            "index": int(idx),
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "true_label": "HUMAN" if y_true[idx] == 0 else "AI",
            "predicted_label": "HUMAN" if y_pred[idx] == 0 else "AI",
            "ai_probability": prob,
            "confidence": float(abs(prob - 0.5)),
            "features": {name: float(features_df.iloc[idx][name]) for name in FEATURE_NAMES},
        })

    return results


def main():
    print("=" * 60)
    print("AI Essay Detector — Model Training")
    print("=" * 60)

    # Load data
    train_path = os.path.join(DATASET_DIR, "train.csv")
    test_path = os.path.join(DATASET_DIR, "test.csv")

    if not os.path.exists(train_path):
        print("[Train] No training data found. Run prepare_dataset.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[Train] Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # Extract features
    train_cache = os.path.join(DATASET_DIR, "train_features.csv")
    test_cache = os.path.join(DATASET_DIR, "test_features.csv")

    train_features = extract_features_for_dataset(train_df, cache_path=train_cache)
    test_features = extract_features_for_dataset(test_df, cache_path=test_cache)

    # Train Logistic Regression
    print("\n[Train] Training Logistic Regression...")
    lr_pipeline, lr_importances = train_model(train_features)

    # Save model
    model_path = os.path.join(MODELS_DIR, "classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lr_pipeline, f)
    print(f"[Train] Model saved to {model_path}")

    # Save feature importances
    importances_path = os.path.join(MODELS_DIR, "feature_importances.json")
    with open(importances_path, "w") as f:
        json.dump(lr_importances, f, indent=2)
    print(f"[Train] Feature importances saved to {importances_path}")

    # Evaluate on test set
    metrics, y_pred, y_proba = evaluate_model(lr_pipeline, test_features, "LogisticRegression")

    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Try XGBoost too
    print("\n[Train] Training XGBoost...")
    xgb_model, xgb_importances = train_xgboost(train_features)
    if xgb_model is not None:
        xgb_metrics, xgb_pred, xgb_proba = evaluate_model(xgb_model, test_features, "XGBoost")

        # Save XGBoost if better
        if xgb_metrics["f1"] > metrics["f1"]:
            print("[Train] XGBoost is better — saving as primary model.")
            with open(model_path, "wb") as f:
                pickle.dump(xgb_model, f)
            with open(importances_path, "w") as f:
                json.dump(xgb_importances, f, indent=2)
            metrics = xgb_metrics
            y_pred = xgb_pred
            y_proba = xgb_proba

            # Save XGBoost-specific info
            xgb_metrics_path = os.path.join(MODELS_DIR, "test_metrics.json")
            with open(xgb_metrics_path, "w") as f:
                json.dump(xgb_metrics, f, indent=2)

    # Find confident wrong predictions
    print("\n[Train] Finding confidently wrong predictions...")
    confident_wrong = find_confident_wrong(test_features, test_df, y_pred, y_proba)

    wrong_path = os.path.join(MODELS_DIR, "confident_wrong.json")
    with open(wrong_path, "w") as f:
        json.dump(confident_wrong, f, indent=2)
    print(f"[Train] Confidently wrong cases saved to {wrong_path}")

    for i, case in enumerate(confident_wrong):
        print(f"\n--- Confident Wrong #{i+1} ---")
        print(f"  True: {case['true_label']}, Predicted: {case['predicted_label']}")
        print(f"  AI probability: {case['ai_probability']:.4f}")
        print(f"  Text: {case['text_preview'][:150]}...")

    print("\n[Train] All done!")


if __name__ == "__main__":
    main()
