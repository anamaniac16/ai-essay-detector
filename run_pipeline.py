"""
run_pipeline.py — Run the full end-to-end pipeline.

1. Prepare dataset
2. Extract features (train + test)
3. Train classifier
4. Evaluate on held-out test set
5. Generate EVALUATION.md
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    start_time = time.time()

    print("=" * 70)
    print("  AI Essay Detector — Full Pipeline")
    print("=" * 70)

    # Phase A: Dataset
    print("\n" + "=" * 70)
    print("  PHASE A: Dataset Preparation")
    print("=" * 70)
    from prepare_dataset import main as prepare_main
    prepare_main()

    # Phase B+C+D: Feature extraction + Training
    print("\n" + "=" * 70)
    print("  PHASES B-D: Feature Extraction + Model Training")
    print("=" * 70)
    from train_classifier import main as train_main
    train_main()

    # Phase F: Evaluation
    print("\n" + "=" * 70)
    print("  PHASE F: Evaluation")
    print("=" * 70)
    from evaluate import run_evaluation
    metrics, confident_wrong, esl_results = run_evaluation()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"  Pipeline complete in {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print("=" * 70)

    print("\n  Results Summary:")
    if metrics:
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")

    print(f"\n  Confident wrong predictions: {len(confident_wrong)}")
    print(f"  ESL bias tests run: {len(esl_results)}")

    print("\n  Output files:")
    print("    dataset/dataset-card.md")
    print("    dataset/train.csv, dataset/test.csv")
    print("    models/classifier.pkl")
    print("    models/feature_importances.json")
    print("    documents/EVALUATION.md")

    print("\n  To launch the Streamlit app:")
    print("    .\\venv\\Scripts\\streamlit.exe run app.py")
    print()


if __name__ == "__main__":
    main()
