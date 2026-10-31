#!/usr/bin/env python3
"""
READ-ONLY trivial baselines derived from existing test labels.
No rule-based predictor, text model, fitting, or prediction regeneration.
Reports majority-class and prevalence/random reference metrics per target.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5", default=".")
    ap.add_argument("--labels", default="../P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031/outputs/after_unseal_inputs/test_labels_234.csv")
    ap.add_argument("--out", default="06_baselines/results/read_only_trivial_baselines")
    args = ap.parse_args()
    root = Path(args.p5).resolve()
    labels_path = (root / args.labels).resolve()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_path)
    if "ipi_affected" not in df.columns or "target_model" not in df.columns:
        raise ValueError("Labels file must contain ipi_affected and target_model")
    rows = []
    for target, g in df.groupby("target_model", sort=True):
        y = g["ipi_affected"].astype(int).to_numpy()
        n = len(y); pos = int(y.sum()); neg = n - pos
        prevalence = pos / n if n else np.nan
        # Constant negative majority-class classifier: thresholded prediction.
        majority_positive = int(pos > neg)
        pred = np.full(n, majority_positive, dtype=int)
        accuracy = float((pred == y).mean()) if n else np.nan
        # For a constant classifier with only one predicted class, balanced accuracy is 0.5
        # when both ground-truth classes are present.
        bal_acc = 0.5 if pos and neg else np.nan
        rows.extend([
            {"target_model": target, "baseline": "majority_class_constant",
             "n": n, "positive": pos, "negative": neg, "prevalence": prevalence,
             "accuracy": accuracy, "balanced_accuracy": bal_acc,
             "expected_auroc": 0.5 if pos and neg else np.nan,
             "expected_average_precision": prevalence,
             "note": "Constant majority-class prediction; AUROC/AP are reference expectations, not thresholded outputs."},
            {"target_model": target, "baseline": "random_ranking_reference",
             "n": n, "positive": pos, "negative": neg, "prevalence": prevalence,
             "accuracy": np.nan, "balanced_accuracy": np.nan,
             "expected_auroc": 0.5 if pos and neg else np.nan,
             "expected_average_precision": prevalence,
             "note": "Theoretical random-ranking expectation; no random model was trained or scored."},
            {"target_model": target, "baseline": "prevalence_only",
             "n": n, "positive": pos, "negative": neg, "prevalence": prevalence,
             "accuracy": np.nan, "balanced_accuracy": np.nan,
             "expected_auroc": np.nan,
             "expected_average_precision": prevalence,
             "note": "No-discrimination AP reference equals test-set prevalence."}
        ])
    result = pd.DataFrame(rows)
    result.to_csv(out / "trivial_baseline_references.csv", index=False)
    print("Wrote:", out / "trivial_baseline_references.csv")
    print(result.to_string(index=False))
    print("\nNo models trained; no new scores or predictions generated.")

if __name__ == "__main__":
    main()
