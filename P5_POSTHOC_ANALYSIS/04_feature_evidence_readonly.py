#!/usr/bin/env python3
"""
READ-ONLY feature evidence analysis.
This is NOT a feature-ablation experiment: it does not remove features,
refit models, or generate new predictions. It summarizes existing P4
permutation-importance, SHAP (if present), and KS outputs.
"""
from pathlib import Path
import argparse
import pandas as pd

def find_file(root: Path, candidates):
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5", default=".")
    ap.add_argument("--out", default="04_feature_ablation/results/read_only_feature_evidence")
    args = ap.parse_args()
    root = Path(args.p5).resolve()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    sources = {
        "permutation": find_file(root, [
            "02_feature_importance/results/development_permutation_importance.csv",
            "02_feature_importance/permutation/development_permutation_importance.csv",
            "P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031/outputs/development_analysis/permutation_importance.csv",
        ]),
        "shap": find_file(root, [
            "02_feature_importance/shap/shap_global_importance.csv",
            "02_feature_importance/results/shap_global_importance.csv",
            "02_feature_importance/shap/shap_feature_importance.csv",
        ]),
        "ks_v69_v77": find_file(root, [
            "03_distribution_shift/ks_test/ks_v69_v77.csv",
            "03_distribution_shift/results/ks_v69_v77.csv",
            "03_distribution_shift/results/feature_distribution_ks.csv",
        ]),
    }

    summary_rows = []
    for kind, path in sources.items():
        if path is None:
            summary_rows.append({"source": kind, "status": "NOT_FOUND", "path": "", "rows": 0,
                                 "note": "No matching existing analysis file found; no new model fitting performed."})
            continue
        try:
            df = pd.read_csv(path)
            summary_rows.append({"source": kind, "status": "READ", "path": str(path), "rows": len(df),
                                 "note": "Existing result imported; interpretation depends on source protocol."})
            if kind == "permutation":
                required = {"feature", "importance_mean"}
                if required.issubset(df.columns):
                    sort_cols = ["importance_mean"]
                    ranked = df.sort_values(sort_cols, ascending=False).copy()
                    ranked.insert(0, "rank", range(1, len(ranked) + 1))
                    ranked.to_csv(out / "permutation_importance_ranked.csv", index=False)
            elif kind == "shap":
                # Accept common column conventions; preserve raw data.
                feature_col = next((c for c in ["feature", "feature_name"] if c in df.columns), None)
                value_col = next((c for c in ["mean_abs_shap", "mean_abs_shap_value", "importance_mean", "shap_importance"] if c in df.columns), None)
                if feature_col and value_col:
                    ranked = df.sort_values(value_col, ascending=False).copy()
                    ranked.insert(0, "rank", range(1, len(ranked) + 1))
                    ranked.to_csv(out / "shap_importance_ranked.csv", index=False)
            elif kind == "ks_v69_v77":
                if {"feature", "ks_statistic", "p_value"}.issubset(df.columns):
                    df.sort_values("p_value").to_csv(out / "ks_distribution_shift_ranked.csv", index=False)
        except Exception as e:
            summary_rows[-1]["status"] = "READ_ERROR"
            summary_rows[-1]["note"] = repr(e)

    pd.DataFrame(summary_rows).to_csv(out / "feature_evidence_source_audit.csv", index=False)
    (out / "README.txt").write_text(
        "READ-ONLY FEATURE EVIDENCE SUMMARY\n"
        "This output is not Feature Ablation. No feature was removed, no model was refit,\n"
        "and no prediction was regenerated. It only ranks/imports already-existing\n"
        "permutation, SHAP, and KS outputs when their files and schemas are available.\n"
        "Permutation importance is tied to the split and scoring procedure used to create it.\n"
        "KS tests describe distribution differences, not predictive contribution or causality.\n",
        encoding="utf-8")
    print("Feature evidence audit:", out)
    print(pd.DataFrame(summary_rows).to_string(index=False))

if __name__ == "__main__":
    main()
