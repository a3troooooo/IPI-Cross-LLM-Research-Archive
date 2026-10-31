#!/usr/bin/env python3
"""
READ-ONLY statistical inference on frozen P4 predictions.
Computes connected-component bootstrap CIs for AUROC and Average Precision
from existing test predictions. Does not fit models or alter predictions.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def union_find_components(meta):
    parent = list(range(len(meta)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    seen = {}
    for i, row in meta.reset_index(drop=True).iterrows():
        for col in ("condition_group_id", "user_task_group_id"):
            if col not in meta.columns: continue
            val = row[col]
            if pd.isna(val) or str(val).strip() == "": continue
            key = (col, str(val))
            if key in seen: union(i, seen[key])
            else: seen[key] = i
    roots = [find(i) for i in range(len(meta))]
    return pd.Series(roots, index=meta.index, name="component_id")

def safe_metrics(y, s):
    if len(np.unique(y)) < 2: return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5", default=".")
    ap.add_argument("--predictions", default="../P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031/outputs/final_test_evaluation/test_predictions_with_labels_after_unseal.csv")
    ap.add_argument("--membership", default="../P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031/data/metadata_membership_1131.csv")
    ap.add_argument("--out", default="reports/tables/statistical_inference_read_only")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261031)
    args = ap.parse_args()
    root = Path(args.p5).resolve()
    pred_path = (root / args.predictions).resolve()
    meta_path = (root / args.membership).resolve()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_path)
    meta = pd.read_csv(meta_path)
    required_pred = {"experiment_id", "trajectory_id", "score", "ipi_affected"}
    required_meta = {"trajectory_id", "condition_group_id", "user_task_group_id"}
    if not required_pred.issubset(pred.columns):
        raise ValueError(f"Prediction file missing columns: {sorted(required_pred-set(pred.columns))}")
    if not required_meta.issubset(meta.columns):
        raise ValueError(f"Membership file missing columns: {sorted(required_meta-set(meta.columns))}")

    comp = union_find_components(meta[["trajectory_id", "condition_group_id", "user_task_group_id"]])
    meta2 = meta[["trajectory_id"]].copy()
    meta2["component_id"] = comp.to_numpy()
    d = pred.merge(meta2, on="trajectory_id", how="left", validate="many_to_one")
    if d["component_id"].isna().any():
        missing = int(d["component_id"].isna().sum())
        raise ValueError(f"{missing} prediction rows did not map to a connected component")

    rng = np.random.default_rng(args.seed)
    rows = []
    for exp_id, g in d.groupby("experiment_id", sort=True):
        g = g.copy()
        y = g["ipi_affected"].astype(int).to_numpy()
        s = g["score"].astype(float).to_numpy()
        auc, apv = safe_metrics(y, s)
        groups = g["component_id"].unique()
        auc_boot, ap_boot = [], []
        # Resample whole connected components, retaining all trajectories in each sampled component.
        by_comp = {c: g[g["component_id"] == c] for c in groups}
        for _ in range(args.replicates):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            b = pd.concat([by_comp[c] for c in sampled], ignore_index=True)
            by, bs = b["ipi_affected"].astype(int).to_numpy(), b["score"].astype(float).to_numpy()
            ba, bp = safe_metrics(by, bs)
            if np.isfinite(ba): auc_boot.append(ba)
            if np.isfinite(bp): ap_boot.append(bp)
        def ci(v):
            return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) else (np.nan, np.nan)
        al, ah = ci(auc_boot); pl, ph = ci(ap_boot)
        rows.append({
            "experiment_id": exp_id, "n": len(g), "positive": int(y.sum()),
            "components": len(groups), "auroc": auc, "auroc_ci_low": al, "auroc_ci_high": ah,
            "ap": apv, "ap_ci_low": pl, "ap_ci_high": ph,
            "valid_auroc_replicates": len(auc_boot), "valid_ap_replicates": len(ap_boot),
            "bootstrap_unit": "connected_component_union_find",
            "replicates_requested": args.replicates
        })
    result = pd.DataFrame(rows)
    result.to_csv(out / "component_bootstrap_ci.csv", index=False)
    audit = {
        "predictions_file": str(pred_path), "membership_file": str(meta_path),
        "prediction_rows": len(pred), "experiments": int(pred["experiment_id"].nunique()),
        "replicates": args.replicates, "seed": args.seed,
        "models_refit": False, "predictions_modified": False,
        "note": "Post-unseal exploratory inference on frozen predictions; not a new blind confirmatory test."
    }
    (out / "run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Wrote:", out / "component_bootstrap_ci.csv")
    print("Experiments:", len(result), "| predictions:", len(pred), "| refit: NO")

if __name__ == "__main__":
    main()
