from pathlib import Path
import re
import joblib
import pandas as pd
import numpy as np

ROOT = Path("P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031")
OUT = Path("P5_POSTHOC_ANALYSIS/02_feature_importance/results")
OUT.mkdir(parents=True, exist_ok=True)

V77 = pd.read_csv(ROOT / "data/features_p3_77_1131.csv")
V69 = pd.read_csv(ROOT / "data/features_p4_69_1131.csv")
META = pd.read_csv(ROOT / "data/metadata_membership_1131.csv")
LOCKED = pd.read_csv(ROOT / "outputs/locked_test_scores/test_scores_locked.csv")

V_EXP = [
    "n_freetext_reads",
    "n_writes_after_freetext",
    "n_egress_after_freetext",
    "n_destructive_after_freetext",
    "frac_writes_after_freetext",
    "n_calls_after_last_freetext",
]

FEATURES = {
    "V77": [c for c in V77.columns if c != "trajectory_id"],
    "V69": [c for c in V69.columns if c != "trajectory_id"],
    "V_exp": V_EXP,
}

DATA = {"V77": V77, "V69": V69}

def parse_model(filename):
    pattern = (
        r"P4_(V69|V77)_(V_all|V_exp)_(GB|LR|RF)_"
        r"(POOLED_LOLO|SINGLE)_(LLM-[ABC])_v1"
    )
    m = re.fullmatch(pattern, filename.removesuffix(".joblib"))

    if not m:
        raise ValueError(f"Cannot parse model filename: {filename}")

    return m.groups()


def main():
    results = []
    model_paths = sorted((ROOT / "outputs/models").glob("*.joblib"))

    metadata = META.copy()
    metadata["split"] = metadata["split"].astype(str).str.lower()

    for i, path in enumerate(model_paths, 1):
        version, regime, algorithm, strategy, suffix_model = parse_model(path.name)

        model = joblib.load(path)

        features = FEATURES[version] if regime == "V_all" else FEATURES["V_exp"]

        base = DATA[version].merge(
            metadata[["trajectory_id", "target_model", "split"]],
            on="trajectory_id",
            validate="one_to_one"
        )

        test = base[base["split"] == "test"].copy()

        if strategy == "SINGLE":
            # Single-source models are evaluated on all three target models.
            expected_targets = ["LLM-A", "LLM-B", "LLM-C"]
        else:
            # The suffix identifies the held-out target in pooled LOLO.
            expected_targets = [suffix_model]

        test = test[test["target_model"].isin(expected_targets)].copy()
        test = test.sort_values(["target_model", "trajectory_id"]).reset_index(drop=True)

        X = test[features]

        locked = LOCKED[
            LOCKED["experiment_id"] == path.stem
        ].copy()

        locked = locked.sort_values(
            ["target_model", "trajectory_id"]
        ).reset_index(drop=True)

        if len(test) != len(locked):
            results.append({
                "model_file": path.name,
                "status": "ROW_COUNT_MISMATCH",
                "n_recomputed": len(test),
                "n_locked": len(locked),
                "same_ids": False,
                "max_abs_diff": np.nan,
                "error": "Test rows and locked rows differ"
            })
            print(
                f"[{i:02d}/{len(model_paths)}] ROW_COUNT_MISMATCH "
                f"| recomputed={len(test)} locked={len(locked)}"
            )
            continue

        same_ids = (
            test["trajectory_id"].astype(str).tolist()
            == locked["trajectory_id"].astype(str).tolist()
        )

        same_targets = (
            test["target_model"].astype(str).tolist()
            == locked["target_model"].astype(str).tolist()
        )

        try:
            pred = model.predict_proba(X)[:, 1]
            locked_scores = locked["score"].to_numpy(dtype=float)

            diff = np.abs(pred - locked_scores)
            max_diff = float(diff.max()) if len(diff) else 0.0

            passed = (
                same_ids
                and same_targets
                and np.allclose(
                    pred,
                    locked_scores,
                    rtol=1e-7,
                    atol=1e-9
                )
            )

            status = "PASS" if passed else "MISMATCH"
            error = ""

        except Exception as e:
            status = "PREDICTION_ERROR"
            max_diff = np.nan
            error = repr(e)

        results.append({
            "model_file": path.name,
            "status": status,
            "n_recomputed": len(test),
            "n_locked": len(locked),
            "same_ids": same_ids,
            "same_targets": same_targets,
            "max_abs_diff": max_diff,
            "error": error
        })

        print(
            f"[{i:02d}/{len(model_paths)}] {status:20s} "
            f"| {path.name} "
            f"| rows={len(test)} "
            f"| max_diff={max_diff}"
        )

        del model

    df = pd.DataFrame(results)
    output = OUT / "p4_prediction_consistency.csv"
    df.to_csv(output, index=False)

    print("\n" + "=" * 65)
    print("PREDICTION CONSISTENCY SUMMARY")
    print("=" * 65)

    print(df["status"].value_counts(dropna=False).to_string())
    print("\nTotal models:", len(df))
    print("Total recomputed rows:", df["n_recomputed"].sum())
    print("Total locked rows:", df["n_locked"].sum())

    print("\nMaximum absolute difference:")
    print(df["max_abs_diff"].max())

    print("\nReport:", output.resolve())


if __name__ == "__main__":
    main()
