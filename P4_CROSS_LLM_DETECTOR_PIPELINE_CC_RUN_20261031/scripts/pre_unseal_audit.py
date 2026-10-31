from pathlib import Path
import hashlib
import json
import pandas as pd
import sys

ROOT = Path.cwd()
DEV = ROOT / "outputs/development"
MODELS = ROOT / "outputs/models"
LOCK = ROOT / "outputs/locked_test_scores"

results = []

def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((status, name))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

def warn(name, detail=""):
    results.append(("WARN", name))
    print(f"[WARN] {name}" + (f" — {detail}" if detail else ""))

print("\n" + "=" * 68)
print("P4 PRE-UNSEAL READ-ONLY AUDIT")
print("=" * 68)

# 1. Check expected files
pred_files = sorted(DEV.glob("*validation_predictions.csv"))
model_files = sorted(MODELS.glob("*.joblib"))

report("Validation prediction files", len(pred_files) == 72, str(len(pred_files)))
report("Model artifacts", len(model_files) == 72, str(len(model_files)))

# 2. Validation prediction structure
schema_ok = True
row_ok = True
id_ok = True
prob_ok = True
label_ok = True
schemas = {}

for f in pred_files:
    try:
        d = pd.read_csv(f)
    except Exception as e:
        schema_ok = False
        print("[FAIL] Cannot read:", f.name, str(e))
        continue

    schemas[tuple(d.columns)] = schemas.get(tuple(d.columns), 0) + 1

    is_pooled = "POOLED" in f.name
    expected_rows = 146 if is_pooled else 73

    if len(d) != expected_rows:
        row_ok = False

    # Detect common trajectory ID column names
    id_col = next(
        (c for c in ["trajectory_id", "trajectory", "id"] if c in d.columns),
        None
    )
    if id_col:
        if d[id_col].isna().any() or d[id_col].duplicated().any():
            id_ok = False
    else:
        id_ok = False

    # Detect common probability column names
    score_col = next(
        (c for c in ["y_score", "probability", "prob", "score", "ipi_probability"]
         if c in d.columns),
        None
    )
    if score_col:
        values = pd.to_numeric(d[score_col], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            prob_ok = False
    else:
        prob_ok = False

    # Detect common predicted-label column names
    pred_col = next(
        (c for c in ["y_pred", "predicted_label", "prediction", "pred"]
         if c in d.columns),
        None
    )
    if pred_col:
        values = pd.to_numeric(d[pred_col], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            label_ok = False
    else:
        label_ok = False

report("Validation prediction row counts (73 / 146)", row_ok)
report("Consistent prediction schemas", len(schemas) == 1,
       str({str(k): v for k, v in schemas.items()}))
report("Unique, non-missing trajectory IDs", id_ok)
report("Probability values valid [0,1]", prob_ok)
report("Predicted labels binary", label_ok)

# 3. Validation metrics
metrics_path = DEV / "validation_metrics_all.csv"
metrics_ok = False

if metrics_path.exists():
    m = pd.read_csv(metrics_path)
    required = {
        "experiment_id", "feature_version", "feature_regime",
        "algorithm", "strategy", "threshold"
    }
    metrics_ok = len(m) == 72 and required.issubset(m.columns)
    report("Validation metrics structure", metrics_ok,
           f"rows={len(m)}, columns={list(m.columns)}")

    if "threshold" in m.columns:
        t = pd.to_numeric(m["threshold"], errors="coerce")
        report("Validation thresholds valid", t.notna().all() and t.between(0, 1).all())
else:
    report("Validation metrics file exists", False)

# 4. Hyperparameter search summary
hp_path = DEV / "hyperparameter_search_summary.csv"

if hp_path.exists():
    h = pd.read_csv(hp_path)
    required = {"experiment_id", "best_cv_average_precision", "best_params", "model_sha256"}
    hp_ok = len(h) == 72 and required.issubset(h.columns)
    report("Hyperparameter summary structure", hp_ok,
           f"rows={len(h)}, columns={list(h.columns)}")
else:
    report("Hyperparameter summary exists", False)

# 5. Lock manifest and test predictions
score_path = LOCK / "test_scores_locked.csv"
manifest_path = LOCK / "lock_manifest.json"
lock_integrity = False

if score_path.exists() and manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    actual_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
    expected_hash = manifest.get("test_scores_sha256")
    d = pd.read_csv(score_path)

    hash_ok = actual_hash == expected_hash
    rows_ok = len(d) == manifest.get("row_count") == 11232
    sealed_ok = manifest.get("test_labels_opened") is False
    status_ok = manifest.get("status") == "LOCKED_BEFORE_TEST_UNSEAL"

    report("Locked test SHA-256", hash_ok, actual_hash)
    report("Locked test row count", rows_ok, str(len(d)))
    report("Test labels marked sealed", sealed_ok)
    report("Lock status", status_ok, str(manifest.get("status")))

    required = {
        "experiment_id", "feature_version", "feature_regime",
        "algorithm", "training_strategy", "target_model"
    }
    test_schema_ok = required.issubset(d.columns)
    report("Test prediction required columns", test_schema_ok,
           str(list(d.columns)))

    if test_schema_ok and "source_model" in d.columns:
        single = d[d["training_strategy"] == "single_source"]
        pooled = d[d["training_strategy"] == "pooled_leave_one_out"]

        single_cells = single.groupby([
            "feature_version", "feature_regime", "algorithm",
            "source_model", "target_model"
        ]).ngroups

        pooled_cells = pooled.groupby([
            "feature_version", "feature_regime", "algorithm",
            "target_model"
        ]).ngroups

        report("Single-source evaluation cells", single_cells == 108, str(single_cells))
        report("Pooled LOLO evaluation cells", pooled_cells == 36, str(pooled_cells))
        report("Total evaluation cells", single_cells + pooled_cells == 144,
               str(single_cells + pooled_cells))
    else:
        warn("Evaluation cell count", "Cannot verify without source_model column")

    # Verify all model files against hashes recorded in manifest
    model_hashes = manifest.get("models", {})
    model_hash_ok = True

    for name, expected in model_hashes.items():
        path = MODELS / name
        if not path.exists():
            model_hash_ok = False
            print("[FAIL] Missing model:", name)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            model_hash_ok = False
            print("[FAIL] Model hash mismatch:", name)

    report("Model artifact hashes match manifest", model_hash_ok,
           f"{len(model_hashes)} manifest entries")

    lock_integrity = hash_ok and rows_ok and sealed_ok and status_ok and model_hash_ok
else:
    report("Lock files exist", False)

# 6. Final summary
print("\n" + "=" * 68)
print("AUDIT SUMMARY")
print("=" * 68)

passed = sum(s == "PASS" for s, _ in results)
failed = sum(s == "FAIL" for s, _ in results)
warnings = sum(s == "WARN" for s, _ in results)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")
print(f"WARN: {warnings}")

if failed == 0 and lock_integrity:
    print("\nSTRUCTURAL AUDIT: PASS")
    print("This audit is read-only. Test labels were NOT opened.")
    print("It does not independently prove group-leakage safety,")
    print("trajectory-to-ground-truth mapping, or threshold application.")
    print("Source-code review is still required before unsealing.")
else:
    print("\nSTRUCTURAL AUDIT: NOT PASSED")
    print("Resolve failures before considering test unsealing.")

print("=" * 68)
