from pathlib import Path
import ast
import hashlib
import json
import re
import sys
import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / "src/pipeline.py"
CFG = ROOT / "config/experiment_config.json"
VEXP = ROOT / "config/V_exp_manifest.json"
DEV = ROOT / "outputs/development"
MODELS = ROOT / "outputs/models"
LOCK = ROOT / "outputs/locked_test_scores"

results = []

def report(section, name, status, detail=""):
    results.append((section, status, name))
    print(f"[{status}] {section}: {name}" +
          (f" | {detail}" if detail else ""))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

def function_source(tree, source, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(source.splitlines()[node.lineno-1:node.end_lineno])
    return ""

print("\n" + "=" * 78)
print("P4 FULL PRE-UNSEAL AUDIT — READ ONLY")
print("=" * 78)
print("Project:", ROOT.resolve())

# ---------------------------------------------------------
# 1. SOURCE CODE INTEGRITY AND STATIC REVIEW
# ---------------------------------------------------------
print("\n[1] SOURCE CODE AND METHODOLOGY")

if not SRC.exists():
    report("SOURCE", "src/pipeline.py exists", "FAIL")
    sys.exit(1)

source = SRC.read_text()
try:
    tree = ast.parse(source)
    report("SOURCE", "Python source parses successfully", "PASS")
except SyntaxError as e:
    report("SOURCE", "Python source parses successfully", "FAIL", str(e))
    sys.exit(1)

function_names = {
    n.name for n in ast.walk(tree)
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
}

required_functions = [
    "load_data", "get_feature_cols", "make_groups",
    "assert_group_disjoint", "threshold_by_balanced_accuracy",
    "train_one", "save_dev_artifacts",
    "create_locked_predictions", "evaluate_after_unseal"
]

for name in required_functions:
    report("SOURCE", f"Function {name} exists",
           "PASS" if name in function_names else "FAIL")

# Print relevant functions for human review
for name in [
    "load_data", "get_feature_cols", "make_groups",
    "assert_group_disjoint", "threshold_by_balanced_accuracy",
    "train_one", "create_locked_predictions",
    "evaluate_after_unseal"
]:
    body = function_source(tree, source, name)
    if body:
        print(f"\n--- SOURCE: {name} ---")
        print(body)

# Static checks: these are indicators, not proof of correctness
checks = {
    "GroupKFold imported": "GroupKFold" in source,
    "GroupKFold receives groups": bool(re.search(r"GroupKFold", source))
                                      and "groups=gtr" in source,
    "RandomizedSearchCV receives groups": "search.fit(Xtr,ytr,groups=gtr)" in source
                                          or "search.fit(Xtr, ytr, groups=gtr)" in source,
    "Validation probabilities generated": "predict_proba(Xval)" in source,
    "Validation threshold selected": "threshold_by_balanced_accuracy(yval,val_scores)" in source
                                     or "threshold_by_balanced_accuracy(yval, val_scores)" in source,
    "Test predictions use predict_proba": "predict_proba(X)" in source,
    "Test predictions store threshold": '"threshold":thr' in source
                                        or '"threshold": thr' in source,
    "Test predictions store predicted label": "predicted_label" in source,
    "Bootstrap code exists": "bootstrap" in source.lower(),
}

for name, ok in checks.items():
    report("STATIC", name, "PASS" if ok else "REVIEW")

# Verify that lock-generation function does not explicitly call label loading
lock_fn = function_source(tree, source, "create_locked_predictions")
label_access_patterns = [
    r"test_labels", r"labels_path", r"ipi_affected",
    r"train_validation_labels", r"blind_test_labels"
]
label_hits = [p for p in label_access_patterns
              if re.search(p, lock_fn, re.I)]
report("TEST ISOLATION",
       "No obvious label-loading references inside create_locked_predictions",
       "PASS" if not label_hits else "REVIEW",
       ", ".join(label_hits) if label_hits else "No matching references")

# Ensure unseal evaluation is a separate function
eval_fn = function_source(tree, source, "evaluate_after_unseal")
report("TEST ISOLATION",
       "Post-unseal evaluation is separated into its own function",
       "PASS" if bool(eval_fn) else "FAIL")

# ---------------------------------------------------------
# 2. CONFIGURATION AND FEATURE MANIFEST
# ---------------------------------------------------------
print("\n[2] EXPERIMENT CONFIGURATION")

config = read_json(CFG)
vexp = read_json(VEXP)

report("CONFIG", "Experiment config exists", "PASS" if config is not None else "FAIL")
report("CONFIG", "V_exp manifest exists", "PASS" if vexp is not None else "FAIL")

if config is not None:
    print("Experiment config:")
    print(json.dumps(config, indent=2))

if vexp is not None:
    print("\nV_exp manifest:")
    print(json.dumps(vexp, indent=2))

# ---------------------------------------------------------
# 3. INPUT DATA SCHEMAS AND SPLIT INFORMATION
# ---------------------------------------------------------
print("\n[3] INPUT DATA")

data_files = [
    ROOT / "data/features_p3_77_1131.csv",
    ROOT / "data/features_p4_69_1131.csv",
    ROOT / "data/metadata_membership_1131.csv",
    ROOT / "data/train_validation_labels_897.csv",
]

frames = {}

for path in data_files:
    if not path.exists():
        report("DATA", f"{path.name} exists", "FAIL")
        continue

    try:
        df = pd.read_csv(path)
        frames[path.name] = df
        report("DATA", f"{path.name} readable", "PASS",
               f"shape={df.shape}; sha256={sha256(path)}")
        print("  Columns:", df.columns.tolist())
    except Exception as e:
        report("DATA", f"{path.name} readable", "FAIL", str(e))

f77 = frames.get("features_p3_77_1131.csv")
f69 = frames.get("features_p4_69_1131.csv")
meta = frames.get("metadata_membership_1131.csv")
labels = frames.get("train_validation_labels_897.csv")

if f77 is not None and f69 is not None:
    id_col = "trajectory_id"
    if id_col in f77.columns and id_col in f69.columns:
        report("FEATURES", "V77 and V69 trajectory IDs match",
               "PASS" if set(f77[id_col]) == set(f69[id_col]) else "FAIL")
        report("FEATURES", "V77 has 77 feature columns",
               "PASS" if len(f77.columns) - 1 == 77 else "REVIEW",
               str(len(f77.columns)-1))
        report("FEATURES", "V69 has 69 feature columns",
               "PASS" if len(f69.columns) - 1 == 69 else "REVIEW",
               str(len(f69.columns)-1))
    else:
        report("FEATURES", "Trajectory ID columns available", "FAIL")

if meta is not None:
    print("\nMembership preview:")
    print(meta.head(5).to_string(index=False))

if labels is not None:
    print("\nTrain/Validation label preview:")
    print(labels.head(5).to_string(index=False))
    print("\nLabel columns:", labels.columns.tolist())

# ---------------------------------------------------------
# 4. GROUP LEAKAGE DATA REVIEW
# ---------------------------------------------------------
print("\n[4] GROUP LEAKAGE REVIEW")

if meta is not None:
    print("Metadata columns:", meta.columns.tolist())

    id_col = next((c for c in ["trajectory_id"] if c in meta.columns), None)
    split_col = next((c for c in meta.columns
                      if c.lower() in ["split", "partition", "dataset_split"]), None)

    group_cols = [c for c in meta.columns
                  if c.lower() in [
                      "condition_group_id", "user_task_group_id"
                  ]]

    if id_col and split_col:
        print("Split counts:")
        print(meta[split_col].value_counts(dropna=False).to_string())
        print("Group columns:", group_cols)

        if group_cols:
            for col in group_cols:
                tmp = meta[[split_col, col]].dropna()
                cross = tmp.groupby(col)[split_col].nunique()
                overlap = int((cross > 1).sum())
                report("GROUP", f"{col} does not cross splits",
                       "PASS" if overlap == 0 else "FAIL",
                       f"{overlap} groups appear in multiple splits")
        else:
            report("GROUP", "Frozen group columns found",
                   "REVIEW", "No recognized group ID columns in metadata")
    else:
        report("GROUP", "Split and trajectory columns identifiable",
               "REVIEW", f"id={id_col}; split={split_col}")
else:
    report("GROUP", "Membership data available", "FAIL")

# ---------------------------------------------------------
# 5. DEVELOPMENT OUTPUTS
# ---------------------------------------------------------
print("\n[5] DEVELOPMENT OUTPUTS")

pred_files = sorted(DEV.glob("*validation_predictions.csv"))
model_files = sorted(MODELS.glob("*.joblib"))

report("OUTPUT", "72 validation prediction files",
       "PASS" if len(pred_files) == 72 else "FAIL", str(len(pred_files)))
report("OUTPUT", "72 model files",
       "PASS" if len(model_files) == 72 else "FAIL", str(len(model_files)))

schema_set = set()
prediction_valid = True
prediction_rows_valid = True
threshold_valid = True

for path in pred_files:
    try:
        d = pd.read_csv(path)
        schema_set.add(tuple(d.columns))

        pooled = "POOLED" in path.name
        expected_rows = 146 if pooled else 73

        if len(d) != expected_rows:
            prediction_rows_valid = False

        required = {"trajectory_id", "y_true", "score", "threshold"}
        if not required.issubset(d.columns):
            prediction_valid = False
            continue

        if d.trajectory_id.isna().any() or d.trajectory_id.duplicated().any():
            prediction_valid = False

        if d.score.isna().any() or not d.score.between(0, 1).all():
            prediction_valid = False

        if d.y_true.isna().any() or not d.y_true.isin([0, 1]).all():
            prediction_valid = False

        if d.threshold.isna().any() or not d.threshold.between(0, 1).all():
            threshold_valid = False

    except Exception:
        prediction_valid = False

report("OUTPUT", "Validation prediction schema and values",
       "PASS" if prediction_valid and len(schema_set) == 1 else "FAIL")
report("OUTPUT", "Validation prediction row counts",
       "PASS" if prediction_rows_valid else "FAIL")
report("OUTPUT", "Validation thresholds valid",
       "PASS" if threshold_valid else "FAIL")

metrics_path = DEV / "validation_metrics_all.csv"
hp_path = DEV / "hyperparameter_search_summary.csv"

if metrics_path.exists():
    m = pd.read_csv(metrics_path)
    report("OUTPUT", "72 validation metric records",
           "PASS" if len(m) == 72 else "FAIL", str(len(m)))
    print("Validation metric columns:", m.columns.tolist())
else:
    report("OUTPUT", "Validation metrics exist", "FAIL")

if hp_path.exists():
    h = pd.read_csv(hp_path)
    report("OUTPUT", "72 hyperparameter records",
           "PASS" if len(h) == 72 else "FAIL", str(len(h)))
else:
    report("OUTPUT", "Hyperparameter summary exists", "FAIL")

# ---------------------------------------------------------
# 6. LOCKED TEST PREDICTIONS AND MODEL HASHES
# ---------------------------------------------------------
print("\n[6] LOCKED TEST INTEGRITY")

score_path = LOCK / "test_scores_locked.csv"
manifest_path = LOCK / "lock_manifest.json"
lock_integrity = False

if score_path.exists() and manifest_path.exists():
    manifest = read_json(manifest_path)
    actual_hash = sha256(score_path)
    d = pd.read_csv(score_path)

    hash_ok = actual_hash == manifest.get("test_scores_sha256")
    row_ok = len(d) == manifest.get("row_count") == 11232
    sealed_ok = manifest.get("test_labels_opened") is False
    status_ok = manifest.get("status") == "LOCKED_BEFORE_TEST_UNSEAL"

    report("LOCK", "Test prediction SHA-256",
           "PASS" if hash_ok else "FAIL", actual_hash)
    report("LOCK", "Test prediction row count",
           "PASS" if row_ok else "FAIL", str(len(d)))
    report("LOCK", "Test labels remain sealed",
           "PASS" if sealed_ok else "FAIL")
    report("LOCK", "Lock status",
           "PASS" if status_ok else "FAIL", str(manifest.get("status")))

    model_hash_ok = True
    model_entries = manifest.get("models", {})

    for name, expected in model_entries.items():
        path = MODELS / name
        if not path.exists() or sha256(path) != expected:
            model_hash_ok = False

    report("LOCK", "Model hashes match manifest",
           "PASS" if model_hash_ok else "FAIL",
           f"{len(model_entries)} model entries")

    required = {
        "experiment_id", "training_strategy", "source_model",
        "target_model", "feature_version", "feature_regime",
        "algorithm", "trajectory_id", "score", "threshold",
        "predicted_label"
    }
    schema_ok = required.issubset(d.columns)
    report("LOCK", "Locked prediction schema",
           "PASS" if schema_ok else "FAIL")

    if schema_ok:
        score_ok = d.score.notna().all() and d.score.between(0, 1).all()
        threshold_ok = d.threshold.notna().all() and d.threshold.between(0, 1).all()
        label_ok = d.predicted_label.isin([0, 1]).all()
        reconstructed = (d.score >= d.threshold).astype(int)
        mapping_ok = (reconstructed == d.predicted_label).all()

        report("LOCK", "Test scores valid", "PASS" if score_ok else "FAIL")
        report("LOCK", "Test thresholds valid", "PASS" if threshold_ok else "FAIL")
        report("LOCK", "Test predicted labels binary", "PASS" if label_ok else "FAIL")
        report("LOCK", "Test labels match score/threshold",
               "PASS" if mapping_ok else "FAIL")

        single = d[d.training_strategy == "single_source"]
        pooled = d[d.training_strategy == "pooled_leave_one_out"]

        single_cells = single.groupby([
            "feature_version", "feature_regime", "algorithm",
            "source_model", "target_model"
        ]).ngroups

        pooled_cells = pooled.groupby([
            "feature_version", "feature_regime", "algorithm",
            "target_model"
        ]).ngroups

        report("LOCK", "Single-source cells",
               "PASS" if single_cells == 108 else "FAIL", str(single_cells))
        report("LOCK", "Pooled LOLO cells",
               "PASS" if pooled_cells == 36 else "FAIL", str(pooled_cells))
        report("LOCK", "Total evaluation cells",
               "PASS" if single_cells + pooled_cells == 144 else "FAIL",
               str(single_cells + pooled_cells))

    lock_integrity = hash_ok and row_ok and sealed_ok and status_ok and model_hash_ok
else:
    report("LOCK", "Locked files exist", "FAIL")

# ---------------------------------------------------------
# 7. STATISTICAL ANALYSIS AND INTERPRETABILITY READINESS
# ---------------------------------------------------------
print("\n[7] ANALYSIS READINESS")

for filename in [
    "V69_ks_train.csv",
    "V77_ks_train.csv",
    "V69_train_feature_correlations.csv",
    "V77_train_feature_correlations.csv",
    "high_correlation_pairs_train.csv",
    "permutation_importance_validation.csv",
]:
    path = DEV / filename
    report("ANALYSIS", filename, "PASS" if path.exists() else "REVIEW")

for keyword, description in [
    ("shap", "SHAP implementation"),
    ("holm", "Holm multiple-comparison correction"),
    ("cohen", "Effect-size calculation"),
    ("paired", "Paired statistical tests"),
    ("bootstrap", "Bootstrap confidence intervals"),
]:
    found = keyword in source.lower()
    report("ANALYSIS", description,
           "PASS" if found else "REVIEW",
           "Source keyword present" if found else "Not detected in pipeline source")

# ---------------------------------------------------------
# 8. FINAL REPORT
# ---------------------------------------------------------
print("\n" + "=" * 78)
print("FINAL AUDIT SUMMARY")
print("=" * 78)

counts = {
    s: sum(1 for _, status, _ in results if status == s)
    for s in ["PASS", "FAIL", "REVIEW"]
}

for status, count in counts.items():
    print(f"{status}: {count}")

print("\nIMPORTANT INTERPRETATION:")
print("- PASS means the specific structural check passed.")
print("- REVIEW means manual source/protocol inspection is required.")
print("- Static source checks do not prove absence of data leakage.")
print("- This script did not open test labels or modify experiment outputs.")

if counts["FAIL"] == 0 and lock_integrity:
    print("\nSTRUCTURAL AUDIT: NO STRUCTURAL FAILURES DETECTED")
    print("This is NOT authorization to unseal test labels.")
    print("Review all REVIEW items and verify frozen ground-truth mapping")
    print("and group-disjointness before unsealing.")
else:
    print("\nSTRUCTURAL AUDIT: FAILURES DETECTED")
    print("Resolve failures before proceeding.")

print("=" * 78)
