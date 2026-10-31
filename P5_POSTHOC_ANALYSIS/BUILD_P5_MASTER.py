from pathlib import Path
import json
import hashlib
import platform
import subprocess
import sys
import csv
import importlib.metadata as md
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
P4 = ROOT / "P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031"
P5 = ROOT / "P5_POSTHOC_ANALYSIS"

OUT = P5 / "provenance"
OUT.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def command_version(cmd):
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception as e:
        return f"UNAVAILABLE: {e}"

def package_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

# Required files
required = [
    "config/experiment_config.json",
    "data/features_p3_77_1131.csv",
    "data/features_p4_69_1131.csv",
    "data/metadata_membership_1131.csv",
    "data/train_validation_labels_897.csv",
    "outputs/after_unseal_inputs/test_labels_234.csv",
    "outputs/locked_test_scores/lock_manifest.json",
    "outputs/locked_test_scores/test_scores_locked.csv",
    "outputs/final_test_evaluation/test_metrics_by_cell.csv",
    "outputs/final_test_evaluation/test_predictions_with_labels_after_unseal.csv",
    "outputs/final_test_evaluation/group_bootstrap_95ci.csv",
    "outputs/final_test_evaluation/fair_cross_llm_comparison.csv",
    "outputs/models",
    "src/pipeline.py",
]

checks = []
for item in required:
    p = P4 / item
    checks.append({
        "item": item,
        "exists": p.exists(),
        "type": "directory" if p.is_dir() else "file"
    })

# Dependency inventory
packages = [
    "numpy", "pandas", "scipy", "scikit-learn",
    "matplotlib", "seaborn", "joblib", "shap",
    "openpyxl", "imbalanced-learn"
]
deps = {p: package_version(p) for p in packages}

# CSV schema inventory
csv_files = [
    "data/features_p3_77_1131.csv",
    "data/features_p4_69_1131.csv",
    "data/metadata_membership_1131.csv",
    "data/train_validation_labels_897.csv",
    "outputs/after_unseal_inputs/test_labels_234.csv",
    "outputs/locked_test_scores/test_scores_locked.csv",
    "outputs/final_test_evaluation/test_metrics_by_cell.csv",
    "outputs/final_test_evaluation/test_predictions_with_labels_after_unseal.csv",
    "outputs/final_test_evaluation/group_bootstrap_95ci.csv",
    "outputs/final_test_evaluation/fair_cross_llm_comparison.csv",
    "outputs/development/permutation_importance_validation.csv",
    "outputs/development/V69_ks_train.csv",
    "outputs/development/V77_ks_train.csv",
    "outputs/development/hyperparameter_search_summary.csv",
    "outputs/development/validation_metrics_all.csv",
]

schemas = []
try:
    import pandas as pd
    for item in csv_files:
        p = P4 / item
        if not p.exists():
            schemas.append({"file": item, "status": "MISSING"})
            continue
        df = pd.read_csv(p, nrows=5)
        schemas.append({
            "file": item,
            "status": "READABLE",
            "columns": list(df.columns),
            "sample_rows": len(df.columns) and len(df)
        })
except Exception as e:
    schemas.append({"status": "PANDAS_READ_ERROR", "error": str(e)})

# P4 file integrity comparison against the previously created snapshot
snapshot = P5 / "provenance/source_hashes.csv"
integrity = {
    "snapshot_exists": snapshot.exists(),
    "checked": 0,
    "mismatches": [],
    "missing": []
}

if snapshot.exists():
    with snapshot.open(newline="") as f:
        for row in csv.DictReader(f):
            p = P4 / row["relative_path"]
            if not p.exists():
                integrity["missing"].append(row["relative_path"])
                continue
            integrity["checked"] += 1
            actual = sha256(p)
            if actual != row["sha256"]:
                integrity["mismatches"].append(row["relative_path"])

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "git": command_version(["git", "--version"]),
    "p4_root": str(P4),
    "p5_root": str(P5),
    "required_files": checks,
    "dependencies": deps,
    "csv_schemas": schemas,
    "p4_integrity": integrity,
    "safety": {
        "p4_modified": False,
        "locked_scores_modified": False,
        "models_modified": False,
        "analysis_executed": False
    }
}

out = OUT / "p5_preflight_audit.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

# Human-readable summary
lines = [
    "P5 MASTER PREFLIGHT AUDIT",
    "=" * 60,
    f"Python: {sys.version.split()[0]}",
    f"Platform: {platform.platform()}",
    "",
    "REQUIRED INPUTS"
]
for x in checks:
    lines.append(f"[{'OK' if x['exists'] else 'MISSING'}] {x['item']}")

lines += ["", "DEPENDENCIES"]
for k, v in deps.items():
    lines.append(f"{k}: {v or 'NOT INSTALLED'}")

lines += ["", "P4 INTEGRITY"]
lines.append(f"Snapshot exists: {integrity['snapshot_exists']}")
lines.append(f"Files checked: {integrity['checked']}")
lines.append(f"Mismatches: {len(integrity['mismatches'])}")
lines.append(f"Missing: {len(integrity['missing'])}")

lines += ["", "CSV SCHEMAS"]
for s in schemas:
    lines.append(f"\n{s['file']}: {s['status']}")
    if "columns" in s:
        lines.append("  " + ", ".join(s["columns"]))

summary = OUT / "p5_preflight_audit.txt"
summary.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("\nJSON report:", out)
print("Text report:", summary)
print("\nPRE-FLIGHT COMPLETE. No analysis was run.")
