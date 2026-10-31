#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
P4="$ROOT/P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031"
PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || { echo "Missing P5 .venv: $PY"; exit 2; }
[ -d "$P4" ] || { echo "Missing P4 folder: $P4"; exit 2; }
mkdir -p provenance/run_logs reports/tables reports/figures \
  00_compliance 01_generalization_gap/results \
  02_feature_importance/results 03_distribution_shift/results
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="provenance/run_logs/p5_master_${STAMP}.log"
echo "P5 MASTER ANALYSIS — $STAMP"
echo "P4 will be read-only. Outputs will be written to P5."
"$PY" -u master_analysis.py --p4 "$P4" --out "$PWD" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo
echo "Review reports/P5_MASTER_REPORT.md and $LOG"
exit "$RC"
