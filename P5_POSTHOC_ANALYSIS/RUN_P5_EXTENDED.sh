#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
ROOT="$(cd .. && pwd)"
P4="$ROOT/P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031"
PY="$HERE/.venv/bin/python"
if [[ ! -x "$PY" ]]; then echo "ERROR: missing P5 .venv at $PY"; exit 2; fi
if [[ ! -d "$P4" ]]; then echo "ERROR: P4 folder not found: $P4"; exit 2; fi
STAMP="$(date '+%Y%m%d_%H%M%S')"
mkdir -p 08_SHAP/{results,figures} 09_FEATURE_ABLATION/{results,figures} \
  10_STATISTICAL_INFERENCE/{results,figures} 11_BASELINE_COMPARISON/{results,figures} \
  reports provenance/run_logs
LOG="provenance/run_logs/p5_extended_${STAMP}.log"
echo "P5 EXTENDED ANALYSIS — $STAMP"
echo "P4 read-only: $P4"
"$PY" -u p5_extended_analysis.py --p4 "$P4" --p5 "$HERE" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo
echo "Finished with exit code $RC"
echo "Report: reports/P5_EXTENDED_REPORT.md"
echo "Log: $LOG"
exit "$RC"
