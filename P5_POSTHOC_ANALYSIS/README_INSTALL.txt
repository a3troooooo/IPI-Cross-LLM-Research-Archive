P5 ONE-COMMAND RUNNER

1. Copy RUN_P5_COMPLETE.sh and master_analysis.py into:
   ~/Desktop/IPI-Cross-LLM-Research-Archive/P5_POSTHOC_ANALYSIS/

2. In Terminal:
   cd ~/Desktop/IPI-Cross-LLM-Research-Archive/P5_POSTHOC_ANALYSIS
   chmod +x RUN_P5_COMPLETE.sh
   ./RUN_P5_COMPLETE.sh

Uses the existing .venv; no package installation is attempted.
Reads P4 from the sibling folder and writes only under P5.

This runner executes:
- input/count audit
- formal single-source directed generalization gaps (source/target definitions)
- separate pooled-LOLO vs mean single-source cross-target comparison
- exploratory matched pooled-vs-single tests with Holm adjustment
- review/copy of existing P4 bootstrap, KS, validation permutation and fair comparison outputs
- run manifest, module status, Markdown report and run log

SHAP, feature ablation, random/rule/text baselines, label sensitivity and Pbase/payload-shift are marked BLOCKED. They require prespecified methods or additional inputs; the runner does not fabricate them or modify frozen P4 results.
