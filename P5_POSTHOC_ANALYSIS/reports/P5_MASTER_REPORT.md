# P5 Master Post-hoc Analysis Report

Started UTC: 2026-10-31T08:23:50.250440+00:00
Finished UTC: 2026-10-31T08:23:50.487597+00:00
P4 read-only: YES

## Module status

| module                                   | status   | detail                                                                                   |
|:-----------------------------------------|:---------|:-----------------------------------------------------------------------------------------|
| Input and schema audit                   | PASS     | metrics=144, locked=11232, labels=234                                                    |
| Formal single-source generalization gaps | PASS     | 432 directed comparisons                                                                 |
| Pooled LOLO comparison                   | PASS     | 36 cells; kept separate from formal single-source gap                                    |
| Matched pooled-vs-single statistics      | PASS     | 6 metric tests; exploratory matched summaries                                            |
| Existing P4 analysis review              | PASS     | Copied bootstrap CI, KS, validation permutation and fair comparison tables               |
| Frozen artifact count audit              | PASS     | Counts recorded; source P4 files not modified                                            |
| SHAP                                     | BLOCKED  | Requires prespecified background/explanation cohort and estimator-specific validation.   |
| Feature ablation                         | BLOCKED  | Requires a prespecified ablation design and secondary model training; P4 remains frozen. |
| Random/rule/text baselines               | BLOCKED  | Baseline definitions/implementations are not present in supplied P4 artifacts.           |
| Label sensitivity                        | BLOCKED  | Requires a prespecified sensitivity protocol; frozen labels are not modified.            |
| Pbase/payload shift                      | BLOCKED  | Required comparison dataset and mapping are not present in supplied schema.              |
| P4 snapshot verification                 | PASS     | checked=182, missing=0, mismatches=0                                                     |

## Interpretation limitations

- Pooled LOLO is reported separately from the paper's single-source gap definitions.
- Matched tests compare target/configuration/algorithm summaries; they are exploratory and not independent-cell inference.
- P4 permutation importance is validation-based, not a test-level importance estimate.
- P4 validation thresholds were selected on validation by maximum balanced accuracy; validation metrics are tuned.
- P4 bootstrap resamples user_task_group_id with condition_group_id fallback, not necessarily connected components.
- SHAP, ablation, baselines, label sensitivity and Pbase/payload shift are BLOCKED pending a prespecified protocol/data.

## Main outputs
- `01_generalization_gap/results/formal_single_source_gaps.csv`
- `01_generalization_gap/results/generalization_gap_summary.csv`
- `01_generalization_gap/results/pooled_lolo_vs_single_cross.csv`
- `reports/tables/pooled_vs_single_paired_statistics.csv`
- `03_distribution_shift/results/group_bootstrap_ci_review.csv`
- `03_distribution_shift/results/V69_KS_existing.csv`
- `03_distribution_shift/results/V77_KS_existing.csv`
- `02_feature_importance/results/permutation_importance_validation_existing.csv`
- `reports/tables/module_status.csv`
- `provenance/p5_run_manifest.json`

Blocked analyses are reported as blocked; no results are fabricated.