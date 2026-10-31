# P5 Extended Analysis Report

Started UTC: 2026-10-31T10:45:57.663977+00:00
Finished UTC: 2026-10-31T10:54:05.257137+00:00
**P4 read-only: YES**

## Module status

                          module  status                                                                                                      detail
                     Input audit    PASS                                       V77=(1131, 78); V69=(1131, 70); train/val labels=897; test labels=234
       Connected-component audit    PASS                                                                 88 components; components spanning splits=0
                            SHAP    PASS                      18/18 model contexts explained; V77/V_all; test explanation is post-unseal exploratory
   Connected-component bootstrap    PASS                                                                             144 experiment-metric intervals
Random/prevalence/rule baselines    PASS                                                    606 rows; rule fixed before test evaluation; exploratory
         Feature ablation design BLOCKED Feature groups and ablation matrix generated; refits require approval of semantic grouping before execution

## Interpretation limitations

- SHAP explanations use a Train-only background but explain post-unseal Test predictions; exploratory, not confirmatory.
- SHAP run is limited to V77/V_all and 18 contexts (3 algorithms x 6 training contexts).
- Feature ablation refits are not run: the generated feature-group taxonomy must be approved before secondary refitting.
- Connected-component bootstrap resamples components; this is a new post-hoc interval analysis, not the original P4 bootstrap.
- Rule baseline is a fixed exposure-then-sensitive-action heuristic, not a validated domain rule; report as exploratory.
- Random baseline uses 200 Monte Carlo score draws per target; prevalence constant has no AUROC ranking information.
- No P4 models, labels, scores, thresholds, or manifests are modified.

## Outputs
- `08_SHAP/results/shap_global_importance.csv`
- `08_SHAP/results/shap_direction_by_model.csv`
- `08_SHAP/results/shap_cross_llm_consistency.csv`
- `09_FEATURE_ABLATION/results/feature_group_manifest.csv`
- `09_FEATURE_ABLATION/results/ablation_design_matrix.csv`
- `10_STATISTICAL_INFERENCE/results/connected_component_bootstrap_ci.csv`
- `11_BASELINE_COMPARISON/results/baseline_test_metrics.csv`
- `11_BASELINE_COMPARISON/results/random_baseline_summary.csv`
- `reports/P5_EXTENDED_MODULE_STATUS.csv`
- `provenance/p5_extended_manifest.json`

Do not describe blocked modules as completed. All Test-based post-unseal analyses are exploratory.