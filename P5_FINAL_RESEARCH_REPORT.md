# P5 Final Research Validation Report

**Project:** Cross-LLM Generalization of Supervised Behavioral Detection for Indirect Prompt Injection in Tool-Using LLM Agents
**Validation artifact:** `P5_FINAL_VALIDATION.zip`
**Analysis run:** 2026-10-31 10:45:57–10:54:05 UTC
**Environment recorded in manifest:** Python 3.14.3; macOS 26.6.2; arm64

---

## 1. Executive summary

The uploaded P5 archive was inspected directly. The final Extended Analysis run is recorded as **5 PASS and 1 BLOCKED**. The SHAP repair is present in the archived script and its outputs cover all **18/18 model contexts**. Connected-component bootstrap intervals and exploratory baseline results are also present.

A separate integrity comparison was performed between the 182 P4 file hashes recorded in P5 and the previously supplied P4 archive: **182 checked, 0 missing, 0 mismatches**. This verifies the P5 snapshot against that P4 archive; it does not independently verify the current contents of the user's live Mac folder.

**Overall assessment:** P5 is a substantial and reproducible post-hoc analysis package, but it is not a fully completed set of every planned analysis. Feature ablation remains blocked, and all analyses using the unsealed Test set must be presented as exploratory rather than blind confirmatory evidence.

## 2. Validation of the uploaded archive

| Check | Result |
|---|---:|
| Files in P5 ZIP | 96 entries |
| P4 files listed in `source_hashes.csv` | 182 |
| P4 files checked against supplied P4 ZIP | 182 |
| Missing P4 files | 0 |
| SHA-256 / size mismatches | 0 |
| P5 Extended Analysis manifest errors | 0 |
| P4 read-only flag in manifest | `true` |

The archive includes the Extended and Master reports, the extended-analysis script, result tables, provenance manifests, and run logs.

## 3. P5 Extended Analysis status

| Module | Status | Evidence / interpretation |
|---|---|---|
| Input and schema audit | PASS | V77: 1,131 × 78; V69: 1,131 × 70; Train/Validation labels: 897; Test labels: 234 |
| Connected-component audit | PASS | 88 connected components; 0 components spanning splits |
| SHAP | PASS | 18/18 contexts; V77 / V_all; 3 algorithms × 6 training contexts |
| Connected-component bootstrap | PASS | 144 experiment-metric intervals |
| Random / prevalence / rule baselines | PASS | 606 result rows; rule fixed before Test evaluation; exploratory |
| Feature-ablation design | BLOCKED | Feature groups and design matrix generated; no ablation refits run |

The Extended Analysis manifest reports no execution errors. “PASS” means the module completed and generated its stated outputs; it does not mean that every result is confirmatory or free of methodological limitations.

## 4. SHAP analysis

### 4.1 Verified output coverage

The archive contains:
- `shap_global_importance.csv`: 1,386 rows
- `shap_direction_by_model.csv`: 1,386 rows
- `shap_cross_llm_consistency.csv`: 462 rows

The global-importance table contains 18 distinct `(algorithm, strategy, source, target)` contexts, with 77 feature rows per context. Each context explains 78 Test trajectories. Every row records the background as `train_only_sample_100`.

The archived script explicitly:
1. Filters the background cohort to rows whose split is `train`.
2. Samples up to 100 Train rows using the configured random seed.
3. Applies the fitted imputer to background and explanation data.
4. Supplies `data=Zbg` and `feature_perturbation="interventional"` to `TreeExplainer`.
5. Selects class index 1 for list/three-dimensional SHAP outputs.

This supports the claim that the archived implementation uses a Train-only background for LR and tree models. The explained cohort is the post-unseal Test cohort, so the explanations remain **exploratory**.

### 4.2 Feature patterns (descriptive)

Examples of high mean absolute SHAP features in the archived aggregates include:
- **LR, Single:** `n_steps`, `trajectory_length`, `n_domain_switches`, `n_act_retrieve`, `n_unique_tools`.
- **GB, Single:** `n_write`, `mean_args_per_call`, `action_entropy`, `n_steps`, `n_act_transfer_value`.
- **LR, Pooled LOLO:** `trajectory_length`, `max_read_streak`, `n_steps`, `n_domain_switches`, `n_unique_tools`.
- **GB, Pooled LOLO:** `n_steps`, `trans_RE`, `n_writes_new_domain`, `action_entropy`, `n_act_transfer_value`.

These are descriptive model-attribution patterns, not causal evidence and not proof that a feature is universally important across LLMs. SHAP magnitudes should not be directly compared across algorithms without accounting for differences in model output scale and explainer semantics. Interpret direction only within the stated model/context and target-class convention.

### 4.3 SHAP limitations

- SHAP explains post-unseal Test predictions; it is not blind confirmatory analysis.
- Scope is limited to V77 / V_all and the 18 contexts (3 algorithms × 6 training contexts), not every feature regime/version.
- SHAP importance is model- and cohort-dependent; correlated features can share or redistribute attribution.
- The output does not establish causal feature effects.

## 5. Connected-component bootstrap

The output contains **144 intervals** across 72 experiment IDs and two metrics:
- AUROC: 72 intervals
- Average Precision: 72 intervals

The requested bootstrap count is 1,000. The archived results contain:
- 139 intervals with 1,000 valid replicates
- 5 intervals with 999 valid replicates
- 0 intervals with reversed bounds or point estimates outside their reported intervals

The resampling unit is recorded as `connected_component`. The split audit found 88 components and no component crossing Train/Validation/Test splits.

This is a new post-hoc connected-component bootstrap. It must not be conflated with the original P4 bootstrap, which used `user_task_group_id` with `condition_group_id` fallback. The two procedures use different resampling units.

## 6. Baseline comparison

The baseline table has **606 rows**:
- 600 random-score simulations: 200 per target LLM
- 3 exposure-then-sensitive-action rule rows
- 3 prevalence-constant rows

### 6.1 Random-score reference

| Target | Mean random AUROC | SD | Mean random AP | SD |
|---|---:|---:|---:|---:|
| LLM-A | 0.5031 | 0.0648 | 0.4940 | 0.0569 |
| LLM-B | 0.5090 | 0.0766 | 0.3076 | 0.0593 |
| LLM-C | 0.4958 | 0.1107 | 0.1440 | 0.0579 |

The random AUROC means are close to 0.5. Random AP varies with the positive-class prevalence, so it should be interpreted against each target's prevalence rather than compared as if all targets had the same class balance.

### 6.2 Rule heuristic

| Target | n | Positives | AUROC | AP | Balanced accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLM-A | 78 | 36 | 0.5952 | 0.5143 | 0.5952 | 0.5143 | 1.0000 | 0.6792 |
| LLM-B | 78 | 21 | 0.6228 | 0.3281 | 0.6228 | 0.3281 | 1.0000 | 0.4941 |
| LLM-C | 78 | 8 | 0.5446 | 0.1116 | 0.5446 | 0.1129 | 0.8750 | 0.2000 |

The rule is the fixed `exposure_then_sensitive_action` heuristic. It is not a validated domain rule and is explicitly marked `exploratory_test_post_unseal`. Do not present it as a trained or independently validated detector.

The prevalence-constant baseline has no ranking information; AUROC is therefore not informative/defined for that constant score.

## 7. Master analysis: generalization and pooled-vs-single comparison

The Master report records:
- 432 directed formal single-source generalization-gap comparisons.
- 36 pooled LOLO comparison cells.
- 6 matched pooled-vs-single metric tests.
- 144 P4 evaluation metric cells and 11,232 locked predictions in the input audit.
- P4 snapshot audit: 182 checked, 0 missing, 0 mismatches.

Pooled LOLO is reported separately from the formal single-source gap definition. It should not be merged into or substituted for that primary comparison.

### 7.1 Matched pooled-vs-single summaries

| Metric | Mean pooled | Mean single-cross | Mean difference | Holm-adjusted Wilcoxon p |
|---|---:|---:|---:|---:|
| AUROC | 0.8587 | 0.8193 | +0.0394 | 0.000052 |
| Average Precision | 0.7003 | 0.6267 | +0.0736 | 0.000083 |
| Balanced accuracy | 0.7306 | 0.7180 | +0.0125 | 0.654698 |
| F1 | 0.5519 | 0.5446 | +0.0074 | 0.654698 |
| Precision | 0.4263 | 0.4309 | −0.0046 | 0.979538 |
| Recall / TPR | 0.9350 | 0.9005 | +0.0344 | 0.030893 |

The matched analysis uses 36 matched units. The report itself labels these summaries exploratory; the matched units should not automatically be treated as independent observations. These p-values do not establish a general causal or universal advantage of pooled training.

### 7.2 Other Master-report caveats

- P4 validation thresholds were selected by maximum balanced accuracy on Validation; Validation metrics are therefore tuned.
- P4 permutation importance is Validation-based, not Test-level importance.
- P4's original bootstrap resampling unit is not necessarily equivalent to connected components.
- Label sensitivity and Pbase/payload-shift analyses remain blocked because the required prespecified protocol and/or data mapping are absent.

## 8. What is not completed

The following must remain explicitly incomplete:
1. **Feature ablation:** only the feature-group manifest and ablation design matrix were generated. No secondary refitting or ablation performance results exist.
2. **Label sensitivity:** no sensitivity results are reported; frozen labels remain unchanged.
3. **Pbase/payload shift:** the required comparison dataset/mapping is unavailable in the supplied analysis.
4. **Text baseline:** no text-baseline result is established by the Master report.
5. **Confirmatory Test analysis:** post-unseal Test analyses are exploratory by design.

No P4 models, labels, scores, thresholds, or manifests were modified by the P5 Extended Analysis, according to its report and manifest.

## 9. Recommended paper framing

A defensible description is:

> We conducted a post-hoc analysis of the frozen P4 detector outputs. The analysis included formal single-source generalization-gap summaries, a separately reported pooled-LOLO comparison, exploratory matched pooled-versus-single statistics, connected-component bootstrap intervals, Train-background SHAP explanations of post-unseal Test predictions, and random/prevalence/rule baseline references. The P4 artifacts were treated as read-only. Because Test labels had been unsealed before these additional analyses, Test-based explanations, intervals, and baseline comparisons are reported as exploratory rather than blind confirmatory evidence. Feature-ablation refits, label-sensitivity analysis, and payload-shift analysis were not completed.

For the paper, keep the following distinctions explicit:
- Primary frozen P4 evaluation vs. P5 post-hoc analysis.
- Single-source generalization gap vs. pooled LOLO.
- Original P4 bootstrap vs. P5 connected-component bootstrap.
- Validation-based feature importance vs. post-unseal Test SHAP.
- Completed analysis modules vs. blocked modules.

## 10. Final verdict

**P5 Extended Analysis: technically completed for the modules marked PASS, with feature-ablation design still BLOCKED.** The uploaded archive's SHAP outputs now cover all 18 contexts and record a Train-only background; the P4 snapshot hash check passes against the supplied P4 archive.

This is ready to serve as the evidence base for a carefully qualified research write-up. It is not accurate to claim that all planned P5 analyses are complete, that feature ablation was performed, or that post-unseal Test analyses are confirmatory.

---

## Appendix: key archived artifacts

- `reports/P5_EXTENDED_REPORT.md`
- `reports/P5_MASTER_REPORT.md`
- `provenance/p5_extended_manifest.json`
- `provenance/source_hashes.csv`
- `08_SHAP/results/shap_global_importance.csv`
- `08_SHAP/results/shap_direction_by_model.csv`
- `08_SHAP/results/shap_cross_llm_consistency.csv`
- `10_STATISTICAL_INFERENCE/results/connected_component_bootstrap_ci.csv`
- `11_BASELINE_COMPARISON/results/baseline_test_metrics.csv`
- `11_BASELINE_COMPARISON/results/random_baseline_summary.csv`
- `reports/tables/pooled_vs_single_paired_statistics.csv`
