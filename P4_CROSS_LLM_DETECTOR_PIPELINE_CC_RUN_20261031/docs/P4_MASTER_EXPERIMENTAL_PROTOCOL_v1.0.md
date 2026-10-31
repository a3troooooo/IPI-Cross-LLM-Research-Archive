# P4_MASTER_EXPERIMENTAL_PROTOCOL_v1.0

**Project:** Cross-LLM Generalization of Behavioral Detection for
Indirect Prompt Injection in LLM Agents\
**Owner:** P4\
**Version:** 1.0\
**Date:** 31 October 2026\
**Status:** Working protocol --- approved design decisions recorded; a
small number of operational choices are explicitly marked for freeze
before training.\
**Scope:** Supervised, action-level behavioral IPI detection using the
frozen P3/P4 dataset.

------------------------------------------------------------------------

## 1. Purpose

This protocol defines the experimental design, model-development
process, evaluation, statistical analysis, blind-test procedure,
reproducibility requirements, and reporting outputs for P4.

The study has four connected aims:

1.  Develop supervised detectors for action-level behavioral indirect
    prompt injection (IPI).
2.  Test whether behavioral trajectory features support transfer across
    different LLMs.
3.  Compare full and reduced feature representations, and compare
    all-feature with exposure-focused feature views.
4.  Compare Logistic Regression (LR), Random Forest (RF), and Gradient
    Boosting (GB) under single-source and pooled leave-one-LLM-out
    training.

The goal is not to maximize a test score at any cost. The goal is to
produce a reproducible, leakage-controlled evaluation and report
positive, neutral, and negative findings honestly.

## 2. Authority, source documents, and decision hierarchy

### 2.1 P4 authority

P4 is responsible for selecting and implementing preprocessing, model
configurations, hyperparameter search, cross-validation, threshold
selection, ablations, error analysis, statistical analysis, code, and
reporting within the frozen research objective. The P4 Action Plan and
Methodology and Autonomy Charter explicitly grant this autonomy without
requiring P1 approval for every method choice.

The reference scripts in the P4 package are examples of mechanics and
file formats. They are not a ceiling on the experimental design. P4 may
extend the pipeline where needed, while preserving all frozen inputs and
invariants.

### 2.2 Non-negotiable invariants

The following must remain unchanged:

-   Ground-truth labels and their frozen hash.
-   Dataset membership and canonical condition exclusions.
-   Train, Validation, and Test membership.
-   Group assignments and group-disjoint split rules.
-   Test-label sealing and the blind-test lock procedure.
-   Data provenance and the lineage of all source files.

No sample may be moved between splits. No label may be reclassified. No
predictor may be derived from test labels, post-hoc rationales,
evaluator outcomes, attacker goals, or metadata prohibited below.

### 2.3 Primary target

The sole primary supervised target is `ipi_affected`:

-   `1`: an unauthorized structured tool action causally induced by
    injection.
-   `0`: not affected, including the approved unaffected matched-IPI and
    historical benign controls in the frozen snapshot.

Do not substitute `scenario_type`, benchmark strict success, or exposure
status for the primary target. Exposure is not itself proof of
behavioral compromise. Secondary endpoints may be examined only after
the primary test evaluation and must be reported separately.

## 3. Frozen dataset and split contract

The supplied frozen snapshot contains 1,131 trajectories and three
target LLMs: Gemma, Qwen, and LFM.

  Split          Total trajectories   Per LLM
  ------------ -------------------- ---------
  Train                         678       226
  Validation                    219        73
  Test                          234        78
  **Total**               **1,131**   **377**

The approved train/validation label totals are:

  Split                  `ipi_affected=1`   `ipi_affected=0`   Total
  -------------------- ------------------ ------------------ -------
  Train                               181                497     678
  Validation                           61                158     219
  Train + Validation                  242                655     897

The test labels remain sealed until all test predictions, selected
thresholds, model artifacts, configuration, and hashes have been frozen.
Do not infer, reconstruct, or use test labels before unsealing.

The P4 charter states that train/validation/test membership is
user-task-group-disjoint and that canonical condition exclusions are
frozen. Preserve the supplied group assignments. Do not split individual
model-trajectory rows independently.

### 3.1 Permitted metadata

Use metadata only for grouping, stratification within allowed splits,
audit, and reporting---not as model predictors. In particular, do not
use:

-   `trajectory_id`, `run_id`, model identity, split name, suite,
    condition IDs, user-task IDs;
-   labels, label sources, review tiers/statuses;
-   attacker goal, scenario type, evaluator outcomes, benchmark
    strict-success outcomes;
-   post-hoc rationales or test-label-derived information.

Predictors must be approved structural features from the frozen feature
matrix, or uniformly generated new features with a documented input
contract and a separately versioned experiment. This v1.0 protocol uses
the frozen P3 feature matrices only.

## 4. Feature representations

### 4.1 Feature versions

Both feature versions have equal standing in this study:

-   **V77:** the approved P3 77-feature matrix.
-   **V69:** the P4 69-feature view.

Run both versions under the same split, labels, model families, training
strategies, tuning rules, threshold policy, and evaluation code.
Differences in feature count and membership must be documented. Do not
silently treat V69 as the only headline result or V77 as an informal
afterthought.

The P4 package documents that V69 excludes eight columns from V77 due to
proxy-risk concerns:

`trajectory_length`, `n_calls_without_result`, `n_unparseable_args`,
`ended_with_final`, `final_output_present`, `n_steps`,
`n_agent_messages`, `n_steps_message_and_call`.

The exact feature-column manifests for both versions must be saved with
each experiment. Do not recreate feature matrices by manually deleting
columns without checking them against the frozen manifests.

### 4.2 Feature regimes: V_all and V_exp

-   **V_all:** all features present in the selected feature version (77
    or 69), subject to the predictor exclusions in Section 3.1.
-   **V_exp:** an exposure-focused feature view intended to test how
    much predictive information is associated with the structure and
    provenance of actions following exposure to third-party free text.

The feature dictionary contains groups such as `exposure_structure` and
`provenance`, and marks some provenance/result-length features as
text-derived. However, the supplied Action Plan and Charter do not
provide a complete, authoritative V_exp column manifest. Therefore, the
exact V_exp membership is an **operational freeze item**: P4 must
produce and review a named CSV/JSON manifest before training. The
manifest must list every included and excluded column, feature group,
rationale, and source dictionary entry.

Recommended construction rule for V_exp: include only pre-existing,
uniformly extracted structural features whose definitions explicitly
represent free-text exposure context or the provenance relationship
between prior free-text tool results and later tool-action arguments. Do
not add raw free text, attacker goals, label-derived information, or
post-hoc annotations. Do not infer that every feature in a group is
automatically admissible; review each feature's definition and input
contract.

For clarity, V_exp is not assumed to be "V_all minus exposure features."
It is a separately specified exposure-focused subset. If P4 intends a
different meaning for the abbreviation V_exp, that meaning and its
manifest must be frozen before running experiments. Until then, V_all
can be prepared, but V_exp experiments must not start.

### 4.3 Feature analysis and leakage review

Before fitting, produce a feature audit table containing:

-   feature name, version membership, feature group, definition, data
    type;
-   whether the feature is text-derived, timing-derived,
    exposure-related, or provenance-related;
-   missingness, non-finite count, unique values, constant/near-constant
    status;
-   correlation and multicollinearity diagnostics on Train only;
-   potential target-proxy or process-artifact risk and the decision
    taken.

Do not remove features based on Test performance. Any feature exclusion
beyond the frozen V69 definition must be predeclared, justified,
versioned, and applied consistently.

## 5. Experimental design

### 5.1 Model families

Use exactly these three model families:

1.  Logistic Regression (LR).
2.  Random Forest (RF).
3.  Gradient Boosting (GB).

Do not add other algorithm families to the confirmatory experiment.
Algorithm 5 and Algorithm 6 remain the project-defined components; their
exact names and responsibilities must be taken from their existing
project specifications rather than invented or renamed in this protocol.
They may implement the approved methods and evaluation stages, but they
do not authorize adding a fourth model family.

### 5.2 Single-source training: full 3×3 matrix

Train one detector using one source LLM's Train partition, then evaluate
it on the Test partition of each target LLM.

  Detector training source      Gemma Test    Qwen Test    LFM Test
  -------------------------- ------------- ------------ -----------
  Gemma Train                  Gemma→Gemma   Gemma→Qwen   Gemma→LFM
  Qwen Train                    Qwen→Gemma    Qwen→Qwen    Qwen→LFM
  LFM Train                      LFM→Gemma     LFM→Qwen     LFM→LFM

Each algorithm produces 9 source-target cells. The diagonal cells are
within-model evaluations; off-diagonal cells are cross-model transfer
evaluations. Report these separately.

For a source-specific detector, use that source's Train and Validation
partitions for fitting, tuning, and threshold selection. Never use a
target model's Validation labels to adapt a source detector in a
zero-shot transfer cell.

### 5.3 Pooled leave-one-LLM-out training

For each target LLM, pool the Train partitions of the other two LLMs and
evaluate on the held-out target LLM's Test partition.

  Held-out target   Pooled Train sources   Test target
  ----------------- ---------------------- -------------
  Gemma             Qwen + LFM             Gemma
  Qwen              Gemma + LFM            Qwen
  LFM               Gemma + Qwen           LFM

For tuning and threshold selection, use only the Validation partitions
of the two source LLMs included in the pool. Do not use the held-out
target's Validation labels for model selection or threshold adaptation;
otherwise the experiment is no longer strict leave-one-LLM-out zero-shot
evaluation.

### 5.4 Experiment count

The confirmatory matrix is:

-   2 feature versions: V77 and V69.
-   2 feature regimes: V_all and V_exp.
-   3 algorithms: LR, RF, GB.
-   9 single-source source-target cells per algorithm.
-   3 pooled held-out-target cells per algorithm.

Therefore:

`2 × 2 × 3 × (9 + 3) = 144` primary
algorithm/feature/training/evaluation cells.

The 144 count excludes CV folds, hyperparameter trials, bootstrap
replicates, model diagnostics, and supplementary analyses. The count
assumes each V_exp manifest is valid for both feature versions; if a
feature is absent from one version, record the resulting manifest
difference rather than silently substituting another feature.

## 6. Preprocessing and model development

### 6.1 General rules

All learned preprocessing must be fitted on the relevant training data
only and then applied unchanged to Validation and Test:

-   imputation;
-   scaling;
-   feature selection, if any;
-   resampling or oversampling;
-   class weights;
-   calibration;
-   hyperparameter optimization.

Use a reproducible `Pipeline`/`ColumnTransformer`-style implementation
so that transformations are refit correctly inside each CV fold. Do not
preprocess the full dataset before splitting or before CV.

### 6.2 Suggested preprocessing by model

-   **LR:** median imputation where needed; standardization; class
    weighting (`balanced`) as a candidate. Scaling must be fitted inside
    the pipeline.
-   **RF:** median imputation where needed; scaling is generally
    unnecessary; compare class weighting with unweighted fitting.
-   **GB:** median imputation where needed; scaling is generally
    unnecessary for tree-based gradient boosting. Use the chosen
    implementation consistently and record its version.

Do not assume a specific GB library until implementation is frozen. The
selected implementation must be a conventional Gradient Boosting
classifier, not an additional algorithm family disguised under another
name.

### 6.3 Class imbalance

The frozen train set is imbalanced (181 positive, 497 negative).
Evaluate class-weighted learning as a candidate. Any oversampling must
occur only within each training fold and must never cross group
boundaries or touch Validation/Test. Do not use synthetic samples
generated from validation/test data.

Report the prevalence of each evaluation target. Accuracy alone is not
an adequate metric.

### 6.4 Cross-validation and leakage control

Use group-aware CV on Train only. The primary selection approach should
be **Stratified Group K-Fold** where the available group structure and
class counts make it feasible. If the chosen library cannot support the
required stratification/group constraints, use Group K-Fold and report
fold-level class balance.

The grouping key must preserve the frozen disjointness contract. At
minimum, audit `user_task_group_id` and `condition_group_id`; if both
are present, confirm that no linked group crosses a fold. Do not split
model-trajectory rows independently.

Recommended process:

1.  Generate folds from the relevant Train partition(s) only.
2.  Check group overlap and label counts in every fold.
3.  Run preprocessing and hyperparameter search within the CV training
    folds.
4.  Generate out-of-fold (OOF) predictions for threshold selection and
    diagnostics.
5.  Refit the selected configuration on the full permitted Train
    partition.
6.  Evaluate on the permitted Validation partition once for
    model/threshold selection.

Use a nested group-CV design only if computationally feasible and if it
materially improves unbiased model selection. A practical primary
approach is group-aware inner CV for hyperparameter search, with the
frozen Validation set reserved for final selection. Record any deviation
and its rationale. CV scores are development diagnostics, not
substitutes for the locked Test evaluation.

### 6.5 Hyperparameter optimization

Choose a resource-appropriate search method. Default recommendation:
randomized search over predeclared, bounded parameter distributions
using group-aware CV. Use grid search only for a small, justified search
space. Bayesian optimization is optional only if implementation
complexity and compute budget are justified.

For every model/configuration, save:

-   search space and parameter distributions;
-   search method, number of trials, scoring metric;
-   CV splitter and grouping fields;
-   random seed;
-   best parameters and fold-level results;
-   runtime and software versions.

Use a consistent primary selection metric across models. Recommended
primary development metric: Average Precision (AP/AUPRC), because the
positive class is the minority. Use AUROC and security operating-point
metrics as secondary selection context. If the paper chooses a different
primary selection metric, predeclare it before training.

Do not optimize hyperparameters independently on each Test target.

## 7. Threshold selection and calibration

The predicted score must be oriented so that larger values mean a higher
probability/score of `ipi_affected=1`.

Thresholds must be selected without Test labels. For each detector:

1.  Obtain OOF scores from the permitted source Train data (or pooled
    source Train data).
2.  Use the corresponding permitted Validation data to assess the
    selected configuration and operating point.
3.  Select and freeze one threshold using a predeclared rule.
4.  Apply that threshold unchanged to every Test target evaluated by
    that detector.

For single-source transfer, the same source detector threshold is
applied to all three target LLMs. For pooled leave-one-out, the
threshold is selected from the two source models' development data and
applied to the held-out target.

**Security operating-point rule:** before training, P4 must choose and
document either (a) a maximum acceptable validation FPR constraint and
select the highest-recall threshold satisfying it, or (b) if no
defensible FPR cap is specified, select the threshold maximizing
Validation balanced accuracy, with FPR, FNR, precision, and recall
reported at that threshold. The chosen rule must be identical across the
confirmatory comparisons unless a deviation is explicitly predeclared.

Do not select thresholds from Test ROC/PR curves, Test labels, or Test
subgroup results. Do not recalibrate or adapt thresholds to the held-out
target's Validation/Test labels in zero-shot transfer.

If calibration is used, fit the calibrator on development data only,
with group-aware separation. Report calibration method and diagnostics.
Calibration is optional and must not silently alter score
interpretation.

## 8. Evaluation metrics

### 8.1 Primary and secondary metrics

For every source-target cell and pooled held-out target, report:

-   AUROC;
-   Average Precision (AP; commonly reported as AUPRC);
-   precision / positive predictive value;
-   recall / true positive rate (TPR);
-   F1;
-   balanced accuracy;
-   false positive rate (FPR);
-   false negative rate (FNR);
-   confusion matrix: TP, FP, TN, FN;
-   support and class prevalence.

Report both benign-control FPR and all-negative FPR if the frozen
annotations permit the distinction. State the exact denominator for
each. Do not call one the other.

### 8.2 Aggregation

Report: - per-cell metrics; - per-source macro summaries; - per-target
macro summaries; - diagonal within-model and off-diagonal transfer
summaries; - macro averages across target LLMs; - pooled counts/metrics
only when the pooled estimand and denominator are explicit.

Do not average confusion matrices or rates without preserving their
denominators. Do not treat repeated predictions on the same target
trajectories from different source detectors as independent
observations.

### 8.3 Prevalence and small-sample caution

Test has 78 trajectories per target LLM, and target prevalence differs
by LLM. Always report per-target support and prevalence next to AP/AUPRC
and threshold metrics. Interpret confidence intervals and significance
tests cautiously because the per-target test sample is small.

## 9. Statistical analysis and uncertainty

### 9.1 Confidence intervals

Use 95% group-aware bootstrap confidence intervals. Resample at the
highest relevant independent grouping unit (canonical condition or
user-task group, consistent with the frozen grouping structure), not
individual rows when rows share a group.

For each replicate, recompute the metric from the resampled units. Use a
fixed seed and a documented number of replicates (recommended 2,000 if
computationally practical; fewer only with a stated resource rationale).
If a replicate lacks one class, record the metric as undefined for that
replicate and report the handling rule.

### 9.2 Paired comparisons

Where models or feature versions are evaluated on the same Test
trajectories, use paired comparisons. The 3 source detectors evaluated
on a given target share the same target cases; these are repeated,
paired predictions, not three independent test sets.

Use paired group-bootstrap differences for metrics such as AUROC, AP,
F1, balanced accuracy, and FPR where appropriate. Report the estimated
difference, 95% CI, and effect size. Avoid unpaired tests that treat
repeated target rows as independent.

### 9.3 Multiple comparisons

The design contains many comparisons. Define families of confirmatory
hypotheses before examining Test results. Apply a multiple-comparison
correction (e.g. Holm correction) within each declared family. Clearly
distinguish confirmatory comparisons from exploratory analyses.

### 9.4 Effect sizes

Report metric differences in their natural units and, where suitable, a
standardized effect size. Do not report p-values alone. Interpret
practical significance alongside statistical uncertainty.

## 10. Feature diagnostics and explainability

Run feature diagnostics on development data only unless the analysis is
explicitly post-lock and descriptive.

Required analyses:

1.  **Permutation importance:** compute on Validation or group-aware OOF
    data, not Test for model selection. Use repeated permutations and
    report variability.
2.  **SHAP:** use a model-compatible explainer; document
    background/reference sample selection from Train only. Report
    computational approximations and limitations.
3.  **KS tests / distribution comparison:** compare feature
    distributions across relevant development groups (e.g., affected vs
    not affected, source LLMs) and report effect-size/distribution
    summaries, not p-values alone. Correct for multiple testing.
4.  **Correlation / multicollinearity:** inspect on Train only; report
    highly correlated pairs/clusters. Do not remove features solely
    because of a correlation threshold without a predeclared rationale.
5.  **Feature ablation:** compare V77 vs V69 and V_all vs V_exp under
    matched splits and protocols. Treat these as planned comparisons,
    not ad hoc feature hunting.

Test-set explainability, if produced after unsealing, is strictly
descriptive and must not trigger feature changes or model retraining for
confirmatory claims.

## 11. Error analysis and robustness

After the locked test evaluation, report error analysis using available
approved annotations only:

-   false positives and false negatives by target LLM;
-   error rates by source-target transfer direction;
-   differences in positive prevalence;
-   confidence/score distributions;
-   feature-regime and feature-version differences;
-   any available secondary annotation breakdowns, only when those
    annotations are actually present and approved.

Do not invent attack subtypes or infer causality from feature
importance. Exposure-only evidence is not proof of attention,
compliance, or action-level compromise.

Robustness analyses may include sensitivity to class weighting,
threshold choice, and feature ablation, but all choices must be labeled
as development, confirmatory, or exploratory. Any post-test adjustment
is exploratory and must not be presented as a new blind-test result.

## 12. Blind-test freeze and unsealing protocol

The sealed-label archive and `after_lock` material are intentionally
stored separately from the supplied P4 ZIP. Their separate location is
part of the user's workflow; their absence from the ZIP is not a
missing-release defect.

Before unsealing:

1.  Complete all Train/Validation fitting, tuning, model selection,
    feature-regime definitions, and threshold selection.
2.  Freeze the exact feature manifest for every configuration.
3.  Freeze code commit/version, environment, dependencies, random seeds,
    model parameters, preprocessing, calibration, and threshold.
4.  Generate predictions for all 234 Test feature rows for each selected
    source detector/configuration.
5.  For single-source evaluation, preserve source and target
    identifiers. For pooled leave-one-out, preserve held-out target and
    pooled training-source identifiers.
6.  Save scores with at least: `trajectory_id`, `experiment_id`,
    `training_strategy`, `source_model` or pooled-source descriptor,
    `target_model`, `feature_version`, `feature_regime`, `algorithm`,
    `score`, `threshold`, `predicted_label`.
7.  Verify that each intended target has exactly its frozen Test
    membership and that no extra or missing rows exist.
8.  Hash prediction files, model artifacts, manifests,
    code/configuration, and environment lock file using SHA-256.
9.  Write a lock manifest with status `LOCKED_BEFORE_TEST_UNSEAL`, exact
    prediction-file hash(es), model/configuration hashes, timestamp, and
    experiment IDs.
10. Preserve an immutable copy of the locked artifacts.
11. Only after the lock is independently verified may P4 open the sealed
    test labels and run the final evaluation.

After unsealing, do not change models, features, thresholds, or
predictions for confirmatory results. Any subsequent changes must
receive new experiment IDs and be labeled exploratory; they cannot be
described as a fresh blind-test evaluation on the same test set.

## 13. Reproducibility and artifact requirements

Each experiment must have a unique, stable ID. Recommended naming
pattern:

`P4_{V77|V69}_{VALL|VEXP}_{LR|RF|GB}_{SINGLE|POOLED}_{SOURCE_OR_LOLO}_v1`

Every run must record:

-   dataset/version and input hashes;
-   feature version and exact feature manifest;
-   target and split membership identifiers;
-   training strategy and source/held-out target;
-   algorithm and implementation/library version;
-   preprocessing and class-imbalance settings;
-   CV folds, grouping fields, search space, trials, seeds;
-   selected hyperparameters and threshold rule/value;
-   model artifact hash and prediction hash;
-   start/end time, runtime, hardware/environment;
-   warnings, failures, exclusions, and deviations.

Do not overwrite a previous run. Store immutable run outputs and a
concise index/registry.

## 14. Required deliverables

### 14.1 Machine-readable outputs

-   experiment registry (CSV/JSON);
-   feature manifests for V77, V69, V_all, and V_exp;
-   CV fold and group-integrity reports;
-   hyperparameter-search results;
-   OOF and Validation predictions;
-   locked Test score files;
-   lock manifest and SHA-256 manifest;
-   per-cell metrics and confusion matrices;
-   bootstrap replicate summaries and confidence intervals;
-   paired-comparison and multiple-testing tables;
-   permutation importance, SHAP, KS, and correlation outputs;
-   error-analysis tables.

### 14.2 Figures and paper-ready tables

-   3×3 source-target heatmaps for each algorithm/feature configuration;
-   within-model vs cross-model comparison;
-   pooled leave-one-LLM-out results;
-   ROC and Precision-Recall curves;
-   confusion matrices;
-   threshold operating-point plots;
-   feature importance and SHAP plots;
-   feature distribution/KS summaries;
-   bootstrap confidence-interval plots;
-   comparison tables for V77 vs V69 and V_all vs V_exp;
-   statistical comparison tables with effect sizes and corrected
    p-values.

### 14.3 Written report

-   concise methodology;
-   dataset and split description;
-   model and tuning rationale;
-   threshold policy;
-   evaluation results and uncertainty;
-   cross-LLM generalization findings;
-   feature-regime/version comparison;
-   explainability and feature diagnostics;
-   error analysis;
-   limitations and negative findings;
-   reproducibility and provenance statement.

## 15. Execution phases

  -----------------------------------------------------------------------
  Phase                   Work                    Exit condition
  ----------------------- ----------------------- -----------------------
  1\. Freeze contract     Load approved inputs;   Input/split/label
                          record hashes and       checks pass
                          invariants without      
                          changing them           

  2\. Feature manifests   Freeze V77, V69, V_all, Exact manifests and
                          and V_exp column        rationale saved
                          manifests               

  3\. Leakage-safe data   Build split-aware       No forbidden predictors
  layer                   loading, group checks,  or group overlap
                          CV folds, predictor     
                          exclusions              

  4\. Model development   Implement LR, RF, GB    Reproducible
                          pipelines and           Train/Validation
                          resource-appropriate    selections
                          search                  

  5\. Training matrix     Run single-source 3×3   All planned development
                          and pooled              runs logged
                          leave-one-out           
                          experiments             

  6\. Advanced evaluation Metrics, thresholds,    Development analyses
                          uncertainty, paired     complete; test remains
                          statistics,             sealed
                          explainability,         
                          diagnostics             

  7\. Lock and final      Freeze artifacts, hash, Final locked results
  evaluation              unseal, evaluate once,  and audit trail
                          prepare paper outputs   complete
  -----------------------------------------------------------------------

Algorithm 5 and Algorithm 6 must retain their project-defined meanings.
Their exact definitions should be imported from the existing project
specification when implementing the phases; this protocol does not
invent their identities.

## 16. Research questions and planned comparisons

The planned analyses address:

-   **RQ1 --- Cross-LLM transfer:** How well does a detector trained on
    one LLM transfer to the other LLMs?
-   **RQ2 --- Pooled training:** Does training on two LLMs improve
    performance on a held-out third LLM compared with single-source
    transfer?
-   **RQ3 --- Feature version:** How do V77 and V69 compare under
    matched experimental conditions?
-   **RQ4 --- Feature regime:** How does the exposure-focused V_exp
    representation compare with V_all?
-   **RQ5 --- Algorithm:** How do LR, RF, and GB differ in
    discrimination, operating-point performance, and cross-LLM transfer?
-   **RQ6 --- Behavioral evidence:** Which structural and
    exposure/provenance features are associated with detector behavior,
    and how stable are those explanations?
-   **RQ7 --- Security utility:** What trade-offs arise between
    detection rate and false-positive rate at the frozen operating
    threshold?

No hypothesis is considered supported merely because one cell has the
highest point estimate. Interpret the complete matrix, uncertainty,
prevalence, and paired comparisons.

## 17. Limitations and interpretation rules

-   Test sample size is 78 per target LLM; subgroup estimates may be
    imprecise.
-   Positive prevalence differs across target LLMs, affecting AP,
    precision, and threshold metrics.
-   A model's ability to predict `ipi_affected` does not prove causal
    understanding; this is behavioral detection from observed structural
    features.
-   Feature importance is associational, not causal evidence.
-   Exposure is not equivalent to successful injection or unauthorized
    action.
-   Historical benign controls and matched unaffected IPI records are
    both part of the negative class in the approved frozen snapshot;
    report this composition when interpreting benign FPR.
-   Results from the historical P0 Isolation Forest experiment (1,158
    rows, 69 features, different endpoint) are not directly comparable
    to this supervised action-level experiment. A fair unsupervised
    baseline would require a separately designed experiment using the
    current frozen endpoint and splits; it is outside this v1.0
    confirmatory matrix.

## 18. Pre-run freeze checklist

The following must be completed before the first confirmatory model run:

-   [ ] Confirm input hashes and preserve frozen dataset/split/label
    files.
-   [ ] Save exact V77 and V69 column manifests.
-   [ ] Decide and save exact V_exp membership for both feature
    versions.
-   [ ] Confirm predictor exclusion list.
-   [ ] Select group-aware CV splitter and verify group integrity.
-   [ ] Freeze hyperparameter search spaces, budget, seed, and selection
    metric.
-   [ ] Freeze class-imbalance strategy candidates.
-   [ ] Freeze threshold policy, including any FPR cap.
-   [ ] Freeze metric definitions and statistical comparison families.
-   [ ] Freeze experiment IDs and output directory structure.
-   [ ] Confirm sealed-label location and access procedure without
    opening labels.
-   [ ] Confirm lock-manifest schema and hash verification procedure.

**Stop rule:** If a check fails, stop the affected run and document the
failure. Do not silently alter frozen data or invariants to make the run
proceed.

------------------------------------------------------------------------

## Appendix A --- Known feature-version distinction

The supplied P4 documentation identifies the following eight columns
removed from the 77-feature view to form the 69-feature view:

1.  `trajectory_length`
2.  `n_calls_without_result`
3.  `n_unparseable_args`
4.  `ended_with_final`
5.  `final_output_present`
6.  `n_steps`
7.  `n_agent_messages`
8.  `n_steps_message_and_call`

This list is included for transparency. The exact column manifests in
the frozen feature files remain authoritative.

## Appendix B --- Important terminology

-   **Source LLM:** the LLM whose Train/Validation data are used to
    develop a detector.
-   **Target LLM:** the LLM whose Test trajectories are scored.
-   **Single-source:** one source LLM trains one detector; that detector
    is evaluated on all three target LLMs.
-   **Pooled leave-one-LLM-out:** two LLMs provide development data; the
    third is held out as the target.
-   **V_all:** all permitted features in a selected feature version.
-   **V_exp:** separately frozen exposure-focused feature subset; exact
    membership must be specified before training.
-   **OOF:** out-of-fold predictions generated only from training folds.
-   **Blind Test:** held-out test set whose labels remain inaccessible
    until predictions and artifacts are frozen and hashed.
