# IPI Cross-LLM Research Archive

Private research archive for:

**Cross-LLM Generalization of Behavioral Detection for Indirect Prompt Injection in LLM Agents**

## Research Scope

This repository preserves the P4 detector development, experimental
configuration, source data, model artifacts, locked test predictions,
post-unseal evaluation outputs, and provenance records.

P1-P3 contributions from other team members will be added separately.
The current archive does not claim to contain their complete original
working files.

## P4 Archive

Main directory:

`P4_CROSS_LLM_DETECTOR_PIPELINE_CC_RUN_20261031/`

It contains:

- `config/` — experiment configurations
- `data/` — feature matrices, metadata, and training/validation labels
- `docs/` — experimental protocol
- `outputs/` — trained models, development results, locked scores,
  and final test evaluation
- `provenance/` — environment records, manifests, and provenance addenda
- `scripts/` — execution and audit scripts
- `src/` — detector pipeline implementation
- `tests/` — project tests
- `SHA256_MANIFEST.csv` — file-level SHA-256 verification manifest

## Experimental Overview

- Models: LLM-A, LLM-B, LLM-C
- Feature versions: V77 and V69
- Feature regimes: V_all and V_exp
- Algorithms: Logistic Regression, Random Forest, Gradient Boosting
- Evaluation regimes: single-source and pooled leave-one-LLM-out
- Random seed: 20261031
- Hyperparameter search: 12 randomized trials
- Group cross-validation: 3 folds
- Bootstrap replicates: 2,000

The experiment configuration defines 144 evaluation cells.
The pipeline trains 72 development models.

## Test Evaluation

The development models and locked test predictions were generated
before the test labels were unsealed.

The test labels were subsequently unsealed and evaluated using the
frozen predictions.

Relevant outputs are preserved under:

- `outputs/locked_test_scores/`
- `outputs/after_unseal_inputs/`
- `outputs/final_test_evaluation/`

Do not modify frozen labels, predictions, thresholds, or locked scores.

## Reproducibility

See:

- `docs/P4_MASTER_EXPERIMENTAL_PROTOCOL_v1.0.md`
- `START_HERE.md`
- `requirements.txt`
- `config/experiment_config.json`
- `provenance/`

The original working environment used Python 3.14.3 on macOS arm64.

The local `.venv` directory is intentionally excluded from version control.
Recreate the environment using the dependency specification and instructions
in the P4 package.

## Important Limitations

- Validation thresholds were selected using validation data; validation
  metrics are therefore tuned and potentially optimistic.
- No FPR cap was configured.
- Bootstrap resampling uses user_task_group_id with condition_group_id
  fallback, rather than the connected components used for group CV.
- SHAP analysis, paired statistical tests, effect sizes, and Holm correction
  are not included in the current pipeline outputs.
- The P1 controlled extract is not established as byte-identical to the
  signed/frozen original; the documented provenance caveat remains.
- Static source inspection cannot prove complete runtime isolation from
  test labels.
- V_exp is an operational exposure-structure feature subset and should not
  be interpreted as direct proof of model attention or causal compliance.

These limitations must be considered when interpreting results.

## Team Contributions

P1-P3 source materials, reports, and evidence will be added by their
respective contributors in a future update.

## Archive Integrity

The P4 source copy was compared against the original working directory
using SHA-256 hashes.

At the time of initial archive verification:

- Original files: 181
- Archived files: 181
- Missing files: 0
- Extra files: 0
- Hash mismatches: 0

The generated SHA256 manifest is retained inside the P4 archive.

## Access and Usage

This repository is private and intended for research collaboration,
reproducibility, and preservation of project evidence.

Do not redistribute datasets, benchmark materials, or third-party
artifacts unless their applicable licenses and permissions allow it.
