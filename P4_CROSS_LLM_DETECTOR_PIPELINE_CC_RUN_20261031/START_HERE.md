# P4 Cross-LLM Detector Pipeline v1.0

## What this package does
Implements the P4 supervised experiment matrix with:
- V77 and V69 feature matrices
- V_all and V_exp
- LR, RF, Gradient Boosting
- Single-source full 3x3 transfer matrix
- Pooled leave-one-LLM-out
- Train-only GroupKFold randomized hyperparameter search
- Validation-only threshold selection (maximum balanced accuracy)
- Validation metrics and permutation importance
- Train-only correlation and KS diagnostics
- Test score generation and SHA-256 lock, without opening test labels
- Separate post-lock evaluation command for the externally stored sealed labels

## Important design choices
1. `V_exp` is operationally defined in `config/V_exp_manifest.json` as the six frozen P3 `exposure_structure` primary features. It is a transparent exposure-structure subset, not proof that an agent attended to or complied with injected content.
2. Threshold selection is Validation maximum balanced accuracy because no numerical FPR ceiling was specified. This choice is frozen in this version; changing it requires a new protocol/config version before test labels are opened.
3. Hyperparameter tuning uses RandomizedSearchCV with GroupKFold, Train only, Average Precision scoring.
4. No model identity, split, suite, IDs, group IDs, labels, attacker goals, evaluator outcomes, or review metadata enter predictors.
5. The sealed test-label file is intentionally NOT included. Test labels must remain outside this package until after the score lock.
6. This package does not redefine the project-specific Algorithm 5/6 labels. It implements the approved LR/RF/GB experimental workflow; attach the exact Algorithm 5/6 labels to the project documentation if they refer to separate orchestration components.

## Setup (macOS / Linux)
From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/p4.py all-dev
```

The `all-dev` command fits 72 unique detector configurations (each single-source detector is then evaluated on three targets, producing 108 single-source cells; pooled runs produce 36 pooled cells; 144 cells total), saves validation outputs, then predicts all test rows and creates the lock manifest. It does NOT read sealed labels.

For a dry development run only:
```bash
python scripts/p4.py train
```

After `all-dev` completes, inspect:
- `outputs/development/validation_metrics_all.csv`
- `outputs/development/hyperparameter_search_summary.csv`
- `outputs/locked_test_scores/test_scores_locked.csv`
- `outputs/locked_test_scores/lock_manifest.json`

Do not edit locked files after lock. Keep a backup copy.

## After the lock
Only after confirming the lock manifest and following the project’s authorized unsealing procedure, pass the externally stored sealed-label CSV:

```bash
python scripts/p4.py evaluate --labels "/absolute/path/to/SEALED_TEST_LABELS.csv" --out outputs/final_evaluation
```

The sealed-label file must be a CSV with `trajectory_id,ipi_affected`. Do not move it into this package before score lock.

## Expected outputs
- 144 validation/model experiment records
- 72 fitted model artifacts
- validation predictions, tuning summary, permutation importance
- V77/V69 Train-only correlation and KS diagnostics
- locked score file with per-source-target and pooled predictions
- lock manifest with score and model hashes
- post-unseal per-cell metrics and group-bootstrap intervals

## Resource note
The default run uses 12 randomized trials per experiment, 3 group folds, 144 experiments. On a laptop it may take a while. Reduce `random_search_iterations` only before the run and record the change. The pipeline is deterministic by fixed seed, subject to library/platform differences.

## Failure behavior
The pipeline stops if trajectory IDs, labels, shared feature values, or frozen group disjointness checks fail. Do not bypass integrity errors by modifying the source data.
