# Final portfolio audit

Scope: published commit `f1deeeb518d88ee28aa99206c05a0b82a85d21da`, plus the uncommitted audit fixes in this workspace. The remote `main` still points to that commit. No commit, push or remote workflow dispatch was performed during this audit.

## Assessment

Suitable for an educational ML/Data Science internship portfolio. No unresolved critical ML correctness defect was found in the standard dataset workflow. Publish the reviewed audit fixes before presenting the finalized version. Windows support remains provisional until the new Windows jobs pass. This is a fictional-data benchmark, not a production churn system or evidence of improved retention revenue.

## Repository and published CI

Confirmed the source dataset, pinned requirements, README, executed notebook, shared preprocessing module, training/prediction CLIs, tests, example input, metrics JSON, validation history, workflow and nine report figures. Generated joblib artifacts and prediction CSVs are correctly ignored. The dataset SHA-256 is unchanged and matches `model_metrics.json`.

Verified through the GitHub Actions API:

- [Published run 38747462360](https://github.com/zhandarkayirbay/Customer-Churn-Prediction/actions/runs/38747462360), for the published commit above: **completed / success**.
- Ubuntu / Python 3.11: installation, dependency check, original 8 tests, training, scoring and EDA all passed.
- Ubuntu / Python 3.12: the same steps all passed.
- The published workflow did not test Windows. Local audit changes add `windows-latest` for Python 3.11 and 3.12 and replace the Linux-specific notebook output directory with a relative directory. Those new CI jobs have not run.

`reports/validation.md` is historical evidence from the initial implementation; its eight-test count and pre-publication Git status should not be interpreted as the current audit state.

## Confirmed issues and local fixes

| Severity | Finding | Fix and verification |
|---|---|---|
| Medium | Numeric-looking customer IDs could lose leading zeros during CSV inference or be conflated in training validation. | Read IDs as strings in both CLIs/data loading; a regression test preserves `001`, `1`, and `0002` distinctly. |
| Medium | A custom training model/report path could overwrite the input dataset or collide with another output. | Validate all planned output paths before fitting/writing; regression tests verify the dataset bytes remain unchanged and colliding outputs are rejected. |
| Low | A trusted joblib file containing a non-model object caused an AttributeError traceback. | Validate that the loaded object exposes `predict_proba`; a real CLI test verifies exit code 2, a concise message, and no output creation. This does not make untrusted pickle loading safe. |
| Documentation | README described CI as not yet run and contained a user-specific installation directory. | Link the successful published run, distinguish pending Windows checks, and use repository-relative instructions. Clarify that example rows demonstrate inference, not independent evaluation. |
| Presentation | Some plots used similar colors for both churn classes, unlabeled group sizes, or two-decimal AUCs that obscured the small difference. | Distinct class colors, annotated percentages and group counts, readable axis labels, symmetric correlation scale, four-decimal AUC legend. All nine figures were visually reviewed. |
| Notebook | Redundant headings, a separate relative data path, and stale execution-version metadata. | Keep every analysis/code cell, remove redundant headings, load from the resolved repository root, and save actual execution outputs and Python version. |

## ML and evidence review

- Each row is a distinct customer ID in the source dataset. There are no exact duplicate rows or duplicate IDs. Different customers can share identical measured feature profiles; that is not evidence that the same customer leaked across the split.
- `customerID` and `Churn` are excluded from features. Numeric conversion is deterministic; missing-value statistics, scaling and category vocabularies are learned inside each pipeline fit and CV fold.
- The stratified 80/20 split uses seed 42. Three-fold shuffled stratified CV on the 5,634 training rows selects the candidate before computing the 1,409-row holdout results. No oversampling, tuning or threshold search uses holdout labels.
- Class 1 means churn. Precision, recall, F1 and accuracy agree with both confusion matrices and classification reports. Selected-model AUC was independently recomputed from ranked holdout scores, and the saved pipeline reproduces its confusion matrix.
- Current metrics JSON is unchanged from the published experiment. Both Python versions reproduced the recorded metrics within absolute tolerance 1e-12.
- README dataset counts, missing charges, group rates, medians, baseline, example predictions and model table are supported by executed calculations. The majority baseline is 73.46% accuracy and zero churn recall.
- The Logistic Regression advantage in CV AUC is small; it is the specified selection rule, not proof of statistically significant superiority. Random Forest has better threshold-specific F1, precision and accuracy.
- Full-sample EDA and previous inspection of holdout results mean this is a development benchmark, not a blinded external study. Future tuning requires a new untouched final evaluation dataset. No temporal timestamps/prediction horizon are available, so prospective leakage and future-customer performance cannot be established from this sample.
- Risk bands are illustrative and class-weighted probabilities are not established as calibrated. No causal, profit, fairness or production-readiness claim is supported or made.

## Actual rerun results

| Model | Mean training CV ROC-AUC | Holdout accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8457 | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |
| Random Forest | 0.8406 | 0.7764 | 0.5646 | 0.6898 | 0.6209 | 0.8375 |

Selected by training-only CV: **Logistic Regression**.

| Check | Python 3.11.17 / Linux | Python 3.12.14 / Linux |
|---|---|---|
| Expanded automated suite | 11 passed | 11 passed |
| Training both candidates, CV and holdout evaluation | Passed | Passed |
| Model save/reload prediction equivalence | Passed | Passed |
| Three-row unlabelled CSV example | Passed | Passed |
| EDA notebook | All 11 code cells passed | All 11 code cells passed |

The full 7,043-customer CSV was also scored and checked for matching IDs, finite probabilities within [0, 1], populated risk bands and priorities. Notebook charts were rendered inline and saved separately. README relative image targets and important project files exist. `git diff --check` passed.

Windows and Python 3.11.5 specifically were not executed. The implementation uses pathlib, ordinary Python entry points and pinned packages supporting Python 3.11, but this is supporting evidence rather than a Windows runtime test. Core dependencies are pinned; transitive Jupyter dependencies remain unlocked. The successful published CI installs dependencies afresh on Ubuntu.

## Files to review before approval

- `src/train.py`, `src/predict.py`, `tests/test_workflow.py`: safeguards and three new regression tests.
- `.github/workflows/validate.yml`: four OS/Python combinations and portable notebook output.
- `notebooks/01_eda.ipynb`: retained analyses, revised presentation, refreshed outputs/metadata.
- Six regenerated figures: contract, internet, payment, tenure/charges, feature relationships, ROC curves.
- `README.md` and this audit report: supported claims, current CI status, limitations.

No new framework or dependency was introduced. Source data, recorded model metrics and modeling hyperparameters remain unchanged.

## Remaining checklist

- [ ] Review and approve the local audit fixes, then commit and push them.
- [ ] Confirm all four jobs in the expanded GitHub Actions matrix pass after publication.
- [ ] Only then describe Windows compatibility as verified; update the pending Windows wording with its run link.
- [ ] In the CV/interview, describe a reproducible educational churn benchmark; use measured ROC-AUC/recall, not unsupported production or retention-impact claims.

Optional future work, not required for internship portfolio use: a full dependency lock, calibration and cost-aware thresholds on separate validation data, confidence intervals, an external/temporal test set, and fairness analysis before any real-world use.
