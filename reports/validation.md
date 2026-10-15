# Executed validation

The final portfolio workflow was executed on Linux with Python 3.12.14 and separately with Python 3.11.17. Python 3.11.5 and Windows were not available for direct execution.

- Dependency compatibility checks passed in both environments.
- `python -m unittest discover -s tests -v`: 8 tests passed in each environment.
- `python src/train.py`: training-only 3-fold CV for both classifiers, holdout evaluation, plotting, serialization and reload equivalence passed.
- Batch prediction with the selected pipeline: 7,043 rows, matching customer IDs, finite probabilities in [0, 1], populated risk bands and retention priorities.
- Unlabelled three-row example: exported successfully with the model from each environment.
- EDA: all 11 code cells executed without errors, with inline chart outputs and nine saved report figures (six EDA and three model evaluation figures).
- Original notebook cells were retained in order; original dataset SHA-256 remained unchanged.
- `git diff --check`: passed.

Both Python versions selected Logistic Regression. Rounded metrics agree between runs. The checked-in `model_metrics.json` records the Python 3.12 experiment and exact dependency versions.

The GitHub Actions workflow is configured but has not been executed on GitHub. No commit or push was performed. The README contains the intended Windows commands, but Windows itself was not tested.

Core versions are pinned in `requirements.txt`; transitive notebook dependencies are not fully locked. The dataset is fictional, and class-weighted scores are not calibrated production probabilities.

## Final Git status before handoff

```text
 M .gitignore
 M README.md
 M notebooks/01_eda.ipynb
 M requirements.txt
 M src/predict.py
 M src/train.py
?? .github/workflows/validate.yml
?? examples/new_customers.csv
?? reports/figures/churn_by_contract.png
?? reports/figures/churn_by_internetservice.png
?? reports/figures/churn_by_paymentmethod.png
?? reports/figures/confusion_matrix_logistic_regression.png
?? reports/figures/confusion_matrix_random_forest.png
?? reports/figures/feature_relationships.png
?? reports/figures/roc_curves.png
?? reports/figures/target_distribution.png
?? reports/figures/tenure_monthly_charges.png
?? reports/model_metrics.json
?? reports/validation.md
?? src/__init__.py
?? src/preprocessing.py
?? tests/test_workflow.py
```
