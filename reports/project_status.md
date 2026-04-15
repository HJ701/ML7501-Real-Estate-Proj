# Project Status Snapshot

## Scope Completed

- Project brief reviewed and translated into a rubric-aligned plan
- Proposal reviewed to confirm the target task and data-join strategy
- Exploratory data analysis implemented in `src/eda.py`
- End-to-end modeling pipeline implemented in `src/modeling.py`
- Artifact-based evaluation pipeline implemented in `src/evaluate_artifacts.py`
- Appendix analysis implemented in `src/appendix_analysis.py`
- Reproducibility tooling implemented with `src/validate_data.py`, `data/dataset_manifest.json`, and `requirements-lock.txt`
- Tracked sample data and schema snapshots added under `data/sample/`, `data/sample_manifest.json`, and `data/schemas/`
- EDA summaries, tables, and plots generated locally under `outputs/eda/`

## Main EDA Findings

1. The transactions dataset is the correct master table because it contains the target `actual_worth` and broad feature coverage.
2. The target is strongly right-skewed and contains large outliers, so later modeling should compare raw and log-target variants.
3. Rental enrichment should be built mainly on `area_id`, `area_name_en`, and time windows, not raw project-name joins alone.
4. Hotel statistics are suitable only as annual macro context.
5. `meter_sale_price` is a strong leakage risk and should be excluded from predictive modeling.

## Best Local Experiment Results So Far

These are local artifact results already produced before the repository cleanup.

Regression:

- Best model: `HistGradientBoostingRegressor`
- Test RMSE: `8,965,434.46`
- Test MAE: `1,394,204.91`
- Test R²: `0.5513`

Classification:

- Best model: `HistGradientBoostingClassifier`
- Test Accuracy: `0.8867`
- Test F1: `0.8323`
- Test ROC AUC: `0.9557`

Evaluation notes:

- The hardest regression segment is the highest-value transaction band.
- Classification remains strong overall, but performance varies by year and property type.
- The current results support the proposal hypothesis that non-linear models fit this problem better than linear baselines.

## Next Engineering Step

Extend the tracked pipeline with final report assets, presentation material, and any final feature-engineering refinements needed before submission.
