# Appendix: Modeling Detail

This tracked note summarizes the strengthened evaluation protocol now implemented in the repository. Exact regenerated tables should come from `python3 -m src.appendix_analysis` and `python3 -m src.evaluate_artifacts` against the desired artifact directory.

## What Changed

- The repository no longer has to rely on a single train/validation/test narrative for the final report.
- `src.appendix_analysis.py` now runs expanding-window backtests for the saved best regression and classification model families.
- `src.modeling.py` now emits merged validation-vs-test comparison tables so the anomalous validation window is visible in the model-comparison artifacts themselves.
- The appendix pipeline now evaluates structural-only, location-only, rental-enriched, and full-feature regression variants across the same rolling folds.
- Ablation gains are accompanied by paired significance-style checks so the report can say whether observed improvements look stable or just period-specific.
- `src.evaluate_artifacts.py` now adds quantile-based prediction intervals and luxury-segment calibration outputs for the heavy-tailed regression task.
- `src.transformers.py` now holds the canonical `QuantileClipper`, removing the previous duplicate-definition risk across modules.
- `src.modeling.py` now also produces a regression comparison scatter grid plus a classification precision-recall comparison and threshold sweep, while applying balanced class weighting consistently across the main non-dummy classifiers where the estimator supports it.

## Recommended Report Framing

1. Use the rolling-origin summaries as the primary evidence for temporal robustness.
2. Use the validation-vs-test tables and EDA split-distribution plot to explain why the validation window behaved differently from the held-out test period.
3. Use the ablation tables to discuss whether rental and hotel context add stable value beyond structural and location baselines, while making clear that the ablation rows are constrained diagnostic feature-regime variants rather than replacements for the main full-pipeline leaderboard.
4. Use the interval coverage and luxury calibration outputs to explain why point estimates alone are incomplete for the top-value segment.

## Generated Artifacts

Running the appendix pipeline now produces tables such as:

- `rolling_origin_backtest_plan.csv`
- `regression_rolling_origin_backtest_summary.csv`
- `classification_rolling_origin_backtest_summary.csv`
- `regression_ablation_table.csv`
- `regression_ablation_significance.csv`
- `raw_vs_log_target_comparison.csv`

Running the evaluation pipeline now additionally produces:

- `regression_best_prediction_intervals.csv`
- `regression_best_value_band_summary.csv`
- `regression_best_luxury_calibration.csv`

Running the modeling pipeline now additionally produces:

- `regression_model_comparison_validation_vs_test.csv`
- `classification_model_comparison_validation_vs_test.csv`
- `classification_threshold_sweep.csv`
- `regression_model_comparison_actual_vs_predicted.png`
- `classification_precision_recall_comparison.png`

These outputs are designed so the final submission can replace unsupported single-split claims with repeated temporal evidence and uncertainty-aware regression analysis.
