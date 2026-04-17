# Appendix: Modeling Detail

This tracked note summarizes the strengthened evaluation protocol now implemented in the repository. Exact regenerated tables should come from `python3 -m src.appendix_analysis` and `python3 -m src.evaluate_artifacts` against the desired artifact directory.

## What Changed

- The repository no longer has to rely on a single train/validation/test narrative for the final report.
- `src.appendix_analysis.py` now runs expanding-window backtests for the saved best regression and classification model families.
- The appendix pipeline now evaluates structural-only, location-only, rental-enriched, and full-feature regression variants across the same rolling folds.
- Ablation gains are accompanied by paired significance-style checks so the report can say whether observed improvements look stable or just period-specific.
- `src.evaluate_artifacts.py` now adds quantile-based prediction intervals and luxury-segment calibration outputs for the heavy-tailed regression task.

## Recommended Report Framing

1. Use the rolling-origin summaries as the primary evidence for temporal robustness.
2. Use the ablation tables to discuss whether rental and hotel context add stable value beyond structural and location baselines.
3. Use the interval coverage and luxury calibration outputs to explain why point estimates alone are incomplete for the top-value segment.

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

These outputs are designed so the final submission can replace unsupported single-split claims with repeated temporal evidence and uncertainty-aware regression analysis.
