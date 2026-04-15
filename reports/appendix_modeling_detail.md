# Appendix: Modeling Detail

This appendix captures three high-value details that strengthen the final submission: exact hyperparameter search spaces, a feature-family ablation table, and a raw-target versus log-target comparison.

## Exact Hyperparameter Search Spaces

Only `random_forest` and `hist_gradient_boosting` are tuned in the default tracked run. The source code also contains optional `xgboost` search spaces, but those were not used in the default saved run summarized in the repository.

### Regression Search Spaces
| model_name             | tuned_in_default_run   | search_space                                                                                                                                                                                              |
|:-----------------------|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dummy_regressor        | False                  | {}                                                                                                                                                                                                        |
| ridge_regression       | False                  | {"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]}                                                                                                                                                                     |
| svm_regression         | False                  | {"C": [0.1, 0.5, 1.0, 2.0], "epsilon": [0.01, 0.05, 0.1, 0.2]}                                                                                                                                            |
| random_forest          | True                   | {"max_depth": [null, 10, 20, 30], "max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 2, 5], "min_samples_split": [2, 5, 10], "n_estimators": [250, 400, 550]}                                    |
| hist_gradient_boosting | True                   | {"l2_regularization": [0.0, 0.1, 1.0], "learning_rate": [0.03, 0.05, 0.08], "max_depth": [null, 6, 8, 10], "max_iter": [250, 350, 500], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [10, 20, 40]} |
| xgboost                | False                  | {"colsample_bytree": [0.7, 0.9, 1.0], "learning_rate": [0.03, 0.05, 0.08], "max_depth": [4, 6, 8], "n_estimators": [250, 400, 550], "reg_lambda": [0.5, 1.0, 2.0], "subsample": [0.8, 0.9, 1.0]}          |

### Classification Search Spaces
| model_name             | tuned_in_default_run   | search_space                                                                                                                                                                                              |
|:-----------------------|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dummy_classifier       | False                  | {}                                                                                                                                                                                                        |
| logistic_regression    | False                  | {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}                                                                                                                                                                          |
| svm_classifier         | False                  | {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}                                                                                                                                                                          |
| random_forest          | True                   | {"max_depth": [null, 10, 20, 30], "max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 2, 5], "min_samples_split": [2, 5, 10], "n_estimators": [250, 400, 550]}                                    |
| hist_gradient_boosting | True                   | {"l2_regularization": [0.0, 0.1, 1.0], "learning_rate": [0.03, 0.05, 0.08], "max_depth": [null, 6, 8, 10], "max_iter": [250, 350, 500], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [10, 20, 40]} |
| xgboost                | False                  | {"colsample_bytree": [0.7, 0.9, 1.0], "learning_rate": [0.03, 0.05, 0.08], "max_depth": [4, 6, 8], "n_estimators": [250, 400, 550], "reg_lambda": [0.5, 1.0, 2.0], "subsample": [0.8, 0.9, 1.0]}          |

## Regression Ablation Table

Ablation uses the best tuned `HistGradientBoostingRegressor` configuration and evaluates the impact of rent and hotel feature families on the held-out test split.

| variant                  |   feature_count |                 mse |         rmse |          mae |     r2 |
|:-------------------------|----------------:|--------------------:|-------------:|-------------:|-------:|
| structural_location_only |              24 | 51901694974569.4375 | 7204283.0993 | 1311321.2812 | 0.7102 |
| structural_plus_hotel    |              43 | 69409980080549.7891 | 8331265.2149 | 1369619.6290 | 0.6125 |
| structural_plus_rent     |              36 | 78031607807562.6094 | 8833550.1248 | 1397195.5521 | 0.5644 |
| full_feature_set         |              55 | 80379015005761.2656 | 8965434.4572 | 1394204.9084 | 0.5513 |

Interpretation:

- The strongest ablation result is `structural_location_only`, which suggests the current coarse rent/hotel features add noise faster than they add signal.
- This does not mean auxiliary data are useless; it means the current engineered versions likely need refinement or stronger regularization.
- That is a stronger report narrative than simply claiming that “more features helped,” because it shows critical evaluation of feature quality.

## Raw-Target vs Log-Target Comparison

The comparison below keeps the feature set and tuned estimator fixed and changes only the target treatment.

| target_version   |                    mse |           rmse |           mae |        r2 |
|:-----------------|-----------------------:|---------------:|--------------:|----------:|
| log1p_target     |    80379015005761.2656 |   8965434.4572 |  1394204.9084 |    0.5513 |
| raw_target       | 22349790049594972.0000 | 149498461.6964 | 16966333.5761 | -123.7751 |

Interpretation:

- `log1p_target` is decisively better than the raw-target version on the same split and estimator.
- This is consistent with the EDA finding that `actual_worth` is extremely right-skewed.
- Including this comparison in the report makes the target-transformation decision explicit and evidence-based.
