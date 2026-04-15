# ML7501 Real Estate Project

End-to-end applied machine learning project for predicting Dubai real-estate transaction value from raw Dubai Land Department data, enriched with rental-market context and annual hotel-sector indicators.

## Project Goal

This project was built for `ML 7501 - Applied Machine Learning`. The core supervised task is regression on `actual_worth`, using a realistic raw-data workflow instead of a toy benchmark. The pipeline direction is:

1. ingest and harmonize multiple public datasets
2. perform rigorous exploratory data analysis
3. engineer leak-free market-context features
4. compare multiple machine learning models
5. evaluate performance and explain the strongest results

## Repository Status

The repository now includes the full source pipeline:

- [src/eda.py](src/eda.py): exploratory data analysis
- [src/modeling.py](src/modeling.py): master-table construction, preprocessing, model training, tuning, artifact export
- [src/evaluate_artifacts.py](src/evaluate_artifacts.py): rigorous post-training evaluation from saved artifacts
- [reports/project_status.md](reports/project_status.md): concise project summary
- [data/README.md](data/README.md): local raw-data notes

## Data Sources

The project uses three public-source datasets stored locally under `data/raw/`:

- Dubai Land Department transaction records
- Dubai rental contracts
- UAE hotel statistics used as annual macro context

The transaction table is the supervised master table. The rent table is used for area-time enrichment. The hotel table is only suitable as annual macro context, not neighborhood-level joining.

## Exploratory Data Analysis

The EDA established the modeling direction and surfaced the main data risks:

- `actual_worth` is strongly right-skewed and contains major outliers
- rental/project fields are useful but sparse
- `meter_sale_price` is a leakage risk and should not be used directly for prediction
- `area_id` and time-based aggregation are more reliable join anchors than raw project names

### Target Distribution

The sale-price target has a heavy right tail, which supports testing log-transformed target variants and robust error analysis later in the pipeline.

![Actual worth distribution](docs/figures/transactions_actual_worth_distribution.png)

### Join Feasibility

Transactions and rent records have strong overlap at `area_id` and `area_name_en`, but much weaker overlap at project-name level. That directly informs the feature-engineering strategy.

![Transactions vs rent join overlap](docs/figures/transactions_rent_join_overlap.png)

## Model Performance Snapshot

These metrics come from the strongest local experiment artifacts produced by the tracked source pipeline.

### Best Regression Result

- Model: `HistGradientBoostingRegressor`
- Test RMSE: `8,965,434.46`
- Test MAE: `1,394,204.91`
- Test R²: `0.5513`

Regression comparison:

![Regression model comparison](docs/figures/regression_model_comparison_rmse.png)

### Best Classification Result

- Derived label: `is_high_value = 1` when `actual_worth >= 2,400,000`
- Model: `HistGradientBoostingClassifier`
- Test Accuracy: `0.8867`
- Test F1: `0.8323`
- Test ROC AUC: `0.9557`

Classification comparison:

![Classification model comparison](docs/figures/classification_model_comparison_roc_auc.png)

### Error Concentration

The strongest regression model performs much worse on the highest-value band than on typical transactions. This matters for the final report because average metrics alone understate the difficulty of rare luxury deals.

![Regression MAE by target value band](docs/figures/regression_best_mae_by_value_band.png)

## Conclusions So Far

1. Tree-based gradient boosting is the strongest model family for both regression and classification in the current experiments.
2. The problem is non-linear and interaction-heavy; size alone is not enough to explain property value.
3. Location and local market context add real predictive value, but they must be engineered carefully to avoid leakage.
4. Model quality is materially better on typical transactions than on the most expensive segment.
5. The EDA and early experiments support the original project hypothesis that non-linear ensemble methods should outperform simple linear baselines.

## Repository Layout

```text
ML7501-Real-Estate-Proj/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── raw/                  # local raw files, not versioned
├── docs/
│   ├── README.md
│   ├── course/               # local course brief reference
│   ├── proposal/             # local proposal reference
│   └── figures/              # tracked summary figures used in the README
├── reports/
│   └── project_status.md
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── modeling.py
│   └── evaluate_artifacts.py
└── outputs/                  # local generated artifacts, not versioned
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run The Current EDA

```bash
python3 -m src.eda
```

This generates summary tables and plots under `outputs/eda/`.

## Run The End-to-End Modeling Pipeline

Train the full regression and classification pipeline and save artifacts:

```bash
python3 -m src.modeling --task both --output-dir outputs/modeling/latest
```

Optional GPU-backed run when `xgboost` is installed:

```bash
python3 -m src.modeling --task both --use-gpu --output-dir outputs/modeling/gpu_run
```

Important options:

- `--train-frac 0.70`
- `--val-frac 0.15`
- `--classification-quantile 0.75`
- `--tune-iterations 8`
- `--cv-splits 4`
- `--n-jobs 1`

## Run Artifact Evaluation

Evaluate a saved artifact directory and generate enriched metrics, subgroup analysis, and summary plots:

```bash
python3 -m src.evaluate_artifacts \
  --artifact-dir outputs/modeling/gpu_run \
  --output-dir outputs/evaluation/gpu_run \
  --bootstrap-iterations 250
```

## Expected Outputs

After running the full pipeline, the main local outputs are:

- `outputs/modeling/<run_name>/tables/`
- `outputs/modeling/<run_name>/plots/`
- `outputs/modeling/<run_name>/models/`
- `outputs/modeling/<run_name>/summaries/`
- `outputs/evaluation/<run_name>/tables/`
- `outputs/evaluation/<run_name>/plots/`
- `outputs/evaluation/<run_name>/summaries/`

## Notes

- Raw data are intentionally excluded from git.
- Large generated experiment artifacts remain local under `outputs/`.
- The repository is structured so the full end-to-end source code is tracked, while heavyweight local outputs stay untracked.
