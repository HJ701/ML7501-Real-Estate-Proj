from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.modeling import (
    BASE_DIR,
    RANDOM_STATE,
    build_master_table,
    build_pipeline,
    build_preprocessor,
    evaluate_regression,
    get_classification_model_specs,
    get_regression_model_specs,
    select_model_features,
    temporal_split,
)


DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "reporting" / "appendix"
DEFAULT_MODELING_ARTIFACT_DIR = BASE_DIR / "outputs" / "modeling" / "gpu_run"
DEFAULT_DATA_DIR = BASE_DIR / "data" / "raw"


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    tables = output_dir / "tables"
    summaries = output_dir / "summaries"
    for directory in (output_dir, tables, summaries):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "tables": tables, "summaries": summaries}


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def exact_search_space_table(artifact_dir: Path, use_gpu: bool, n_jobs: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tuned_by_task: dict[str, set[str]] = {}
    for task in ("regression", "classification"):
        tuning_path = artifact_dir / "tables" / f"{task}_tuning_results.csv"
        if tuning_path.exists():
            tuned_by_task[task] = set(pd.read_csv(tuning_path)["model_name"].tolist())
        else:
            tuned_by_task[task] = set()

    for task, specs in [
        ("regression", get_regression_model_specs(use_gpu=use_gpu, random_state=RANDOM_STATE, n_jobs=n_jobs)),
        ("classification", get_classification_model_specs(use_gpu=use_gpu, random_state=RANDOM_STATE, n_jobs=n_jobs)),
    ]:
        for spec in specs:
            rows.append(
                {
                    "task": task,
                    "model_name": spec.name,
                    "tuned_in_default_run": spec.name in tuned_by_task[task],
                    "search_space": json.dumps(spec.param_distributions, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def parse_best_regression_params(artifact_dir: Path) -> dict[str, object]:
    tuning_df = pd.read_csv(artifact_dir / "tables" / "regression_tuning_results.csv")
    row = tuning_df.loc[tuning_df["model_name"] == "hist_gradient_boosting"].iloc[0]
    params = ast.literal_eval(row["best_params"])
    cleaned = {}
    prefix = "regressor__model__"
    for key, value in params.items():
        cleaned[key[len(prefix) :]] = value
    return cleaned


def fit_regression_variant(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    estimator: HistGradientBoostingRegressor,
    use_log_target: bool,
) -> dict[str, float]:
    preprocessor = build_preprocessor(X_dev)
    if use_log_target:
        model = build_pipeline(task="regression", preprocessor=preprocessor, estimator=clone(estimator))
    else:
        model = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
    model.fit(X_dev, y_dev)
    preds = model.predict(X_test)
    return evaluate_regression(y_test, preds)


def regression_ablation_table(
    artifact_dir: Path,
    data_dir: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = build_master_table(data_dir=data_dir)
    train_df, val_df, test_df = temporal_split(master, train_frac=0.70, val_frac=0.15)

    X_train = select_model_features(train_df)
    X_val = select_model_features(val_df)
    X_test_full = select_model_features(test_df)
    X_dev_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_dev = pd.concat([train_df["actual_worth"], val_df["actual_worth"]], axis=0).reset_index(drop=True)
    y_test = test_df["actual_worth"].reset_index(drop=True)

    rent_cols = [column for column in X_dev_full.columns if column.startswith("rent_")]
    hotel_cols = [column for column in X_dev_full.columns if column.startswith("hotel_")]

    variant_columns = {
        "structural_location_only": [col for col in X_dev_full.columns if col not in set(rent_cols + hotel_cols)],
        "structural_plus_rent": [col for col in X_dev_full.columns if col not in set(hotel_cols)],
        "structural_plus_hotel": [col for col in X_dev_full.columns if col not in set(rent_cols)],
        "full_feature_set": list(X_dev_full.columns),
    }

    best_params = parse_best_regression_params(artifact_dir)
    estimator = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **best_params)

    rows: list[dict[str, object]] = []
    for variant_name, cols in variant_columns.items():
        metrics = fit_regression_variant(
            X_dev=X_dev_full[cols],
            y_dev=y_dev,
            X_test=X_test_full[cols],
            y_test=y_test,
            estimator=estimator,
            use_log_target=True,
        )
        rows.append(
            {
                "variant": variant_name,
                "feature_count": len(cols),
                **metrics,
            }
        )

    ablation_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    target_compare_rows = []
    for use_log_target, label in [(False, "raw_target"), (True, "log1p_target")]:
        metrics = fit_regression_variant(
            X_dev=X_dev_full,
            y_dev=y_dev,
            X_test=X_test_full,
            y_test=y_test,
            estimator=estimator,
            use_log_target=use_log_target,
        )
        target_compare_rows.append({"target_version": label, **metrics})

    target_compare_df = pd.DataFrame(target_compare_rows).sort_values("rmse").reset_index(drop=True)
    return ablation_df, target_compare_df


def build_appendix_markdown(
    output_path: Path,
    search_space_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    target_compare_df: pd.DataFrame,
) -> None:
    regression_spaces = search_space_df.loc[search_space_df["task"] == "regression"].copy()
    classification_spaces = search_space_df.loc[search_space_df["task"] == "classification"].copy()

    best_ablation = ablation_df.iloc[0]
    best_target = target_compare_df.iloc[0]

    lines = [
        "# Appendix: Modeling Detail",
        "",
        "This appendix captures three high-value details that strengthen the final submission: exact hyperparameter search spaces, a feature-family ablation table, and a raw-target versus log-target comparison.",
        "",
        "## Exact Hyperparameter Search Spaces",
        "",
        "Only `random_forest` and `hist_gradient_boosting` are tuned in the default tracked run. The full model-spec search spaces are below.",
        "",
        "### Regression Search Spaces",
        regression_spaces[["model_name", "tuned_in_default_run", "search_space"]].to_markdown(index=False),
        "",
        "### Classification Search Spaces",
        classification_spaces[["model_name", "tuned_in_default_run", "search_space"]].to_markdown(index=False),
        "",
        "## Regression Ablation Table",
        "",
        "Ablation uses the best tuned `HistGradientBoostingRegressor` configuration and evaluates the impact of rent and hotel feature families on the held-out test split.",
        "",
        ablation_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Best ablation result: `{best_ablation['variant']}` with RMSE `{best_ablation['rmse']:,.2f}`, MAE `{best_ablation['mae']:,.2f}`, and R² `{best_ablation['r2']:.4f}`.",
        "",
        "## Raw-Target vs Log-Target Comparison",
        "",
        "The comparison below keeps the feature set and tuned estimator fixed and changes only the target treatment.",
        "",
        target_compare_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Best target treatment: `{best_target['target_version']}` with RMSE `{best_target['rmse']:,.2f}`, MAE `{best_target['mae']:,.2f}`, and R² `{best_target['r2']:.4f}`.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate appendix tables for the ML7501 project.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_MODELING_ARTIFACT_DIR,
        help="Modeling artifact directory used to recover tuned parameters.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the input data files expected by the modeling pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where appendix tables and summaries will be saved.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for any estimator search-space definitions. Defaults to 1.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Include XGBoost search spaces when xgboost is available and GPU should be considered.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = ensure_output_dirs(args.output_dir.resolve())

    search_space_df = exact_search_space_table(
        artifact_dir=args.artifact_dir.resolve(),
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs,
    )
    ablation_df, target_compare_df = regression_ablation_table(
        artifact_dir=args.artifact_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        n_jobs=args.n_jobs,
    )

    save_dataframe(search_space_df, directories["tables"] / "exact_hyperparameter_search_spaces.csv")
    save_dataframe(ablation_df, directories["tables"] / "regression_ablation_table.csv")
    save_dataframe(target_compare_df, directories["tables"] / "raw_vs_log_target_comparison.csv")
    build_appendix_markdown(
        output_path=directories["summaries"] / "appendix_modeling_detail.md",
        search_space_df=search_space_df,
        ablation_df=ablation_df,
        target_compare_df=target_compare_df,
    )

    print(f"Saved appendix artifacts to {directories['root']}")


if __name__ == "__main__":
    main()
