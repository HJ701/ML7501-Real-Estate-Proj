from __future__ import annotations

import argparse
import ast
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.modeling import (
    BASE_DIR,
    RANDOM_STATE,
    build_feature_variants,
    build_master_table,
    build_pipeline,
    build_preprocessor,
    describe_expanding_window_splits,
    evaluate_classification,
    evaluate_regression,
    expanding_window_splits,
    get_classification_model_specs,
    get_prediction_scores,
    get_regression_model_specs,
    select_model_features,
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


def load_best_model_name(artifact_dir: Path, task: str) -> str:
    metrics = pd.read_csv(artifact_dir / "tables" / f"{task}_test_metrics.csv")
    if task == "regression":
        return str(metrics.sort_values("rmse", ascending=True).iloc[0]["model_name"])
    return str(metrics.sort_values("roc_auc", ascending=False).iloc[0]["model_name"])


def parse_tuned_params(artifact_dir: Path, task: str, model_name: str) -> dict[str, object]:
    tuning_path = artifact_dir / "tables" / f"{task}_tuning_results.csv"
    if not tuning_path.exists():
        return {}

    tuning_df = pd.read_csv(tuning_path)
    matches = tuning_df.loc[tuning_df["model_name"] == model_name]
    if matches.empty:
        return {}

    params = ast.literal_eval(matches.iloc[0]["best_params"])
    prefix = "regressor__model__" if task == "regression" else "model__"
    cleaned = {}
    for key, value in params.items():
        if key.startswith(prefix):
            cleaned[key[len(prefix) :]] = value
    return cleaned


def resolve_estimator(artifact_dir: Path, task: str, model_name: str, use_gpu: bool, n_jobs: int):
    if task == "regression":
        specs = get_regression_model_specs(use_gpu=use_gpu, random_state=RANDOM_STATE, n_jobs=n_jobs)
    else:
        specs = get_classification_model_specs(use_gpu=use_gpu, random_state=RANDOM_STATE, n_jobs=n_jobs)

    spec_map = {spec.name: spec for spec in specs}
    if model_name not in spec_map:
        raise ValueError(f"Model '{model_name}' not found in {task} specs. Install optional dependencies if needed.")

    estimator = clone(spec_map[model_name].estimator)
    tuned_params = parse_tuned_params(artifact_dir=artifact_dir, task=task, model_name=model_name)
    if tuned_params:
        estimator.set_params(**tuned_params)
    return estimator


def summarize_backtest_metrics(fold_df: pd.DataFrame, task: str) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    if task == "regression":
        summary = (
            fold_df.groupby("model_name", observed=False)
            .agg(
                folds=("fold", "nunique"),
                mean_rmse=("rmse", "mean"),
                std_rmse=("rmse", "std"),
                min_rmse=("rmse", "min"),
                max_rmse=("rmse", "max"),
                mean_mae=("mae", "mean"),
                std_mae=("mae", "std"),
                mean_r2=("r2", "mean"),
                std_r2=("r2", "std"),
            )
            .reset_index()
            .sort_values("mean_rmse", ascending=True)
            .reset_index(drop=True)
        )
        summary["rmse_instability_ratio"] = summary["std_rmse"] / summary["mean_rmse"].replace(0, np.nan)
        return summary

    summary = (
        fold_df.groupby("model_name", observed=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
        )
        .reset_index()
        .sort_values("mean_roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    summary["roc_auc_instability_ratio"] = summary["std_roc_auc"] / summary["mean_roc_auc"].replace(0, np.nan)
    return summary


def bootstrap_mean_interval(values: np.ndarray, random_state: int, iterations: int = 2000) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    samples = rng.choice(clean, size=(iterations, len(clean)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_sign_flip_pvalue(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")

    observed = abs(clean.mean())
    patterns = np.array(list(product([-1.0, 1.0], repeat=len(clean))), dtype=float)
    randomized_means = np.abs((patterns * clean).mean(axis=1))
    exceed = np.count_nonzero(randomized_means >= observed)
    return float((exceed + 1) / (len(randomized_means) + 1))


def fit_regression_variant(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    estimator,
    use_log_target: bool,
) -> dict[str, float]:
    preprocessor = build_preprocessor(X_train)
    if use_log_target:
        model = build_pipeline(task="regression", preprocessor=preprocessor, estimator=clone(estimator))
    else:
        model = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_regression(y_test, preds)


def rolling_origin_backtest(
    artifact_dir: Path,
    data_dir: Path,
    task: str,
    model_name: str,
    classification_quantile: float,
    backtest_splits: int,
    use_gpu: bool,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = build_master_table(data_dir=data_dir)
    folds = expanding_window_splits(master, n_splits=backtest_splits)
    estimator = resolve_estimator(
        artifact_dir=artifact_dir,
        task=task,
        model_name=model_name,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
    )

    rows: list[dict[str, object]] = []
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = master.iloc[train_idx].reset_index(drop=True)
        test_df = master.iloc[test_idx].reset_index(drop=True)
        X_train = select_model_features(train_df)
        X_test = select_model_features(test_df)
        preprocessor = build_preprocessor(X_train)

        if task == "regression":
            y_train = train_df["actual_worth"].reset_index(drop=True)
            y_test = test_df["actual_worth"].reset_index(drop=True)
            model = build_pipeline(task="regression", preprocessor=preprocessor, estimator=clone(estimator))
            model.fit(X_train, y_train)
            metrics = evaluate_regression(y_test, model.predict(X_test))
        else:
            threshold = float(train_df["actual_worth"].quantile(classification_quantile))
            y_train = train_df["actual_worth"].ge(threshold).astype(int).reset_index(drop=True)
            y_test = test_df["actual_worth"].ge(threshold).astype(int).reset_index(drop=True)
            model = build_pipeline(task="classification", preprocessor=preprocessor, estimator=clone(estimator))
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            scores = get_prediction_scores(model, X_test)
            metrics = evaluate_classification(y_test, preds, scores)
            metrics["classification_threshold"] = threshold

        rows.append(
            {
                "fold": fold_number,
                "model_name": model_name,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_start": str(train_df["instance_date"].min().date()),
                "train_end": str(train_df["instance_date"].max().date()),
                "test_start": str(test_df["instance_date"].min().date()),
                "test_end": str(test_df["instance_date"].max().date()),
                **metrics,
            }
        )

    fold_df = pd.DataFrame(rows)
    plan_df = describe_expanding_window_splits(master, folds)
    return fold_df, summarize_backtest_metrics(fold_df, task=task), plan_df


def regression_ablation_backtest(
    artifact_dir: Path,
    data_dir: Path,
    backtest_splits: int,
    use_gpu: bool,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = build_master_table(data_dir=data_dir)
    folds = expanding_window_splits(master, n_splits=backtest_splits)
    best_model_name = load_best_model_name(artifact_dir, task="regression")
    estimator = resolve_estimator(
        artifact_dir=artifact_dir,
        task="regression",
        model_name=best_model_name,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
    )
    variants = build_feature_variants(select_model_features(master).columns)

    rows: list[dict[str, object]] = []
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = master.iloc[train_idx].reset_index(drop=True)
        test_df = master.iloc[test_idx].reset_index(drop=True)
        X_train_full = select_model_features(train_df)
        X_test_full = select_model_features(test_df)
        y_train = train_df["actual_worth"].reset_index(drop=True)
        y_test = test_df["actual_worth"].reset_index(drop=True)

        for variant_name, columns in variants.items():
            metrics = fit_regression_variant(
                X_train=X_train_full[columns],
                y_train=y_train,
                X_test=X_test_full[columns],
                y_test=y_test,
                estimator=estimator,
                use_log_target=True,
            )
            rows.append(
                {
                    "fold": fold_number,
                    "variant": variant_name,
                    "feature_count": len(columns),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(rows)
    summary_df = (
        fold_df.groupby("variant", observed=False)
        .agg(
            folds=("fold", "nunique"),
            feature_count=("feature_count", "first"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
        )
        .reset_index()
        .sort_values("mean_rmse", ascending=True)
        .reset_index(drop=True)
    )
    summary_df["rmse_instability_ratio"] = summary_df["std_rmse"] / summary_df["mean_rmse"].replace(0, np.nan)
    return fold_df, summary_df


def regression_ablation_significance(fold_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    comparisons = [
        ("location_only", "structural_only"),
        ("rental_enriched", "structural_only"),
        ("full_feature_set", "rental_enriched"),
        ("full_feature_set", "structural_only"),
    ]
    metric_directions = {"rmse": "lower", "mae": "lower", "r2": "higher"}

    rows: list[dict[str, object]] = []
    for candidate, baseline in comparisons:
        candidate_df = fold_df.loc[fold_df["variant"] == candidate].sort_values("fold")
        baseline_df = fold_df.loc[fold_df["variant"] == baseline].sort_values("fold")
        merged = candidate_df.merge(baseline_df, on="fold", suffixes=("_candidate", "_baseline"))

        for metric, direction in metric_directions.items():
            raw_delta = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
            improvement = -raw_delta if direction == "lower" else raw_delta
            ci_lower, ci_upper = bootstrap_mean_interval(improvement.to_numpy(), random_state=random_state)
            p_value = paired_sign_flip_pvalue(raw_delta.to_numpy())
            rows.append(
                {
                    "candidate_variant": candidate,
                    "baseline_variant": baseline,
                    "metric": metric,
                    "mean_improvement": float(improvement.mean()),
                    "improvement_ci_lower": ci_lower,
                    "improvement_ci_upper": ci_upper,
                    "paired_sign_flip_pvalue": p_value,
                    "meaningful_at_5pct": bool(ci_lower > 0 and p_value < 0.05),
                }
            )

    return pd.DataFrame(rows)


def regression_target_treatment_backtest(
    artifact_dir: Path,
    data_dir: Path,
    backtest_splits: int,
    use_gpu: bool,
    n_jobs: int,
) -> pd.DataFrame:
    master = build_master_table(data_dir=data_dir)
    folds = expanding_window_splits(master, n_splits=backtest_splits)
    best_model_name = load_best_model_name(artifact_dir, task="regression")
    estimator = resolve_estimator(
        artifact_dir=artifact_dir,
        task="regression",
        model_name=best_model_name,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
    )

    rows: list[dict[str, object]] = []
    for label, use_log_target in [("raw_target", False), ("log1p_target", True)]:
        for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
            train_df = master.iloc[train_idx].reset_index(drop=True)
            test_df = master.iloc[test_idx].reset_index(drop=True)
            X_train = select_model_features(train_df)
            X_test = select_model_features(test_df)
            y_train = train_df["actual_worth"].reset_index(drop=True)
            y_test = test_df["actual_worth"].reset_index(drop=True)
            metrics = fit_regression_variant(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                estimator=estimator,
                use_log_target=use_log_target,
            )
            rows.append({"target_version": label, "fold": fold_number, **metrics})

    fold_df = pd.DataFrame(rows)
    return (
        fold_df.groupby("target_version", observed=False)
        .agg(
            folds=("fold", "nunique"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
        )
        .reset_index()
        .sort_values("mean_rmse", ascending=True)
        .reset_index(drop=True)
    )


def build_appendix_markdown(
    output_path: Path,
    search_space_df: pd.DataFrame,
    rolling_plan_df: pd.DataFrame,
    regression_backtest_df: pd.DataFrame,
    classification_backtest_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    ablation_significance_df: pd.DataFrame,
    target_compare_df: pd.DataFrame,
) -> None:
    regression_spaces = search_space_df.loc[search_space_df["task"] == "regression"].copy()
    classification_spaces = search_space_df.loc[search_space_df["task"] == "classification"].copy()

    best_ablation = ablation_df.iloc[0]
    best_target = target_compare_df.iloc[0]
    best_regression_backtest = regression_backtest_df.iloc[0] if not regression_backtest_df.empty else None
    best_classification_backtest = classification_backtest_df.iloc[0] if not classification_backtest_df.empty else None

    lines = [
        "# Appendix: Modeling Detail",
        "",
        "This appendix replaces the single-split narrative with expanding-window backtests, feature-family ablations, and target-treatment robustness checks.",
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
        "## Rolling-Origin Backtest",
        "",
        f"The backtest uses `{len(rolling_plan_df)}` expanding temporal folds so performance is measured repeatedly across different future windows instead of one validation/test boundary.",
        "",
        "### Fold Schedule",
        rolling_plan_df.to_markdown(index=False),
        "",
        "### Best Regression Model Backtest",
        regression_backtest_df.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]

    if best_regression_backtest is not None:
        lines.append(
            f"Regression stability summary: mean RMSE `{best_regression_backtest['mean_rmse']:,.2f}` with fold-to-fold SD `{best_regression_backtest['std_rmse']:,.2f}`."
        )

    lines.extend(
        [
            "",
            "### Best Classification Model Backtest",
            classification_backtest_df.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    )

    if best_classification_backtest is not None:
        lines.append(
            f"Classification stability summary: mean ROC AUC `{best_classification_backtest['mean_roc_auc']:.4f}` with fold-to-fold SD `{best_classification_backtest['std_roc_auc']:.4f}`."
        )

    lines.extend(
        [
            "",
            "## Regression Feature Ablation And Robustness",
            "",
            "Ablation holds the best saved regression estimator family fixed and changes only the available feature families across the same rolling folds. These rows are constrained diagnostic variants, not replacements for the main full-pipeline leaderboard.",
            "",
            ablation_df.to_markdown(index=False, floatfmt=".4f"),
            "",
            "### Paired Significance Checks",
            ablation_significance_df.to_markdown(index=False, floatfmt=".4f"),
            "",
            f"Best ablation result across folds: `{best_ablation['variant']}` with mean RMSE `{best_ablation['mean_rmse']:,.2f}`, mean MAE `{best_ablation['mean_mae']:,.2f}`, and mean R² `{best_ablation['mean_r2']:.4f}`.",
            "",
            "## Raw-Target vs Log-Target Backtest",
            "",
            "The comparison below keeps the best regression estimator family fixed and changes only the target treatment across the same rolling folds.",
            "",
            target_compare_df.to_markdown(index=False, floatfmt=".4f"),
            "",
            f"Best target treatment: `{best_target['target_version']}` with mean RMSE `{best_target['mean_rmse']:,.2f}`, mean MAE `{best_target['mean_mae']:,.2f}`, and mean R² `{best_target['mean_r2']:.4f}`.",
        ]
    )
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
    parser.add_argument(
        "--classification-quantile",
        type=float,
        default=0.75,
        help="Training-fold quantile used to define the high-value class inside rolling backtests.",
    )
    parser.add_argument(
        "--backtest-splits",
        type=int,
        default=5,
        help="Number of expanding-window folds used in the appendix robustness analysis.",
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

    best_regression_name = load_best_model_name(args.artifact_dir.resolve(), task="regression")
    best_classification_name = load_best_model_name(args.artifact_dir.resolve(), task="classification")

    regression_backtest_folds, regression_backtest_df, rolling_plan_df = rolling_origin_backtest(
        artifact_dir=args.artifact_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        task="regression",
        model_name=best_regression_name,
        classification_quantile=args.classification_quantile,
        backtest_splits=args.backtest_splits,
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs,
    )
    classification_backtest_folds, classification_backtest_df, _ = rolling_origin_backtest(
        artifact_dir=args.artifact_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        task="classification",
        model_name=best_classification_name,
        classification_quantile=args.classification_quantile,
        backtest_splits=args.backtest_splits,
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs,
    )
    ablation_folds_df, ablation_df = regression_ablation_backtest(
        artifact_dir=args.artifact_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        backtest_splits=args.backtest_splits,
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs,
    )
    ablation_significance_df = regression_ablation_significance(
        fold_df=ablation_folds_df,
        random_state=RANDOM_STATE,
    )
    target_compare_df = regression_target_treatment_backtest(
        artifact_dir=args.artifact_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        backtest_splits=args.backtest_splits,
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs,
    )

    save_dataframe(search_space_df, directories["tables"] / "exact_hyperparameter_search_spaces.csv")
    save_dataframe(rolling_plan_df, directories["tables"] / "rolling_origin_backtest_plan.csv")
    save_dataframe(regression_backtest_folds, directories["tables"] / "regression_rolling_origin_backtest_folds.csv")
    save_dataframe(regression_backtest_df, directories["tables"] / "regression_rolling_origin_backtest_summary.csv")
    save_dataframe(
        classification_backtest_folds,
        directories["tables"] / "classification_rolling_origin_backtest_folds.csv",
    )
    save_dataframe(
        classification_backtest_df,
        directories["tables"] / "classification_rolling_origin_backtest_summary.csv",
    )
    save_dataframe(ablation_folds_df, directories["tables"] / "regression_ablation_backtest_folds.csv")
    save_dataframe(ablation_df, directories["tables"] / "regression_ablation_table.csv")
    save_dataframe(ablation_significance_df, directories["tables"] / "regression_ablation_significance.csv")
    save_dataframe(target_compare_df, directories["tables"] / "raw_vs_log_target_comparison.csv")
    build_appendix_markdown(
        output_path=directories["summaries"] / "appendix_modeling_detail.md",
        search_space_df=search_space_df,
        rolling_plan_df=rolling_plan_df,
        regression_backtest_df=regression_backtest_df,
        classification_backtest_df=classification_backtest_df,
        ablation_df=ablation_df,
        ablation_significance_df=ablation_significance_df,
        target_compare_df=target_compare_df,
    )

    print(f"Saved appendix artifacts to {directories['root']}")


if __name__ == "__main__":
    main()
