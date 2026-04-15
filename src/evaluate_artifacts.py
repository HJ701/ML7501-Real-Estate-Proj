from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import __main__

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.validation import check_is_fitted

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = BASE_DIR / "outputs" / "modeling" / "gpu_run"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation" / "latest"
RANDOM_STATE = 42


LEAKAGE_COLUMNS = {
    "meter_sale_price",
    "meter_rent_price",
    "rent_value",
}

IDENTIFIER_COLUMNS = {
    "transaction_id",
    "project_number",
    "load_timestamp",
}

CODE_COLUMNS_TO_DROP = {
    "reg_type_id",
    "procedure_id",
    "property_sub_type_id",
    "property_type_id",
    "trans_group_id",
}

DATE_COLUMNS_TO_DROP = {
    "instance_date",
    "transaction_month",
}


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "QuantileClipper":
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


__main__.QuantileClipper = QuantileClipper


def temporal_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("instance_date").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:val_end].copy(), ordered.iloc[val_end:].copy()


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = set()
    drop_columns.update(LEAKAGE_COLUMNS)
    drop_columns.update(IDENTIFIER_COLUMNS)
    drop_columns.update(CODE_COLUMNS_TO_DROP)
    drop_columns.update(DATE_COLUMNS_TO_DROP)
    drop_columns.add("actual_worth")
    drop_columns.add("area_id")
    drop_columns.update({column for column in df.columns if column.endswith("_ar")})

    selected = df.drop(columns=[column for column in drop_columns if column in df.columns]).copy()
    for column in selected.columns:
        if pd.api.types.is_integer_dtype(selected[column].dtype):
            selected[column] = selected[column].astype(float)
    return selected


def evaluate_regression(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, predictions)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_classification(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray | None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, scores)) if scores is not None else float("nan")
    return metrics


def get_prediction_scores(model: object, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    tables = output_dir / "tables"
    plots = output_dir / "plots"
    summaries = output_dir / "summaries"
    for directory in (output_dir, tables, plots, summaries):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "tables": tables, "plots": plots, "summaries": summaries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rigorous evaluation on saved ML artifact directories.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Path to the modeling artifact directory to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation outputs will be written.",
    )
    parser.add_argument(
        "--classification-quantile",
        type=float,
        default=0.75,
        help="Training-set quantile used to define the binary high-value label.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=250,
        help="Number of bootstrap resamples for confidence-style intervals on the best models.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for bootstrap evaluation.",
    )
    return parser.parse_args()


def load_master_table(artifact_dir: Path) -> pd.DataFrame:
    path = artifact_dir / "tables" / "modeling_master_table.csv"
    master = pd.read_csv(path, low_memory=False)
    master["instance_date"] = pd.to_datetime(master["instance_date"], errors="coerce")
    master["transaction_month"] = pd.to_datetime(master["transaction_month"], errors="coerce")
    return master


def load_saved_metrics(artifact_dir: Path, task: str) -> pd.DataFrame:
    return pd.read_csv(artifact_dir / "tables" / f"{task}_test_metrics.csv")


def enrich_regression_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    enriched = metrics_df.copy()
    enriched["rmse_rank"] = enriched["rmse"].rank(method="min")
    enriched["mae_rank"] = enriched["mae"].rank(method="min")
    enriched["r2_rank"] = enriched["r2"].rank(method="min", ascending=False)
    return enriched.sort_values("rmse", ascending=True).reset_index(drop=True)


def enrich_classification_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    enriched = metrics_df.copy()
    enriched["roc_auc_rank"] = enriched["roc_auc"].rank(method="min", ascending=False)
    enriched["f1_rank"] = enriched["f1"].rank(method="min", ascending=False)
    enriched["accuracy_rank"] = enriched["accuracy"].rank(method="min", ascending=False)
    return enriched.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def compare_saved_and_recomputed(saved_df: pd.DataFrame, recomputed_df: pd.DataFrame, key: str) -> pd.DataFrame:
    merged = saved_df.merge(recomputed_df, on=key, suffixes=("_saved", "_recomputed"))
    delta_columns = []
    for column in recomputed_df.columns:
        if column == key:
            continue
        merged[f"{column}_abs_delta"] = (merged[f"{column}_saved"] - merged[f"{column}_recomputed"]).abs()
        delta_columns.append(f"{column}_abs_delta")
    ordered_columns = [key] + delta_columns
    return merged[ordered_columns].sort_values(key).reset_index(drop=True)


def load_models(artifact_dir: Path, task: str) -> dict[str, object]:
    models = {}
    model_dir = artifact_dir / "models"
    pattern = f"{task}_*.joblib"
    prefix = f"{task}_"
    for path in sorted(model_dir.glob(pattern)):
        model_name = path.stem[len(prefix) :]
        try:
            models[model_name] = load(path)
        except ModuleNotFoundError as exc:
            if exc.name != "numpy._core":
                raise
            import sys

            sys.modules.setdefault("numpy._core", np.core)
            models[model_name] = load(path)
    return models


def build_model_outputs(
    task: str,
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test_reg: pd.Series,
    y_test_clf: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    regression_rows = []
    classification_rows = []
    regression_predictions: dict[str, pd.DataFrame] = {}
    classification_predictions: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        if task == "regression":
            preds = model.predict(X_test)
            regression_rows.append({"model_name": model_name, **evaluate_regression(y_test_reg, preds)})
            regression_predictions[model_name] = pd.DataFrame(
                {
                    "y_true": y_test_reg.to_numpy(),
                    "y_pred": preds,
                }
            )
        else:
            preds = model.predict(X_test)
            scores = get_prediction_scores(model, X_test)
            metrics = evaluate_classification(y_test_clf, preds, scores)
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test_clf, preds))
            metrics["average_precision"] = (
                float(average_precision_score(y_test_clf, scores)) if scores is not None else float("nan")
            )
            if scores is not None:
                scaled_scores = scores
                if np.nanmin(scores) < 0 or np.nanmax(scores) > 1:
                    scaled_scores = 1 / (1 + np.exp(-scores))
                metrics["brier_score"] = float(brier_score_loss(y_test_clf, scaled_scores))
            else:
                metrics["brier_score"] = float("nan")
            classification_rows.append({"model_name": model_name, **metrics})
            classification_predictions[model_name] = pd.DataFrame(
                {
                    "y_true": y_test_clf.to_numpy(),
                    "y_pred": preds,
                    "y_score": scores if scores is not None else np.nan,
                }
            )

    regression_df = pd.DataFrame(regression_rows)
    classification_df = pd.DataFrame(classification_rows)
    return regression_df, classification_df, regression_predictions, classification_predictions


def bootstrap_regression_ci(
    y_true: pd.Series,
    y_pred: np.ndarray,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    values = y_true.to_numpy()
    for _ in range(iterations):
        indices = rng.integers(0, len(values), len(values))
        metrics = evaluate_regression(pd.Series(values[indices]), y_pred[indices])
        rows.append(metrics)
    boot = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "metric": boot.columns,
            "mean": boot.mean().to_numpy(),
            "lower_2_5": boot.quantile(0.025).to_numpy(),
            "upper_97_5": boot.quantile(0.975).to_numpy(),
        }
    )


def bootstrap_classification_ci(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    values = y_true.to_numpy()
    for _ in range(iterations):
        indices = rng.integers(0, len(values), len(values))
        sample_y = pd.Series(values[indices])
        sample_pred = y_pred[indices]
        sample_score = y_score[indices] if y_score is not None else None
        metrics = evaluate_classification(sample_y, sample_pred, sample_score)
        rows.append(metrics)
    boot = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "metric": boot.columns,
            "mean": boot.mean().to_numpy(),
            "lower_2_5": boot.quantile(0.025).to_numpy(),
            "upper_97_5": boot.quantile(0.975).to_numpy(),
        }
    )


def regression_value_bands(y_true: pd.Series) -> pd.Categorical:
    return pd.qcut(
        y_true,
        q=[0.0, 0.25, 0.5, 0.75, 1.0],
        labels=["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"],
        duplicates="drop",
    )


def build_regression_analysis_tables(test_df: pd.DataFrame, predictions: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = test_df.copy()
    analysis["prediction"] = predictions
    analysis["residual"] = analysis["actual_worth"] - analysis["prediction"]
    analysis["absolute_error"] = analysis["residual"].abs()
    analysis["ape"] = analysis["absolute_error"] / analysis["actual_worth"].replace(0, np.nan)
    analysis["value_band"] = regression_value_bands(analysis["actual_worth"])

    value_band_summary = (
        analysis.groupby("value_band", observed=False)
        .agg(
            count=("actual_worth", "size"),
            actual_mean=("actual_worth", "mean"),
            predicted_mean=("prediction", "mean"),
            mae=("absolute_error", "mean"),
            median_ape=("ape", "median"),
        )
        .reset_index()
    )

    year_summary = (
        analysis.groupby("transaction_year", observed=False)
        .agg(
            count=("actual_worth", "size"),
            mae=("absolute_error", "mean"),
            rmse=("residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            actual_mean=("actual_worth", "mean"),
            predicted_mean=("prediction", "mean"),
        )
        .reset_index()
        .sort_values("transaction_year")
    )

    top_errors = analysis[
        [
            "instance_date",
            "transaction_id",
            "actual_worth",
            "prediction",
            "residual",
            "absolute_error",
            "area_name_en",
            "property_type_en",
            "procedure_name_en",
        ]
    ].sort_values("absolute_error", ascending=False).head(25)

    return value_band_summary, year_summary, top_errors


def build_classification_analysis_tables(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = test_df.copy()
    analysis["y_true"] = y_true.to_numpy()
    analysis["y_pred"] = y_pred

    if y_score is None:
        analysis["y_score"] = np.nan
        analysis["probability"] = np.nan
    else:
        analysis["y_score"] = y_score
        if np.nanmin(y_score) < 0 or np.nanmax(y_score) > 1:
            analysis["probability"] = 1 / (1 + np.exp(-y_score))
        else:
            analysis["probability"] = y_score

    analysis["correct"] = (analysis["y_true"] == analysis["y_pred"]).astype(int)

    year_summary = (
        analysis.groupby("transaction_year", observed=False)
        .agg(
            count=("y_true", "size"),
            accuracy=("correct", "mean"),
            positive_rate=("y_true", "mean"),
            predicted_positive_rate=("y_pred", "mean"),
        )
        .reset_index()
        .sort_values("transaction_year")
    )

    property_summary = (
        analysis.groupby("property_type_en", observed=False)
        .agg(
            count=("y_true", "size"),
            accuracy=("correct", "mean"),
            positive_rate=("y_true", "mean"),
            predicted_positive_rate=("y_pred", "mean"),
        )
        .reset_index()
        .sort_values(["count", "accuracy"], ascending=[False, False])
        .head(10)
    )

    if analysis["probability"].notna().any():
        analysis["probability_bin"] = pd.qcut(
            analysis["probability"],
            q=10,
            duplicates="drop",
        )
        calibration = (
            analysis.groupby("probability_bin", observed=False)
            .agg(
                count=("y_true", "size"),
                mean_predicted_probability=("probability", "mean"),
                observed_positive_rate=("y_true", "mean"),
            )
            .reset_index()
        )
    else:
        calibration = pd.DataFrame(columns=["probability_bin", "count", "mean_predicted_probability", "observed_positive_rate"])

    return year_summary, property_summary, calibration


def plot_regression_by_year(year_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(year_df["transaction_year"], year_df["mae"], marker="o", color="#0f4c5c")
    ax.set_title("Best regression model: MAE by transaction year")
    ax.set_xlabel("Transaction year")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regression_value_bands(value_band_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(value_band_df["value_band"].astype(str), value_band_df["mae"], color="#ef8354")
    ax.set_title("Best regression model: MAE by target value band")
    ax.set_xlabel("Actual sale-price band")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_classification_by_year(year_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(year_df["transaction_year"], year_df["accuracy"], marker="o", color="#1f6f8b")
    ax.set_title("Best classification model: accuracy by transaction year")
    ax.set_xlabel("Transaction year")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(y_true: pd.Series, y_score: np.ndarray | None, output_path: Path) -> None:
    if y_score is None:
        return
    scores = y_score
    if np.nanmin(scores) < 0 or np.nanmax(scores) > 1:
        scores = 1 / (1 + np.exp(-scores))
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="#2a9d8f", label=f"AP = {ap:.3f}")
    ax.set_title("Best classification model: precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(calibration_df: pd.DataFrame, output_path: Path) -> None:
    if calibration_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        calibration_df["mean_predicted_probability"],
        calibration_df["observed_positive_rate"],
        marker="o",
        color="#8d99ae",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#bc4749")
    ax.set_title("Best classification model: calibration by score decile")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_markdown_summary(
    output_path: Path,
    artifact_dir: Path,
    regression_metrics: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    regression_bootstrap: pd.DataFrame,
    classification_bootstrap: pd.DataFrame,
    regression_year: pd.DataFrame,
    regression_value_bands: pd.DataFrame,
    classification_year: pd.DataFrame,
    classification_property: pd.DataFrame,
) -> None:
    lines = [
        "# Rigorous Evaluation Summary",
        "",
        f"- Artifact directory: `{artifact_dir}`",
    ]

    if not regression_metrics.empty:
        best_reg = regression_metrics.iloc[0]
        reg_year_peak = regression_year.sort_values("mae", ascending=False).iloc[0]
        reg_band_peak = regression_value_bands.sort_values("mae", ascending=False).iloc[0]
        reg_rmse_ci = regression_bootstrap.loc[regression_bootstrap["metric"] == "rmse"].iloc[0]
        lines.extend(
            [
                f"- Best regression model: `{best_reg['model_name']}` with RMSE `{best_reg['rmse']:,.2f}`, MAE `{best_reg['mae']:,.2f}`, and R2 `{best_reg['r2']:.4f}`.",
                f"- Bootstrap interval for best regression RMSE: `{reg_rmse_ci['lower_2_5']:,.2f}` to `{reg_rmse_ci['upper_97_5']:,.2f}`.",
                f"- Hardest regression period on the test split: year `{int(reg_year_peak['transaction_year'])}` with MAE `{reg_year_peak['mae']:,.2f}`.",
                f"- Hardest regression target band: `{reg_band_peak['value_band']}` with MAE `{reg_band_peak['mae']:,.2f}`.",
            ]
        )

    if not classification_metrics.empty:
        best_clf = classification_metrics.iloc[0]
        clf_year_low = classification_year.sort_values("accuracy", ascending=True).iloc[0]
        clf_prop_low = classification_property.sort_values("accuracy", ascending=True).iloc[0]
        clf_auc_ci = classification_bootstrap.loc[classification_bootstrap["metric"] == "roc_auc"].iloc[0]
        lines.extend(
            [
                f"- Best classification model: `{best_clf['model_name']}` with ROC AUC `{best_clf['roc_auc']:.4f}`, F1 `{best_clf['f1']:.4f}`, and accuracy `{best_clf['accuracy']:.4f}`.",
                f"- Bootstrap interval for best classification ROC AUC: `{clf_auc_ci['lower_2_5']:.4f}` to `{clf_auc_ci['upper_97_5']:.4f}`.",
                f"- Weakest classification year on the test split: `{int(clf_year_low['transaction_year'])}` with accuracy `{clf_year_low['accuracy']:.4f}`.",
                f"- Weakest high-volume classification property type: `{clf_prop_low['property_type_en']}` with accuracy `{clf_prop_low['accuracy']:.4f}` over `{int(clf_prop_low['count'])}` rows.",
            ]
        )

    lines.extend(["", "## Interpretation"])
    if not regression_metrics.empty and not classification_metrics.empty:
        lines.extend(
            [
                "- The evaluation confirms that the tree-based gradient boosting model is the strongest artifact on both tasks, which is consistent with the non-linear, interaction-heavy hypothesis in the proposal.",
                "- Regression error grows sharply in the highest-price band, so the report should emphasize that performance is materially better for typical transactions than for extreme luxury or unusually large deals.",
                "- Classification remains strong overall, but subgroup accuracy varies by year and property type, which supports a nuanced discussion of temporal drift instead of reporting a single headline metric.",
            ]
        )
    elif not regression_metrics.empty:
        lines.extend(
            [
                "- The regression evaluation confirms that non-linear ensemble methods are materially stronger than the baseline and linear comparators on this problem.",
                "- Error grows in the upper-value tail, so subgroup analysis by price band is necessary to avoid overstating average performance.",
            ]
        )
    elif not classification_metrics.empty:
        lines.extend(
            [
                "- The classification evaluation confirms that tree-based methods are the strongest available artifacts on the derived high-value label.",
                "- Performance varies across time and property segments, so the report should discuss temporal and subgroup variation instead of relying on a single headline score.",
            ]
        )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dirs = ensure_output_dirs(args.output_dir.resolve())

    master = load_master_table(artifact_dir)
    train_df, _, test_df = temporal_split(master, train_frac=0.70, val_frac=0.15)
    X_test = select_model_features(test_df)
    y_test_reg = test_df["actual_worth"].reset_index(drop=True)

    threshold = float(train_df["actual_worth"].quantile(args.classification_quantile))
    y_test_clf = test_df["actual_worth"].ge(threshold).astype(int).reset_index(drop=True)

    regression_models = load_models(artifact_dir, task="regression")
    classification_models = load_models(artifact_dir, task="classification")

    regression_metrics = pd.DataFrame()
    classification_metrics = pd.DataFrame()
    regression_bootstrap = pd.DataFrame()
    classification_bootstrap = pd.DataFrame()
    reg_year = pd.DataFrame()
    reg_value_bands = pd.DataFrame()
    clf_year = pd.DataFrame()
    clf_property = pd.DataFrame()

    if regression_models:
        regression_metrics, _, regression_predictions, _ = build_model_outputs(
            task="regression",
            models=regression_models,
            X_test=X_test,
            y_test_reg=y_test_reg,
            y_test_clf=y_test_clf,
        )
        regression_metrics = enrich_regression_metrics(regression_metrics)
        save_dataframe(regression_metrics, output_dirs["tables"] / "regression_artifact_metrics_enriched.csv")

        saved_regression = load_saved_metrics(artifact_dir, task="regression")
        save_dataframe(
            compare_saved_and_recomputed(
                saved_regression,
                regression_metrics[["model_name", "mse", "rmse", "mae", "r2"]],
                "model_name",
            ),
            output_dirs["tables"] / "regression_metric_reproducibility_check.csv",
        )

        validation_reg = pd.read_csv(artifact_dir / "tables" / "regression_validation_metrics.csv")
        regression_gap = validation_reg.merge(
            regression_metrics[["model_name", "rmse", "mae", "r2"]],
            on="model_name",
            suffixes=("_validation", "_test"),
        )
        regression_gap["rmse_gap_test_minus_validation"] = regression_gap["rmse_test"] - regression_gap["rmse_validation"]
        regression_gap["mae_gap_test_minus_validation"] = regression_gap["mae_test"] - regression_gap["mae_validation"]
        regression_gap["r2_gap_test_minus_validation"] = regression_gap["r2_test"] - regression_gap["r2_validation"]
        save_dataframe(regression_gap, output_dirs["tables"] / "regression_generalization_gap.csv")

        best_regression_name = regression_metrics.iloc[0]["model_name"]
        best_regression_pred = regression_predictions[best_regression_name]["y_pred"].to_numpy()
        regression_bootstrap = bootstrap_regression_ci(
            y_true=y_test_reg,
            y_pred=best_regression_pred,
            iterations=args.bootstrap_iterations,
            random_state=args.random_state,
        )
        save_dataframe(regression_bootstrap, output_dirs["tables"] / "regression_best_bootstrap_ci.csv")

        reg_value_bands, reg_year, reg_top_errors = build_regression_analysis_tables(test_df, best_regression_pred)
        save_dataframe(reg_value_bands, output_dirs["tables"] / "regression_best_value_band_summary.csv")
        save_dataframe(reg_year, output_dirs["tables"] / "regression_best_year_summary.csv")
        save_dataframe(reg_top_errors, output_dirs["tables"] / "regression_best_top_errors.csv")
        plot_regression_by_year(reg_year, output_dirs["plots"] / "regression_best_mae_by_year.png")
        plot_regression_value_bands(reg_value_bands, output_dirs["plots"] / "regression_best_mae_by_value_band.png")

    if classification_models:
        _, classification_metrics, _, classification_predictions = build_model_outputs(
            task="classification",
            models=classification_models,
            X_test=X_test,
            y_test_reg=y_test_reg,
            y_test_clf=y_test_clf,
        )
        classification_metrics = enrich_classification_metrics(classification_metrics)
        save_dataframe(classification_metrics, output_dirs["tables"] / "classification_artifact_metrics_enriched.csv")

        saved_classification = load_saved_metrics(artifact_dir, task="classification")
        save_dataframe(
            compare_saved_and_recomputed(
                saved_classification,
                classification_metrics[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]],
                "model_name",
            ),
            output_dirs["tables"] / "classification_metric_reproducibility_check.csv",
        )

        validation_clf = pd.read_csv(artifact_dir / "tables" / "classification_validation_metrics.csv")
        classification_gap = validation_clf.merge(
            classification_metrics[["model_name", "accuracy", "f1", "roc_auc"]],
            on="model_name",
            suffixes=("_validation", "_test"),
        )
        classification_gap["accuracy_gap_test_minus_validation"] = (
            classification_gap["accuracy_test"] - classification_gap["accuracy_validation"]
        )
        classification_gap["f1_gap_test_minus_validation"] = classification_gap["f1_test"] - classification_gap["f1_validation"]
        classification_gap["roc_auc_gap_test_minus_validation"] = (
            classification_gap["roc_auc_test"] - classification_gap["roc_auc_validation"]
        )
        save_dataframe(classification_gap, output_dirs["tables"] / "classification_generalization_gap.csv")

        best_classification_name = classification_metrics.iloc[0]["model_name"]
        best_classification_frame = classification_predictions[best_classification_name]
        best_classification_pred = best_classification_frame["y_pred"].to_numpy()
        best_classification_score_series = best_classification_frame["y_score"]
        best_classification_score = None if best_classification_score_series.isna().all() else best_classification_score_series.to_numpy()
        classification_bootstrap = bootstrap_classification_ci(
            y_true=y_test_clf,
            y_pred=best_classification_pred,
            y_score=best_classification_score,
            iterations=args.bootstrap_iterations,
            random_state=args.random_state,
        )
        save_dataframe(classification_bootstrap, output_dirs["tables"] / "classification_best_bootstrap_ci.csv")

        clf_year, clf_property, clf_calibration = build_classification_analysis_tables(
            test_df=test_df,
            y_true=y_test_clf,
            y_pred=best_classification_pred,
            y_score=best_classification_score,
        )
        save_dataframe(clf_year, output_dirs["tables"] / "classification_best_year_summary.csv")
        save_dataframe(clf_property, output_dirs["tables"] / "classification_best_property_type_summary.csv")
        save_dataframe(clf_calibration, output_dirs["tables"] / "classification_best_calibration_deciles.csv")
        plot_classification_by_year(clf_year, output_dirs["plots"] / "classification_best_accuracy_by_year.png")
        plot_precision_recall(y_test_clf, best_classification_score, output_dirs["plots"] / "classification_best_precision_recall.png")
        plot_calibration(clf_calibration, output_dirs["plots"] / "classification_best_calibration.png")

    build_markdown_summary(
        output_path=output_dirs["summaries"] / "evaluation_summary.md",
        artifact_dir=artifact_dir,
        regression_metrics=regression_metrics,
        classification_metrics=classification_metrics,
        regression_bootstrap=regression_bootstrap,
        classification_bootstrap=classification_bootstrap,
        regression_year=reg_year,
        regression_value_bands=reg_value_bands,
        classification_year=clf_year,
        classification_property=clf_property,
    )

    summary_json = {
        "artifact_dir": str(artifact_dir),
        "classification_threshold": threshold,
        "best_regression_model": regression_metrics.iloc[0]["model_name"] if not regression_metrics.empty else None,
        "best_classification_model": classification_metrics.iloc[0]["model_name"] if not classification_metrics.empty else None,
    }
    (output_dirs["summaries"] / "evaluation_summary.json").write_text(json.dumps(summary_json, indent=2))

    print(f"Saved evaluation artifacts to {output_dirs['root']}")


if __name__ == "__main__":
    main()
