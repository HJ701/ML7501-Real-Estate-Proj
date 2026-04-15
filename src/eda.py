from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "eda"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    date_columns: tuple[str, ...]
    target_column: str | None = None


def make_dataset_configs(data_dir: Path) -> tuple[DatasetConfig, ...]:
    return (
        DatasetConfig(
            name="transactions",
            path=data_dir / "Real-estate_Transactions_2026-03-27.csv",
            date_columns=("instance_date", "load_timestamp"),
            target_column="actual_worth",
        ),
        DatasetConfig(
            name="rent_contracts",
            path=data_dir / "rent_contracts.csv",
            date_columns=("contract_start_date", "contract_end_date", "load_timestamp"),
            target_column="annual_amount",
        ),
        DatasetConfig(
            name="hotel_stats",
            path=data_dir / "FCSA,DF_HOT_TYPE,4.3.0+...A.....csv",
            date_columns=("TIME_PERIOD",),
            target_column="OBS_VALUE",
        ),
    )


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, plot_dir


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(config.path, low_memory=False)
    for column in config.date_columns:
        if column not in df.columns:
            continue
        if column == "TIME_PERIOD":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        else:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_rows = len(df)
    for column in df.columns:
        non_null = df[column].dropna()
        sample_values = [str(value) for value in non_null.astype(str).unique()[:3]]
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": round(float(df[column].isna().mean() * 100), 2),
                "unique_count": int(df[column].nunique(dropna=True)),
                "non_null_count": int(non_null.shape[0]),
                "non_null_pct": round(float((non_null.shape[0] / total_rows) * 100), 2),
                "sample_values": " | ".join(sample_values),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "unique_count"], ascending=[False, False])


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary["missing_count"] = numeric_df.isna().sum()
    summary["missing_pct"] = numeric_df.isna().mean().mul(100).round(2)
    summary["zero_count"] = (numeric_df == 0).sum()
    summary["negative_count"] = (numeric_df < 0).sum()
    return summary.reset_index().rename(columns={"index": "column"})


def dataset_overview(name: str, df: pd.DataFrame, date_columns: tuple[str, ...]) -> dict[str, object]:
    date_count = sum(1 for column in date_columns if column in df.columns)
    future_dated_rows = 0
    for column in date_columns:
        if column in df.columns and pd.api.types.is_datetime64_any_dtype(df[column]):
            future_dated_rows += int((df[column] > pd.Timestamp.today()).sum())

    return {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": int(df.select_dtypes(include=[np.number]).shape[1]),
        "object_columns": int(df.select_dtypes(include=["object"]).shape[1]),
        "datetime_columns": int(date_count),
        "future_dated_rows": future_dated_rows,
    }


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_pct": round(float(outlier_mask.mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def top_correlations(df: pd.DataFrame, target: str, top_n: int = 12) -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric_df.columns:
        return pd.DataFrame()

    correlations = numeric_df.corr(numeric_only=True)[target].dropna()
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    return correlations.head(top_n).reset_index().rename(columns={"index": "feature", target: "correlation"})


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, index=False)


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, dataset_name: str, plot_dir: Path, top_n: int = 15) -> None:
    missing = df.isna().mean().mul(100).sort_values(ascending=False).head(top_n)
    missing = missing[missing > 0]
    if missing.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, palette="crest")
    ax.set_title(f"{dataset_name.replace('_', ' ').title()}: Top Missing Columns")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("")
    save_plot(fig, plot_dir / f"{dataset_name}_missingness_top{top_n}.png")


def plot_distribution(df: pd.DataFrame, column: str, dataset_name: str, plot_dir: Path) -> None:
    if column not in df.columns:
        return
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return

    clipped = series.clip(upper=series.quantile(0.99))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(clipped, kde=True, bins=40, ax=axes[0], color="#16697a")
    axes[0].set_title(f"{column} distribution (trimmed at 99th pct)")
    axes[0].set_xlabel(column)

    positive = series[series > 0]
    if not positive.empty:
        sns.histplot(np.log1p(positive), kde=True, bins=40, ax=axes[1], color="#f4a261")
        axes[1].set_title(f"log1p({column}) distribution")
        axes[1].set_xlabel(f"log1p({column})")
    else:
        axes[1].text(0.5, 0.5, "No positive values for log plot", ha="center", va="center")
        axes[1].set_axis_off()

    save_plot(fig, plot_dir / f"{dataset_name}_{slugify(column)}_distribution.png")


def plot_top_categories(df: pd.DataFrame, column: str, dataset_name: str, plot_dir: Path, top_n: int = 10) -> None:
    if column not in df.columns:
        return
    counts = df[column].fillna("Missing").astype(str).value_counts().head(top_n)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title(f"{dataset_name.replace('_', ' ').title()}: Top {top_n} values for {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    save_plot(fig, plot_dir / f"{dataset_name}_{slugify(column)}_top_categories.png")


def plot_value_by_category(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    dataset_name: str,
    plot_dir: Path,
    top_n: int = 10,
) -> None:
    if value_col not in df.columns or category_col not in df.columns:
        return

    working = df[[value_col, category_col]].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working[category_col] = working[category_col].fillna("Missing").astype(str)
    top_categories = working[category_col].value_counts().head(top_n).index
    working = working[working[category_col].isin(top_categories)].dropna(subset=[value_col])
    if working.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=working, x=category_col, y=value_col, ax=ax, showfliers=False, color="#8ecae6")
    ax.set_title(f"{value_col} by {category_col}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    if (working[value_col] > 0).all():
        ax.set_yscale("log")
        ax.set_ylabel(f"{value_col} (log scale)")
    save_plot(fig, plot_dir / f"{dataset_name}_{slugify(value_col)}_by_{slugify(category_col)}.png")


def plot_time_trend(df: pd.DataFrame, date_col: str, value_col: str, dataset_name: str, plot_dir: Path) -> None:
    if date_col not in df.columns or value_col not in df.columns:
        return
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return

    working = df[[date_col, value_col]].copy().dropna(subset=[date_col])
    working["year"] = working[date_col].dt.year
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    yearly = working.groupby("year").agg(record_count=(value_col, "size"), median_value=(value_col, "median")).reset_index()
    if yearly.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sns.barplot(data=yearly, x="year", y="record_count", ax=axes[0], color="#264653")
    axes[0].set_title(f"{dataset_name.replace('_', ' ').title()}: records by year")
    axes[0].set_ylabel("Record count")

    sns.lineplot(data=yearly, x="year", y="median_value", marker="o", ax=axes[1], color="#e76f51")
    axes[1].set_title(f"{dataset_name.replace('_', ' ').title()}: median {value_col} by year")
    axes[1].set_ylabel(f"Median {value_col}")
    axes[1].set_xlabel("Year")
    save_plot(fig, plot_dir / f"{dataset_name}_{slugify(value_col)}_time_trend.png")


def plot_correlation_heatmap(df: pd.DataFrame, dataset_name: str, target: str, plot_dir: Path, max_features: int = 12) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty or target not in numeric_df.columns:
        return

    corr_target = numeric_df.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
    selected_columns = corr_target.head(max_features).index.tolist()
    corr_matrix = numeric_df[selected_columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{dataset_name.replace('_', ' ').title()}: correlation heatmap")
    save_plot(fig, plot_dir / f"{dataset_name}_correlation_heatmap.png")


def plot_outlier_boxplots(df: pd.DataFrame, columns: list[str], dataset_name: str, plot_dir: Path) -> None:
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return

    working = df[available_columns].apply(pd.to_numeric, errors="coerce")
    melted = working.melt(var_name="feature", value_name="value").dropna()
    if melted.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", ax=ax, showfliers=False, color="#bde0fe")
    ax.tick_params(axis="x", rotation=30)
    if melted["value"].gt(0).all():
        ax.set_yscale("log")
        ax.set_ylabel("Value (log scale)")
    ax.set_title(f"{dataset_name.replace('_', ' ').title()}: numeric spread")
    save_plot(fig, plot_dir / f"{dataset_name}_numeric_boxplots.png")


def plot_hotel_trends(df: pd.DataFrame, plot_dir: Path) -> None:
    required = {"TIME_PERIOD", "Hotels Indicator", "Hotel Type", "OBS_VALUE"}
    if not required.issubset(df.columns):
        return

    working = df[list(required)].copy()
    working["OBS_VALUE"] = pd.to_numeric(working["OBS_VALUE"], errors="coerce")
    working = working.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    if working.empty:
        return

    top_hotel_types = [
        "Total Hotels and Hotel Apartments",
        "Total Hotels",
        "Total Hotel Apartments",
        "5 Star",
        "4 Star",
        "1-3 Star",
    ]
    working = working[working["Hotel Type"].isin(top_hotel_types)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    for idx, indicator in enumerate(sorted(working["Hotels Indicator"].dropna().unique())):
        subset = working[working["Hotels Indicator"] == indicator]
        sns.lineplot(
            data=subset,
            x="TIME_PERIOD",
            y="OBS_VALUE",
            hue="Hotel Type",
            marker="o",
            ax=axes[idx],
        )
        axes[idx].set_title(indicator)
        axes[idx].set_xlabel("Year")
        axes[idx].set_ylabel("Observation value")
        axes[idx].legend(loc="best", fontsize=8)

    save_plot(fig, plot_dir / "hotel_stats_obs_value_trends.png")


def join_key_overlap(transactions: pd.DataFrame, rents: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("area_id", "area_id"),
        ("area_name_en", "area_name_en"),
        ("master_project_en", "master_project_en"),
        ("project_name_en", "project_name_en"),
        ("property_sub_type_en", "ejari_property_sub_type_en"),
    ]
    rows = []
    for left_col, right_col in pairs:
        if left_col not in transactions.columns or right_col not in rents.columns:
            continue
        left_values = set(transactions[left_col].dropna())
        right_values = set(rents[right_col].dropna())
        overlap = left_values & right_values
        rows.append(
            {
                "transactions_key": left_col,
                "rent_key": right_col,
                "transactions_unique": len(left_values),
                "rent_unique": len(right_values),
                "overlap_unique": len(overlap),
                "transactions_coverage_pct": round((len(overlap) / len(left_values)) * 100, 2) if left_values else 0.0,
                "rent_coverage_pct": round((len(overlap) / len(right_values)) * 100, 2) if right_values else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_join_overlap(overlap_df: pd.DataFrame, plot_dir: Path) -> None:
    if overlap_df.empty:
        return

    plot_df = overlap_df.melt(
        id_vars=["transactions_key"],
        value_vars=["transactions_coverage_pct", "rent_coverage_pct"],
        var_name="dataset_side",
        value_name="coverage_pct",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="coverage_pct", y="transactions_key", hue="dataset_side", ax=ax, palette="Set2")
    ax.set_title("Transactions vs rent join-key coverage")
    ax.set_xlabel("Unique-key overlap coverage (%)")
    ax.set_ylabel("")
    save_plot(fig, plot_dir / "transactions_rent_join_overlap.png")


def write_summary_report(
    output_dir: Path,
    overview_df: pd.DataFrame,
    transaction_profile: pd.DataFrame,
    rent_profile: pd.DataFrame,
    hotel_profile: pd.DataFrame,
    transaction_outliers: pd.DataFrame,
    rent_outliers: pd.DataFrame,
    transaction_corr: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> None:
    transaction_missing = transaction_profile.loc[transaction_profile["missing_pct"] > 20, ["column", "missing_pct"]].head(8)
    rent_missing = rent_profile.loc[rent_profile["missing_pct"] > 10, ["column", "missing_pct"]].head(8)
    hotel_missing = hotel_profile.loc[hotel_profile["missing_pct"] > 0, ["column", "missing_pct"]].head(8)

    overview_lines = overview_df.to_markdown(index=False)
    transaction_missing_lines = transaction_missing.to_markdown(index=False) if not transaction_missing.empty else "None"
    rent_missing_lines = rent_missing.to_markdown(index=False) if not rent_missing.empty else "None"
    hotel_missing_lines = hotel_missing.to_markdown(index=False) if not hotel_missing.empty else "None"
    outlier_lines = transaction_outliers.to_markdown(index=False) if not transaction_outliers.empty else "None"
    rent_outlier_lines = rent_outliers.to_markdown(index=False) if not rent_outliers.empty else "None"
    corr_lines = transaction_corr.to_markdown(index=False) if not transaction_corr.empty else "None"
    overlap_lines = overlap_df.to_markdown(index=False) if not overlap_df.empty else "None"

    report_lines = [
        "# Exploratory Data Analysis Summary",
        "",
        "This EDA is aligned to the course project's `EDA & Data Understanding` requirement: summary statistics, feature distributions, correlation analysis, missingness, outliers, and dataset-integration feasibility for the proposed regression pipeline.",
        "",
        "## Dataset Overview",
        overview_lines,
        "",
        "## Key Findings",
        "1. The transaction table is the correct master table for supervised learning because it contains the regression target `actual_worth` and broad feature coverage, but it is strongly right-skewed with large-value outliers.",
        "2. The rent dataset is useful for area-time enrichment, but project-level fields are highly sparse, so future joins should rely primarily on `area_id`, `area_name_en`, and carefully engineered time windows rather than raw project names alone.",
        "3. The hotel dataset is annual UAE-level macro context, not neighborhood-level context. It should be merged only at the year level and treated as coarse exogenous signal rather than a direct geographic join source.",
        "4. Several columns are leakage risks for the later regression task. In particular, `meter_sale_price` is almost perfectly correlated with `actual_worth`, and `rent_value` is sparsely populated inside the transactions table but also mechanically related to price for the limited populated rows.",
        "5. The datasets contain substantial skewness and outliers, so the later pipeline should compare raw versus log-transformed targets and use robust error diagnostics.",
        "",
        "## Transactions: High Missingness",
        transaction_missing_lines,
        "",
        "## Rent Contracts: High Missingness",
        rent_missing_lines,
        "",
        "## Hotel Stats: Missingness Notes",
        hotel_missing_lines,
        "",
        "## Transactions: Correlation Snapshot",
        corr_lines,
        "",
        "## Transactions: IQR Outlier Snapshot",
        outlier_lines,
        "",
        "## Rent Contracts: IQR Outlier Snapshot",
        rent_outlier_lines,
        "",
        "## Join Feasibility Between Transactions and Rent Contracts",
        overlap_lines,
        "",
        "## Modeling Implications For The Next Stage",
        "1. Drop or quarantine leakage-prone columns before modeling, especially `meter_sale_price`, and scrutinize any variable derived from sale value.",
        "2. Use stratified reporting by year, area, property type, and registration type during preprocessing and model evaluation because the distributions are not stable across time or category.",
        "3. Engineer leak-free rental aggregates by area and past time window, not by unrestricted full-sample averages.",
        "4. Apply robust preprocessing for missing values, rare categories, and extreme numeric tails; linear models and tree models should be compared under the same train/validation/test protocol.",
    ]

    (output_dir / "eda_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis for the ML7501 real-estate project.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the input data files expected by the EDA pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where EDA outputs will be written.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    table_dir, plot_dir = ensure_output_dirs(output_dir)
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    sns.set_theme(style="whitegrid", context="talk")

    dataset_configs = make_dataset_configs(args.data_dir.resolve())
    loaded = {config.name: load_dataset(config) for config in dataset_configs}

    overview_rows = []
    profiles: dict[str, pd.DataFrame] = {}
    for config in dataset_configs:
        df = loaded[config.name]
        overview_rows.append(dataset_overview(config.name, df, config.date_columns))

        profile = column_profile(df)
        numeric = numeric_summary(df)
        profiles[config.name] = profile
        save_dataframe(profile, table_dir / f"{config.name}_column_profile.csv")
        save_dataframe(numeric, table_dir / f"{config.name}_numeric_summary.csv")
        plot_missingness(df, config.name, plot_dir)

    transactions = loaded["transactions"]
    rents = loaded["rent_contracts"]
    hotel = loaded["hotel_stats"]

    transaction_outliers = iqr_outlier_summary(
        transactions,
        ["actual_worth", "procedure_area", "meter_sale_price", "rent_value"],
    )
    rent_outliers = iqr_outlier_summary(rents, ["annual_amount", "actual_area", "contract_amount"])
    transaction_corr = top_correlations(transactions, "actual_worth")
    overlap_df = join_key_overlap(transactions, rents)

    save_dataframe(pd.DataFrame(overview_rows), table_dir / "dataset_overview.csv")
    save_dataframe(transaction_outliers, table_dir / "transactions_outlier_summary.csv")
    save_dataframe(rent_outliers, table_dir / "rent_contracts_outlier_summary.csv")
    save_dataframe(transaction_corr, table_dir / "transactions_top_correlations.csv")
    save_dataframe(overlap_df, table_dir / "transactions_rent_join_overlap.csv")

    plot_distribution(transactions, "actual_worth", "transactions", plot_dir)
    plot_distribution(transactions, "procedure_area", "transactions", plot_dir)
    plot_distribution(rents, "annual_amount", "rent_contracts", plot_dir)
    plot_distribution(rents, "actual_area", "rent_contracts", plot_dir)

    plot_top_categories(transactions, "procedure_name_en", "transactions", plot_dir)
    plot_top_categories(transactions, "property_type_en", "transactions", plot_dir)
    plot_top_categories(transactions, "property_usage_en", "transactions", plot_dir)
    plot_top_categories(rents, "ejari_property_type_en", "rent_contracts", plot_dir)
    plot_top_categories(rents, "contract_reg_type_en", "rent_contracts", plot_dir)
    plot_top_categories(rents, "tenant_type_en", "rent_contracts", plot_dir)

    plot_value_by_category(transactions, "actual_worth", "property_type_en", "transactions", plot_dir)
    plot_value_by_category(transactions, "actual_worth", "area_name_en", "transactions", plot_dir)
    plot_value_by_category(rents, "annual_amount", "ejari_property_type_en", "rent_contracts", plot_dir)
    plot_value_by_category(rents, "annual_amount", "area_name_en", "rent_contracts", plot_dir)

    plot_time_trend(transactions, "instance_date", "actual_worth", "transactions", plot_dir)
    plot_time_trend(rents, "contract_start_date", "annual_amount", "rent_contracts", plot_dir)
    plot_correlation_heatmap(transactions, "transactions", "actual_worth", plot_dir)
    plot_correlation_heatmap(rents, "rent_contracts", "annual_amount", plot_dir)
    plot_outlier_boxplots(transactions, ["actual_worth", "procedure_area", "meter_sale_price"], "transactions", plot_dir)
    plot_outlier_boxplots(rents, ["annual_amount", "actual_area", "contract_amount"], "rent_contracts", plot_dir)
    plot_hotel_trends(hotel, plot_dir)
    plot_join_overlap(overlap_df, plot_dir)

    write_summary_report(
        output_dir=output_dir,
        overview_df=pd.DataFrame(overview_rows),
        transaction_profile=profiles["transactions"],
        rent_profile=profiles["rent_contracts"],
        hotel_profile=profiles["hotel_stats"],
        transaction_outliers=transaction_outliers,
        rent_outliers=rent_outliers,
        transaction_corr=transaction_corr,
        overlap_df=overlap_df,
    )


if __name__ == "__main__":
    run()
