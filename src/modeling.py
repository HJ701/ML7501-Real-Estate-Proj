from __future__ import annotations

import argparse
import json
import os
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.utils.validation import check_is_fitted


warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

src_package = sys.modules.setdefault("src", types.ModuleType("src"))
setattr(src_package, "modeling", sys.modules[__name__])
sys.modules["src.modeling"] = sys.modules[__name__]

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "modeling" / "latest"
RANDOM_STATE = 42
DEFAULT_N_JOBS = 1
NULL_STRINGS = ("", " ", "null", "NULL", "None", "none", "NaN", "nan")


TRANSACTIONS_NUMERIC_COLUMNS = [
    "actual_worth",
    "reg_type_id",
    "procedure_id",
    "property_sub_type_id",
    "procedure_area",
    "area_id",
    "property_type_id",
    "has_parking",
    "meter_sale_price",
    "no_of_parties_role_3",
    "no_of_parties_role_2",
    "no_of_parties_role_1",
    "trans_group_id",
    "project_number",
    "rent_value",
    "meter_rent_price",
]

RENT_NUMERIC_COLUMNS = [
    "area_id",
    "ejari_bus_property_type_id",
    "no_of_prop",
    "line_number",
    "ejari_property_sub_type_id",
    "tenant_type_id",
    "annual_amount",
    "is_free_hold",
    "project_number",
    "actual_area",
    "contract_reg_type_id",
    "ejari_property_type_id",
    "contract_amount",
]

HOTEL_NUMERIC_COLUMNS = [
    "TIME_PERIOD",
    "OBS_VALUE",
    "DECIMALS",
    "UNIT_MULT",
]

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

LOCATION_FEATURE_COLUMNS = {
    "area_name_en",
    "project_name_en",
    "building_name_en",
    "master_project_en",
    "nearest_metro_en",
    "nearest_mall_en",
    "nearest_landmark_en",
}

LOCATION_FEATURE_PREFIXES = (
    "area_",
    "nearest_",
    "project_",
    "building_",
    "master_project_",
)

TIME_FEATURE_COLUMNS = {
    "transaction_year",
    "transaction_quarter",
    "transaction_month_number",
    "transaction_day_of_week",
    "days_since_2010",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_distributions: dict[str, list[object]]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric tails using training-set quantiles to stabilize models."""

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

    def get_feature_names_out(self, input_features: list[str] | np.ndarray | None = None) -> np.ndarray:
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)


QuantileClipper.__module__ = "src.modeling"


def slugify(text: str) -> str:
    chars = []
    for ch in text.lower():
        chars.append(ch if ch.isalnum() else "_")
    return "".join(chars).strip("_")


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    tables = output_dir / "tables"
    plots = output_dir / "plots"
    models = output_dir / "models"
    summaries = output_dir / "summaries"
    for directory in (output_dir, tables, plots, models, summaries):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_dir,
        "tables": tables,
        "plots": plots,
        "models": models,
        "summaries": summaries,
    }


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)
    return cleaned.replace({value: np.nan for value in NULL_STRINGS})


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def load_transactions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "Real-estate_Transactions_2026-03-27.csv"
    df = pd.read_csv(path, low_memory=False)
    df = normalize_missing_values(df)
    df = coerce_numeric(df, TRANSACTIONS_NUMERIC_COLUMNS)
    df["instance_date"] = pd.to_datetime(df["instance_date"], errors="coerce")
    df["load_timestamp"] = pd.to_datetime(df["load_timestamp"], errors="coerce")
    df = df.loc[df["actual_worth"].gt(0) & df["instance_date"].notna()].copy()

    df["area_id"] = pd.to_numeric(df["area_id"], errors="coerce").astype("Int64")
    df["transaction_year"] = df["instance_date"].dt.year.astype("Int64")
    df["transaction_quarter"] = df["instance_date"].dt.quarter.astype("Int64")
    df["transaction_month_number"] = df["instance_date"].dt.month.astype("Int64")
    df["transaction_day_of_week"] = df["instance_date"].dt.dayofweek.astype("Int64")
    df["transaction_month"] = df["instance_date"].dt.to_period("M").dt.to_timestamp()
    df["days_since_2010"] = (df["instance_date"] - pd.Timestamp("2010-01-01")).dt.days
    return df.sort_values("instance_date").reset_index(drop=True)


def load_rent_contracts(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "rent_contracts.csv"
    df = pd.read_csv(path, low_memory=False)
    df = normalize_missing_values(df)
    df = coerce_numeric(df, RENT_NUMERIC_COLUMNS)
    for column in ("contract_start_date", "contract_end_date", "load_timestamp"):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["area_id"] = pd.to_numeric(df["area_id"], errors="coerce").astype("Int64")
    return df


def load_hotel_stats(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "FCSA,DF_HOT_TYPE,4.3.0+...A.....csv"
    df = pd.read_csv(path, low_memory=False)
    df = normalize_missing_values(df)
    df = coerce_numeric(df, HOTEL_NUMERIC_COLUMNS)
    return df


def build_rent_features(rent_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    rent = rent_df.copy()
    rent = rent.loc[rent["contract_start_date"].notna() & rent["area_id"].notna()].copy()
    rent["rent_month"] = rent["contract_start_date"].dt.to_period("M").dt.to_timestamp()
    rent["rent_price_per_area"] = np.where(
        rent["actual_area"].gt(0),
        rent["annual_amount"] / rent["actual_area"],
        np.nan,
    )

    monthly = (
        rent.groupby(["area_id", "rent_month"], dropna=False)
        .agg(
            rent_contract_count=("contract_id", "count"),
            rent_annual_amount_sum=("annual_amount", "sum"),
            rent_actual_area_sum=("actual_area", "sum"),
            rent_monthly_amount_mean=("annual_amount", "mean"),
            rent_monthly_area_mean=("actual_area", "mean"),
            rent_monthly_price_per_area_mean=("rent_price_per_area", "mean"),
        )
        .reset_index()
    )

    all_areas = (
        pd.Index(pd.concat([transactions_df["area_id"], rent["area_id"]]).dropna().astype("Int64").unique())
        .sort_values()
    )
    min_month = rent["rent_month"].min()
    max_month = transactions_df["transaction_month"].max()
    all_months = pd.date_range(min_month, max_month, freq="MS")
    grid = pd.MultiIndex.from_product([all_areas, all_months], names=["area_id", "transaction_month"])

    panel = (
        monthly.rename(columns={"rent_month": "transaction_month"})
        .set_index(["area_id", "transaction_month"])
        .reindex(grid)
        .reset_index()
    )

    for column in ("rent_contract_count", "rent_annual_amount_sum", "rent_actual_area_sum"):
        panel[column] = panel[column].fillna(0.0)

    grouped = panel.groupby("area_id", group_keys=False)
    panel["rent_hist_contract_count"] = grouped["rent_contract_count"].transform(lambda s: s.cumsum().shift(1))
    panel["rent_hist_annual_amount_sum"] = grouped["rent_annual_amount_sum"].transform(lambda s: s.cumsum().shift(1))
    panel["rent_hist_actual_area_sum"] = grouped["rent_actual_area_sum"].transform(lambda s: s.cumsum().shift(1))
    panel["rent_hist_annual_amount_mean"] = (
        panel["rent_hist_annual_amount_sum"] / panel["rent_hist_contract_count"].replace(0, np.nan)
    )
    panel["rent_hist_actual_area_mean"] = (
        panel["rent_hist_actual_area_sum"] / panel["rent_hist_contract_count"].replace(0, np.nan)
    )
    panel["rent_hist_price_per_area_mean"] = (
        panel["rent_hist_annual_amount_sum"] / panel["rent_hist_actual_area_sum"].replace(0, np.nan)
    )

    panel["rent_last_3m_amount_mean"] = grouped["rent_monthly_amount_mean"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    panel["rent_last_6m_amount_mean"] = grouped["rent_monthly_amount_mean"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).mean()
    )
    panel["rent_last_12m_amount_mean"] = grouped["rent_monthly_amount_mean"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean()
    )
    panel["rent_last_6m_contract_count"] = grouped["rent_contract_count"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum()
    )
    panel["rent_last_12m_contract_count"] = grouped["rent_contract_count"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).sum()
    )
    panel["rent_last_6m_price_per_area_mean"] = grouped["rent_monthly_price_per_area_mean"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).mean()
    )
    panel["rent_last_12m_price_per_area_mean"] = grouped["rent_monthly_price_per_area_mean"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean()
    )
    panel["rent_feature_available"] = panel["rent_hist_contract_count"].fillna(0).gt(0).astype(int)

    feature_columns = [
        "area_id",
        "transaction_month",
        "rent_feature_available",
        "rent_hist_contract_count",
        "rent_hist_annual_amount_mean",
        "rent_hist_actual_area_mean",
        "rent_hist_price_per_area_mean",
        "rent_last_3m_amount_mean",
        "rent_last_6m_amount_mean",
        "rent_last_12m_amount_mean",
        "rent_last_6m_contract_count",
        "rent_last_12m_contract_count",
        "rent_last_6m_price_per_area_mean",
        "rent_last_12m_price_per_area_mean",
    ]
    return panel[feature_columns]


def build_hotel_features(hotel_df: pd.DataFrame, min_year: int, max_year: int) -> pd.DataFrame:
    hotel = hotel_df.copy()
    hotel = hotel.loc[hotel["TIME_PERIOD"].notna() & hotel["OBS_VALUE"].notna()].copy()
    hotel["TIME_PERIOD"] = hotel["TIME_PERIOD"].astype(int)
    hotel["hotel_feature_name"] = (
        "hotel_"
        + hotel["H_INDICATOR"].fillna("unknown").astype(str).str.lower()
        + "_"
        + hotel["H_TYPE"].fillna("unknown").astype(str).str.lower()
    )
    hotel["hotel_feature_name"] = hotel["hotel_feature_name"].map(slugify)

    pivot = hotel.pivot_table(
        index="TIME_PERIOD",
        columns="hotel_feature_name",
        values="OBS_VALUE",
        aggfunc="sum",
    )

    indicator_totals = (
        hotel.assign(hotel_total_name="hotel_total_" + hotel["H_INDICATOR"].fillna("unknown").astype(str).str.lower())
        .pivot_table(index="TIME_PERIOD", columns="hotel_total_name", values="OBS_VALUE", aggfunc="sum")
    )

    features = pivot.join(indicator_totals, how="outer").sort_index()
    full_index = pd.Index(range(min_year, max_year + 1), name="transaction_year")
    features = features.reindex(full_index).sort_index().ffill()
    available_years = set(hotel["TIME_PERIOD"].unique().tolist())
    features["hotel_feature_available"] = features.index.isin(available_years).astype(int)
    return features.reset_index()


def build_master_table(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    transactions = load_transactions(data_dir=data_dir)
    rent = load_rent_contracts(data_dir=data_dir)
    hotel = load_hotel_stats(data_dir=data_dir)

    rent_features = build_rent_features(rent, transactions)
    hotel_features = build_hotel_features(
        hotel_df=hotel,
        min_year=int(transactions["transaction_year"].min()),
        max_year=int(transactions["transaction_year"].max()),
    )

    master = transactions.merge(rent_features, on=["area_id", "transaction_month"], how="left")
    master = master.merge(hotel_features, on="transaction_year", how="left")
    return master.sort_values("instance_date").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1.")

    ordered = df.sort_values("instance_date").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))

    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    return train, val, test


def summarize_temporal_slice(df: pd.DataFrame, prefix: str) -> dict[str, object]:
    if df.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_start": None,
            f"{prefix}_end": None,
        }
    return {
        f"{prefix}_rows": int(len(df)),
        f"{prefix}_start": str(df["instance_date"].min().date()),
        f"{prefix}_end": str(df["instance_date"].max().date()),
    }


def build_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    summary.update(summarize_temporal_slice(train_df, "train"))
    summary.update(summarize_temporal_slice(val_df, "validation"))
    summary.update(summarize_temporal_slice(test_df, "test"))
    return summary


def expanding_window_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int | None = None,
    test_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = df.sort_values("instance_date").reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < 3:
        return []

    if min_train_size is None:
        min_train_size = max(50, int(n_rows * 0.50))
    min_train_size = min(max(1, min_train_size), n_rows - 2)

    remaining = n_rows - min_train_size
    if remaining <= 0:
        return []

    if test_size is None:
        test_size = max(25, int(n_rows * 0.10))
    test_size = min(max(1, test_size), remaining)

    max_splits = max(1, remaining // test_size)
    effective_splits = min(max(1, n_splits), max_splits)
    first_test_start = n_rows - (effective_splits * test_size)

    if first_test_start < min_train_size:
        first_test_start = min_train_size
        effective_splits = max(1, (n_rows - first_test_start) // test_size)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(effective_splits):
        test_start = first_test_start + (fold_id * test_size)
        test_end = min(test_start + test_size, n_rows)
        if test_end <= test_start or test_start <= 0:
            continue
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def describe_expanding_window_splits(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    ordered = df.sort_values("instance_date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = ordered.iloc[train_idx]
        test_df = ordered.iloc[test_idx]
        rows.append(
            {
                "fold": fold_number,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_start": str(train_df["instance_date"].min().date()),
                "train_end": str(train_df["instance_date"].max().date()),
                "test_start": str(test_df["instance_date"].min().date()),
                "test_end": str(test_df["instance_date"].max().date()),
            }
        )
    return pd.DataFrame(rows)


def build_feature_family_map(columns: list[str] | pd.Index) -> dict[str, list[str]]:
    ordered_columns = list(columns)
    families = {
        "rent": [column for column in ordered_columns if column.startswith("rent_")],
        "hotel": [column for column in ordered_columns if column.startswith("hotel_")],
        "location": [
            column
            for column in ordered_columns
            if column in LOCATION_FEATURE_COLUMNS or column.startswith(LOCATION_FEATURE_PREFIXES)
        ],
        "time": [column for column in ordered_columns if column in TIME_FEATURE_COLUMNS],
    }

    assigned = set().union(*families.values())
    families["structural"] = [column for column in ordered_columns if column not in assigned]
    return families


def build_feature_variants(columns: list[str] | pd.Index) -> dict[str, list[str]]:
    families = build_feature_family_map(columns)
    structural_only = families["structural"] + families["time"]
    location_only = families["location"] + families["time"]
    rental_enriched = structural_only + families["location"] + families["rent"]
    full_feature_set = list(columns)
    return {
        "structural_only": structural_only,
        "location_only": location_only,
        "rental_enriched": rental_enriched,
        "full_feature_set": full_feature_set,
    }


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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clipper", QuantileClipper(lower=0.01, upper=0.99)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=15,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_regression_pipeline(preprocessor: ColumnTransformer, estimator: object) -> Pipeline:
    from sklearn.compose import TransformedTargetRegressor

    return TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def make_classification_pipeline(preprocessor: ColumnTransformer, estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )


def maybe_add_xgboost_models(
    task: str,
    use_gpu: bool,
    random_state: int,
) -> list[ModelSpec]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        return []

    device = "cuda" if use_gpu else "cpu"
    tree_method = "hist"
    if task == "regression":
        return [
            ModelSpec(
                name="xgboost",
                estimator=XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    tree_method=tree_method,
                    device=device,
                    random_state=random_state,
                ),
                param_distributions={
                    "n_estimators": [250, 400, 550],
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.08],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.9, 1.0],
                    "reg_lambda": [0.5, 1.0, 2.0],
                },
            )
        ]

    return [
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                objective="binary:logistic",
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                tree_method=tree_method,
                device=device,
                eval_metric="logloss",
                random_state=random_state,
            ),
            param_distributions={
                "n_estimators": [250, 400, 550],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0],
            },
        )
    ]


def get_regression_model_specs(use_gpu: bool, random_state: int, n_jobs: int) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="dummy_regressor",
            estimator=DummyRegressor(strategy="median"),
            param_distributions={},
        ),
        ModelSpec(
            name="ridge_regression",
            estimator=Ridge(alpha=1.0, random_state=random_state),
            param_distributions={
                "alpha": [0.1, 0.5, 1.0, 5.0, 10.0],
            },
        ),
        ModelSpec(
            name="svm_regression",
            estimator=LinearSVR(C=1.0, epsilon=0.1, dual="auto", random_state=random_state, max_iter=20000),
            param_distributions={
                "C": [0.1, 0.5, 1.0, 2.0],
                "epsilon": [0.01, 0.05, 0.1, 0.2],
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            param_distributions={
                "n_estimators": [250, 400, 550],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=350,
                max_depth=8,
                random_state=random_state,
            ),
            param_distributions={
                "learning_rate": [0.03, 0.05, 0.08],
                "max_iter": [250, 350, 500],
                "max_depth": [None, 6, 8, 10],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [10, 20, 40],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
    ]
    specs.extend(maybe_add_xgboost_models(task="regression", use_gpu=use_gpu, random_state=random_state))
    return specs


def get_classification_model_specs(use_gpu: bool, random_state: int, n_jobs: int) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="dummy_classifier",
            estimator=DummyClassifier(strategy="prior"),
            param_distributions={},
        ),
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            ),
            param_distributions={
                "C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
        ),
        ModelSpec(
            name="svm_classifier",
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                random_state=random_state,
                max_iter=20000,
            ),
            param_distributions={
                "C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            param_distributions={
                "n_estimators": [250, 400, 550],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=350,
                max_depth=8,
                random_state=random_state,
            ),
            param_distributions={
                "learning_rate": [0.03, 0.05, 0.08],
                "max_iter": [250, 350, 500],
                "max_depth": [None, 6, 8, 10],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [10, 20, 40],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
    ]
    specs.extend(maybe_add_xgboost_models(task="classification", use_gpu=use_gpu, random_state=random_state))
    return specs


def primary_metric(task: str) -> str:
    return "neg_root_mean_squared_error" if task == "regression" else "roc_auc"


def prefix_params(task: str, params: dict[str, list[object]]) -> dict[str, list[object]]:
    if task == "regression":
        return {f"regressor__model__{key}": value for key, value in params.items()}
    return {f"model__{key}": value for key, value in params.items()}


def build_pipeline(task: str, preprocessor: ColumnTransformer, estimator: object) -> Pipeline:
    if task == "regression":
        return make_regression_pipeline(preprocessor, estimator)
    return make_classification_pipeline(preprocessor, estimator)


def get_prediction_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


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


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def save_json(payload: dict[str, object], path: Path) -> None:
    serializable = json.loads(json.dumps(payload, default=str))
    path.write_text(json.dumps(serializable, indent=2))


def plot_metric_bars(
    df: pd.DataFrame,
    metric: str,
    task: str,
    output_path: Path,
) -> None:
    if df.empty or metric not in df.columns:
        return

    plot_df = df.sort_values(metric, ascending=(task == "regression"))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["model_name"], plot_df[metric], color="#1f6f8b")
    ax.set_title(f"{task.title()} model comparison ({metric})")
    ax.set_xlabel(metric.upper())
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regression_diagnostics(
    y_true: pd.Series,
    predictions: np.ndarray,
    output_prefix: Path,
) -> None:
    residuals = y_true - predictions

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, predictions, alpha=0.35, s=20, color="#0f4c5c")
    diagonal_min = min(y_true.min(), predictions.min())
    diagonal_max = max(y_true.max(), predictions.max())
    ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="#bc4749")
    ax.set_title("Actual vs predicted sale price")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_actual_vs_predicted.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color="#ef8354", edgecolor="white")
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_residuals.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_classification_diagnostics(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_confusion_matrix.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    if scores is None:
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#0f4c5c", label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    ax.set_title("ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_roc_curve.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_permutation_importance_table(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    random_state: int,
) -> pd.DataFrame:
    scoring = primary_metric(task)
    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        n_repeats=5,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importance_df


def plot_permutation_importance(importance_df: pd.DataFrame, output_path: Path, title: str) -> None:
    top = importance_df.head(15).sort_values("importance_mean", ascending=True)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["importance_mean"], color="#2a9d8f")
    ax.set_title(title)
    ax.set_xlabel("Permutation importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_baseline_validation(
    task: str,
    model_specs: list[ModelSpec],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in model_specs:
        model = build_pipeline(task=task, preprocessor=preprocessor, estimator=clone(spec.estimator))
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        row = {"model_name": spec.name}
        if task == "regression":
            row.update(evaluate_regression(y_val, predictions))
        else:
            scores = get_prediction_scores(model, X_val)
            row.update(evaluate_classification(y_val, predictions, scores))
        rows.append(row)
    return pd.DataFrame(rows)


def run_hyperparameter_tuning(
    task: str,
    model_specs: list[ModelSpec],
    preprocessor: ColumnTransformer,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    tune_iterations: int,
    cv_splits: int,
    random_state: int,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    tuned_models: dict[str, Pipeline] = {}
    tuning_rows: list[dict[str, object]] = []

    preferred_names = ["random_forest", "xgboost"] if any(spec.name == "xgboost" for spec in model_specs) else [
        "random_forest",
        "hist_gradient_boosting",
    ]

    for spec in model_specs:
        if spec.name not in preferred_names or not spec.param_distributions:
            continue

        search = RandomizedSearchCV(
            estimator=build_pipeline(task=task, preprocessor=preprocessor, estimator=clone(spec.estimator)),
            param_distributions=prefix_params(task, spec.param_distributions),
            n_iter=min(tune_iterations, max(1, len(spec.param_distributions) * 3)),
            scoring=primary_metric(task),
            cv=TimeSeriesSplit(n_splits=cv_splits),
            refit=True,
            random_state=random_state,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_dev, y_dev)
        tuned_models[spec.name] = search.best_estimator_
        tuning_rows.append(
            {
                "model_name": spec.name,
                "best_cv_score": float(search.best_score_),
                "best_params": json.dumps(search.best_params_),
            }
        )
    return tuned_models, pd.DataFrame(tuning_rows)


def run_final_test_evaluation(
    task: str,
    model_specs: list[ModelSpec],
    preprocessor: ColumnTransformer,
    tuned_models: dict[str, Pipeline],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, dict[str, np.ndarray | pd.Series | None]]]:
    rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    diagnostics_payload: dict[str, dict[str, np.ndarray | pd.Series | None]] = {}

    for spec in model_specs:
        if spec.name in tuned_models:
            model = tuned_models[spec.name]
        else:
            model = build_pipeline(task=task, preprocessor=preprocessor, estimator=clone(spec.estimator))
            model.fit(X_dev, y_dev)

        predictions = model.predict(X_test)
        scores = get_prediction_scores(model, X_test) if task == "classification" else None
        row = {"model_name": spec.name}
        if task == "regression":
            row.update(evaluate_regression(y_test, predictions))
        else:
            row.update(evaluate_classification(y_test, predictions, scores))

        rows.append(row)
        fitted_models[spec.name] = model
        diagnostics_payload[spec.name] = {
            "y_true": y_test,
            "predictions": predictions,
            "scores": scores,
        }
        dump(model, model_dir / f"{task}_{spec.name}.joblib")

    metrics_df = pd.DataFrame(rows)
    return metrics_df, fitted_models, diagnostics_payload


def choose_best_model(metrics_df: pd.DataFrame, task: str) -> str:
    if task == "regression":
        return metrics_df.sort_values("rmse", ascending=True).iloc[0]["model_name"]
    return metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model_name"]


def build_markdown_summary(
    output_path: Path,
    split_summary: dict[str, object],
    rolling_plan: pd.DataFrame,
    regression_metrics: pd.DataFrame | None,
    classification_metrics: pd.DataFrame | None,
    classification_threshold: float | None,
) -> None:
    lines = [
        "# Modeling Summary",
        "",
        "## Split Overview",
        f"- Train rows: {split_summary['train_rows']}",
        f"- Validation rows: {split_summary['validation_rows']}",
        f"- Test rows: {split_summary['test_rows']}",
        f"- Train date range: {split_summary['train_start']} to {split_summary['train_end']}",
        f"- Validation date range: {split_summary['validation_start']} to {split_summary['validation_end']}",
        f"- Test date range: {split_summary['test_start']} to {split_summary['test_end']}",
    ]

    if not rolling_plan.empty:
        lines.extend(
            [
                "",
                "## Rolling-Origin Backtest Plan",
                f"- Expanding-window folds: {len(rolling_plan)}",
                f"- First fold train/test window: {rolling_plan.iloc[0]['train_start']} to {rolling_plan.iloc[0]['train_end']} / {rolling_plan.iloc[0]['test_start']} to {rolling_plan.iloc[0]['test_end']}",
                f"- Last fold train/test window: {rolling_plan.iloc[-1]['train_start']} to {rolling_plan.iloc[-1]['train_end']} / {rolling_plan.iloc[-1]['test_start']} to {rolling_plan.iloc[-1]['test_end']}",
            ]
        )

    if classification_threshold is not None:
        lines.extend(
            [
                "",
                "## Classification Label",
                f"- `is_high_value = 1` when `actual_worth >= {classification_threshold:,.2f}`",
            ]
        )

    if regression_metrics is not None and not regression_metrics.empty:
        best = regression_metrics.sort_values("rmse", ascending=True).iloc[0]
        lines.extend(
            [
                "",
                "## Best Regression Test Result",
                f"- Model: {best['model_name']}",
                f"- RMSE: {best['rmse']:,.2f}",
                f"- MAE: {best['mae']:,.2f}",
                f"- R2: {best['r2']:.4f}",
            ]
        )

    if classification_metrics is not None and not classification_metrics.empty:
        best = classification_metrics.sort_values("roc_auc", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                "## Best Classification Test Result",
                f"- Model: {best['model_name']}",
                f"- ROC AUC: {best['roc_auc']:.4f}",
                f"- F1: {best['f1']:.4f}",
                f"- Recall: {best['recall']:.4f}",
            ]
        )

    output_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ML7501 Dubai property modeling pipeline.")
    parser.add_argument(
        "--task",
        choices=("regression", "classification", "both"),
        default="both",
        help="Which modeling track to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where modeling artifacts will be saved.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the input data files expected by the pipeline.",
    )
    parser.add_argument("--train-frac", type=float, default=0.70, help="Fraction of rows used for training.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of rows used for validation.")
    parser.add_argument(
        "--classification-quantile",
        type=float,
        default=0.75,
        help="Training-set quantile used to define the high-value classification label.",
    )
    parser.add_argument(
        "--tune-iterations",
        type=int,
        default=8,
        help="Number of random-search iterations per tuned model.",
    )
    parser.add_argument("--cv-splits", type=int, default=4, help="Number of time-series CV splits for tuning.")
    parser.add_argument(
        "--backtest-splits",
        type=int,
        default=5,
        help="Number of expanding-window folds to save for downstream robustness analysis.",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Parallel workers for estimators that support n_jobs. Defaults to 1 for portability.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU-backed XGBoost when xgboost is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = ensure_output_dirs(args.output_dir.resolve())

    master = build_master_table(data_dir=args.data_dir.resolve())
    save_dataframe(master, directories["tables"] / "modeling_master_table.csv")

    train_df, val_df, test_df = temporal_split(master, train_frac=args.train_frac, val_frac=args.val_frac)
    split_summary = build_split_summary(train_df, val_df, test_df)
    save_json(split_summary, directories["summaries"] / "split_summary.json")
    rolling_folds = expanding_window_splits(master, n_splits=args.backtest_splits)
    rolling_plan = describe_expanding_window_splits(master, rolling_folds)
    save_dataframe(rolling_plan, directories["tables"] / "rolling_origin_split_plan.csv")

    X_train = select_model_features(train_df)
    X_val = select_model_features(val_df)
    X_test = select_model_features(test_df)
    X_dev = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)

    preprocessor = build_preprocessor(X_train)

    regression_test_metrics: pd.DataFrame | None = None
    classification_test_metrics: pd.DataFrame | None = None
    classification_threshold: float | None = None

    if args.task in {"regression", "both"}:
        y_train_reg = train_df["actual_worth"].reset_index(drop=True)
        y_val_reg = val_df["actual_worth"].reset_index(drop=True)
        y_test_reg = test_df["actual_worth"].reset_index(drop=True)
        y_dev_reg = pd.concat([y_train_reg, y_val_reg], axis=0).reset_index(drop=True)

        regression_specs = get_regression_model_specs(
            use_gpu=args.use_gpu,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        regression_validation = run_baseline_validation(
            task="regression",
            model_specs=regression_specs,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train_reg,
            X_val=X_val,
            y_val=y_val_reg,
        )
        save_dataframe(regression_validation, directories["tables"] / "regression_validation_metrics.csv")

        regression_tuned, regression_tuning_table = run_hyperparameter_tuning(
            task="regression",
            model_specs=regression_specs,
            preprocessor=preprocessor,
            X_dev=X_dev,
            y_dev=y_dev_reg,
            tune_iterations=args.tune_iterations,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
        )
        save_dataframe(regression_tuning_table, directories["tables"] / "regression_tuning_results.csv")

        regression_test_metrics, fitted_regression_models, regression_payload = run_final_test_evaluation(
            task="regression",
            model_specs=regression_specs,
            preprocessor=preprocessor,
            tuned_models=regression_tuned,
            X_dev=X_dev,
            y_dev=y_dev_reg,
            X_test=X_test,
            y_test=y_test_reg,
            model_dir=directories["models"],
        )
        save_dataframe(regression_test_metrics, directories["tables"] / "regression_test_metrics.csv")
        plot_metric_bars(
            regression_test_metrics,
            metric="rmse",
            task="regression",
            output_path=directories["plots"] / "regression_model_comparison_rmse.png",
        )

        best_regression_name = choose_best_model(regression_test_metrics, task="regression")
        best_regression_model = fitted_regression_models[best_regression_name]
        best_regression_payload = regression_payload[best_regression_name]
        plot_regression_diagnostics(
            y_true=best_regression_payload["y_true"],
            predictions=best_regression_payload["predictions"],
            output_prefix=directories["plots"] / f"regression_best_{best_regression_name}",
        )
        regression_importance = compute_permutation_importance_table(
            model=best_regression_model,
            X=X_test,
            y=y_test_reg,
            task="regression",
            random_state=args.random_state,
        )
        save_dataframe(
            regression_importance,
            directories["tables"] / f"regression_permutation_importance_{best_regression_name}.csv",
        )
        plot_permutation_importance(
            regression_importance,
            output_path=directories["plots"] / f"regression_permutation_importance_{best_regression_name}.png",
            title=f"Regression permutation importance ({best_regression_name})",
        )

    if args.task in {"classification", "both"}:
        classification_threshold = float(train_df["actual_worth"].quantile(args.classification_quantile))
        y_train_clf = train_df["actual_worth"].ge(classification_threshold).astype(int).reset_index(drop=True)
        y_val_clf = val_df["actual_worth"].ge(classification_threshold).astype(int).reset_index(drop=True)
        y_test_clf = test_df["actual_worth"].ge(classification_threshold).astype(int).reset_index(drop=True)
        y_dev_clf = pd.concat([y_train_clf, y_val_clf], axis=0).reset_index(drop=True)

        classification_specs = get_classification_model_specs(
            use_gpu=args.use_gpu,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        classification_validation = run_baseline_validation(
            task="classification",
            model_specs=classification_specs,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train_clf,
            X_val=X_val,
            y_val=y_val_clf,
        )
        save_dataframe(classification_validation, directories["tables"] / "classification_validation_metrics.csv")

        classification_tuned, classification_tuning_table = run_hyperparameter_tuning(
            task="classification",
            model_specs=classification_specs,
            preprocessor=preprocessor,
            X_dev=X_dev,
            y_dev=y_dev_clf,
            tune_iterations=args.tune_iterations,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
        )
        save_dataframe(classification_tuning_table, directories["tables"] / "classification_tuning_results.csv")

        classification_test_metrics, fitted_classification_models, classification_payload = run_final_test_evaluation(
            task="classification",
            model_specs=classification_specs,
            preprocessor=preprocessor,
            tuned_models=classification_tuned,
            X_dev=X_dev,
            y_dev=y_dev_clf,
            X_test=X_test,
            y_test=y_test_clf,
            model_dir=directories["models"],
        )
        save_dataframe(classification_test_metrics, directories["tables"] / "classification_test_metrics.csv")
        plot_metric_bars(
            classification_test_metrics,
            metric="roc_auc",
            task="classification",
            output_path=directories["plots"] / "classification_model_comparison_roc_auc.png",
        )

        best_classification_name = choose_best_model(classification_test_metrics, task="classification")
        best_classification_model = fitted_classification_models[best_classification_name]
        best_classification_payload = classification_payload[best_classification_name]
        plot_classification_diagnostics(
            y_true=best_classification_payload["y_true"],
            predictions=best_classification_payload["predictions"],
            scores=best_classification_payload["scores"],
            output_prefix=directories["plots"] / f"classification_best_{best_classification_name}",
        )
        classification_importance = compute_permutation_importance_table(
            model=best_classification_model,
            X=X_test,
            y=y_test_clf,
            task="classification",
            random_state=args.random_state,
        )
        save_dataframe(
            classification_importance,
            directories["tables"] / f"classification_permutation_importance_{best_classification_name}.csv",
        )
        plot_permutation_importance(
            classification_importance,
            output_path=directories["plots"] / f"classification_permutation_importance_{best_classification_name}.png",
            title=f"Classification permutation importance ({best_classification_name})",
        )

    build_markdown_summary(
        output_path=directories["summaries"] / "modeling_summary.md",
        split_summary=split_summary,
        rolling_plan=rolling_plan,
        regression_metrics=regression_test_metrics,
        classification_metrics=classification_test_metrics,
        classification_threshold=classification_threshold,
    )

    print(f"Saved modeling artifacts to {directories['root']}")


if __name__ == "__main__":
    main()
