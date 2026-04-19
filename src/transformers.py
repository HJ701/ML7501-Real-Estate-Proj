from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ObservedColumnSelector(BaseEstimator, TransformerMixin):
    """Keep only columns with at least one observed value in the fit data."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ObservedColumnSelector":
        if hasattr(X, "isna"):
            observed_mask = ~X.isna().all(axis=0).to_numpy()
            feature_names = np.asarray(getattr(X, "columns", np.arange(X.shape[1])), dtype=object)
        else:
            values = np.asarray(X, dtype=object)
            observed_mask = ~np.all(pd.isna(values), axis=0)
            feature_names = np.arange(values.shape[1], dtype=object)

        self.observed_mask_ = observed_mask
        self.feature_names_in_ = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["observed_mask_", "feature_names_in_"])
        if hasattr(X, "iloc"):
            return X.iloc[:, self.observed_mask_]
        values = np.asarray(X)
        return values[:, self.observed_mask_]

    def get_feature_names_out(self, input_features: list[str] | np.ndarray | None = None) -> np.ndarray:
        check_is_fitted(self, ["observed_mask_", "feature_names_in_"])
        features = self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)
        return features[self.observed_mask_]


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
