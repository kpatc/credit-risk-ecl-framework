import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence transformer for binary classification.

    Fits WOE mappings on training data only. Numerical features are
    binned into quantiles; categorical features are mapped directly.
    Unknown categories at transform time receive a neutral WOE of 0.
    Laplace smoothing prevents infinite WOE when a bin contains only
    defaults or only non-defaults.
    """

    def __init__(self, numerical_features, categorical_features, max_bins=5, smoothing=0.5):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.max_bins = max_bins
        self.smoothing = smoothing

    def fit(self, X, y):
        self.bin_edges_ = {}
        self.woe_maps_ = {}
        self.iv_summary_ = {}
        self._medians_ = {}

        y = pd.Series(y).reset_index(drop=True)
        X = pd.DataFrame(X).reset_index(drop=True)

        for col in self.numerical_features:
            values = pd.to_numeric(X[col], errors="coerce")
            median = values.median()
            self._medians_[col] = median
            values = values.fillna(median)

            _, edges = pd.qcut(
                values, q=self.max_bins, retbins=True, duplicates="drop"
            )
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_[col] = edges

            bins = pd.cut(values, bins=edges, include_lowest=True)
            woe_map, iv = self._calculate_woe(bins, y)
            self.woe_maps_[col] = woe_map
            self.iv_summary_[col] = iv

        for col in self.categorical_features:
            bins = X[col].fillna("MISSING").astype(str)
            woe_map, iv = self._calculate_woe(bins, y)
            self.woe_maps_[col] = woe_map
            self.iv_summary_[col] = iv

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.numerical_features:
            values = pd.to_numeric(X[col], errors="coerce").fillna(self._medians_[col])
            bins = pd.cut(values, bins=self.bin_edges_[col], include_lowest=True)
            X[col] = bins.map(self.woe_maps_[col]).fillna(0).astype(float)

        for col in self.categorical_features:
            bins = X[col].fillna("MISSING").astype(str)
            X[col] = bins.map(self.woe_maps_[col]).fillna(0).astype(float)

        return X

    def iv_table(self):
        return (
            pd.DataFrame.from_dict(
                self.iv_summary_, orient="index", columns=["IV"]
            )
            .sort_values("IV", ascending=False)
            .assign(Interpretation=lambda df: df["IV"].map(self._interpret_iv))
        )

    @staticmethod
    def _interpret_iv(iv):
        if iv < 0.02:
            return "Not predictive"
        elif iv < 0.10:
            return "Weak"
        elif iv < 0.30:
            return "Medium"
        elif iv < 0.50:
            return "Strong"
        return "Suspicious"

    def _calculate_woe(self, bins, y):
        temp = pd.DataFrame({"bin": bins, "target": y})
        grouped = temp.groupby("bin", observed=True)["target"].agg(["count", "sum"])
        grouped["non_default"] = grouped["count"] - grouped["sum"]

        total_bad = grouped["sum"].sum()
        total_good = grouped["non_default"].sum()
        n = len(grouped)
        alpha = self.smoothing

        grouped["bad_dist"] = (grouped["sum"] + alpha) / (total_bad + alpha * n)
        grouped["good_dist"] = (grouped["non_default"] + alpha) / (total_good + alpha * n)
        grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
        grouped["iv"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

        return grouped["woe"].to_dict(), grouped["iv"].sum()
