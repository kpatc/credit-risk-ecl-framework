import numpy as np
import pandas as pd
import pytest
from src.features.woe import WOETransformer


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "score": rng.normal(700, 50, n),
        "ltv": rng.uniform(50, 100, n),
        "loan_type": rng.choice(["A", "B", "C"], n),
    })
    y = pd.Series(rng.binomial(1, 0.15, n))
    return X, y


def test_fit_transform_shape(sample_data):
    X, y = sample_data
    woe = WOETransformer(
        numerical_features=["score", "ltv"],
        categorical_features=["loan_type"],
    )
    X_out = woe.fit_transform(X, y)
    assert X_out.shape == X.shape


def test_woe_maps_fitted(sample_data):
    X, y = sample_data
    woe = WOETransformer(
        numerical_features=["score", "ltv"],
        categorical_features=["loan_type"],
    )
    woe.fit(X, y)
    assert "score" in woe.woe_maps_
    assert "ltv" in woe.woe_maps_
    assert "loan_type" in woe.woe_maps_


def test_unknown_category_gets_zero(sample_data):
    X, y = sample_data
    woe = WOETransformer(
        numerical_features=[],
        categorical_features=["loan_type"],
    )
    woe.fit(X, y)
    X_new = pd.DataFrame({"loan_type": ["UNKNOWN_CAT"]})
    X_out = woe.transform(X_new)
    assert X_out["loan_type"].iloc[0] == 0.0


def test_iv_table_columns(sample_data):
    X, y = sample_data
    woe = WOETransformer(
        numerical_features=["score"],
        categorical_features=["loan_type"],
    )
    woe.fit(X, y)
    table = woe.iv_table()
    assert "IV" in table.columns
    assert "Interpretation" in table.columns


def test_no_leakage_between_fit_transform(sample_data):
    X, y = sample_data
    split = 400
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train = y.iloc[:split]

    woe = WOETransformer(
        numerical_features=["score", "ltv"],
        categorical_features=["loan_type"],
    )
    woe.fit(X_train, y_train)

    X_train_out = woe.transform(X_train)
    X_test_out = woe.transform(X_test)

    assert X_train_out.shape[1] == X_train.shape[1]
    assert X_test_out.shape[1] == X_test.shape[1]
