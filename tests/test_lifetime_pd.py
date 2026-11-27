import pandas as pd
import pytest
from src.models.lifetime_pd import LifetimePDProjector


@pytest.fixture
def sample_portfolio():
    return pd.DataFrame({
        "id_loan": ["L001", "L002"],
        "ead": [200_000.0, 150_000.0],
        "pd_12m": [0.05, 0.02],
        "mths_remng": [120, 240],
    })


def test_lifetime_pd_gt_12m_pd(sample_portfolio):
    proj = LifetimePDProjector()
    result = proj.project(sample_portfolio)
    assert (result["lifetime_pd"] >= result["pd_12m"]).all()


def test_lifetime_pd_between_0_and_1(sample_portfolio):
    proj = LifetimePDProjector()
    result = proj.project(sample_portfolio)
    assert result["lifetime_pd"].between(0, 1).all()


def test_reconciliation_passes(sample_portfolio):
    proj = LifetimePDProjector()
    result = proj.project(sample_portfolio)
    curve = proj.build_annual_curve(result)
    marginal_sum = curve.groupby("id_loan")["marginal_pd"].sum()
    lifetime = result.set_index("id_loan")["lifetime_pd"]
    diff = (marginal_sum - lifetime).abs().max()
    assert diff < 1e-10


def test_missing_column_raises(sample_portfolio):
    proj = LifetimePDProjector()
    bad = sample_portfolio.drop(columns=["mths_remng"])
    with pytest.raises(ValueError, match="Missing columns"):
        proj.project(bad)
