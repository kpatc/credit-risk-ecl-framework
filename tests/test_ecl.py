import numpy as np
import pandas as pd
import pytest
from src.ecl.calculator import ECLCalculator
from src.models.lgd import LGDScenarios


@pytest.fixture
def sample_portfolio():
    return pd.DataFrame({
        "id_loan": ["L001", "L002", "L003"],
        "ead": [200_000.0, 150_000.0, 300_000.0],
        "pd_12m": [0.03, 0.08, 0.015],
        "mths_remng": [120, 240, 60],
        "orig_int_rt": [4.5, 3.75, 5.0],
    })


def test_ecl_keys(sample_portfolio):
    calc = ECLCalculator()
    results = calc.calculate(sample_portfolio)
    assert "loan" in results
    assert "annual" in results
    assert "summary" in results


def test_12m_leq_lifetime(sample_portfolio):
    calc = ECLCalculator()
    loan_df = calc.calculate(sample_portfolio)["loan"]
    assert (loan_df["ecl_12m_base"] <= loan_df["ecl_lifetime_base"] + 1e-8).all()


def test_scenario_ordering(sample_portfolio):
    calc = ECLCalculator()
    loan_df = calc.calculate(sample_portfolio)["loan"]
    assert (loan_df["ecl_lifetime_optimistic"] <= loan_df["ecl_lifetime_base"]).all()
    assert (loan_df["ecl_lifetime_base"] <= loan_df["ecl_lifetime_adverse"]).all()


def test_marginal_pd_reconciliation(sample_portfolio):
    calc = ECLCalculator()
    loan_df = calc.calculate(sample_portfolio)["loan"]
    diff = (loan_df["lifetime_marginal_pd_sum"] - loan_df["lifetime_pd"]).abs().max()
    assert diff < 1e-10


def test_summary_has_all_scenarios(sample_portfolio):
    lgd = LGDScenarios({"optimistic": 0.20, "base": 0.35, "adverse": 0.50})
    calc = ECLCalculator(lgd)
    summary = calc.calculate(sample_portfolio)["summary"]
    assert set(summary["Scenario"]) == {"optimistic", "base", "adverse"}


def test_duplicate_loan_id_raises(sample_portfolio):
    bad = pd.concat([sample_portfolio, sample_portfolio.iloc[[0]]], ignore_index=True)
    calc = ECLCalculator()
    with pytest.raises(ValueError, match="Duplicate loan IDs"):
        calc.calculate(bad)


def test_negative_ead_raises(sample_portfolio):
    bad = sample_portfolio.copy()
    bad.loc[0, "ead"] = -1000
    calc = ECLCalculator()
    with pytest.raises(ValueError, match="EAD must be positive"):
        calc.calculate(bad)
