import numpy as np
import pandas as pd
import logging
from src.models.lgd import LGDScenarios

logger = logging.getLogger(__name__)

RECONCILIATION_TOLERANCE = 1e-10


class ECLCalculator:
    """
    Loan-level ECL calculator implementing the IFRS 9 discounted cash flow approach.

    For each loan and each month t:
        marginal_pd_t  = (1 - h)^(t-1) * h          where h = monthly hazard
        projected_EAD_t = amortized balance at month t
        DF_t           = (1 + r)^(-(t-0.5)/12)       mid-month convention
        ECL_t          = marginal_pd_t * EAD_t / DF_t * LGD

    12-month ECL  = sum of months 1–12
    Lifetime ECL  = sum of all months to remaining maturity

    Post-calculation checks (all raising ValueError if failed):
        - 12m ECL <= Lifetime ECL per loan
        - LGD scenario ordering: optimistic <= base <= adverse
        - Sum of marginal PDs reconciles to lifetime PD (atol=1e-10)
    """

    def __init__(self, lgd_scenarios: LGDScenarios = None):
        self.lgd = lgd_scenarios or LGDScenarios()

    def calculate(self, portfolio: pd.DataFrame) -> dict:
        """
        Returns:
            loan_df    : loan-level ECL results
            annual_df  : annual breakdown
            summary_df : portfolio-level scenario summary
        """
        self._validate(portfolio)
        portfolio = portfolio.copy()
        portfolio["mths_remng"] = portfolio["mths_remng"].astype(int)
        portfolio["discount_rate"] = portfolio["orig_int_rt"] / 100

        loan_results = []
        annual_results = []

        for row in portfolio.itertuples(index=False):
            loan_res, annual_res = self._calculate_loan(row)
            loan_results.append(loan_res)
            annual_results.extend(annual_res)

        loan_df = pd.DataFrame(loan_results)
        annual_df = pd.DataFrame(annual_results)

        self._run_checks(loan_df)

        summary_df = self._portfolio_summary(loan_df, portfolio["ead"].sum())

        logger.info(
            "ECL calculated for %d loans. Base scenario — 12m: %.2fM, Lifetime: %.2fM",
            len(loan_df),
            loan_df["ecl_12m_base"].sum() / 1e6,
            loan_df["ecl_lifetime_base"].sum() / 1e6,
        )
        return {"loan": loan_df, "annual": annual_df, "summary": summary_df}

    def _calculate_loan(self, row):
        loan_id = row.id_loan
        initial_ead = float(row.ead)
        pd_12m = float(row.pd_12m)
        maturity = int(row.mths_remng)
        annual_rate = float(row.discount_rate)

        monthly_hazard = 1 - (1 - pd_12m) ** (1 / 12)
        months = np.arange(1, maturity + 1)
        survival = (1 - monthly_hazard) ** (months - 1)
        marginal_pd = survival * monthly_hazard

        monthly_rate = annual_rate / 12
        if abs(monthly_rate) < 1e-12:
            payment = initial_ead / maturity
            projected_ead = initial_ead - payment * (months - 1)
        else:
            payment = initial_ead * monthly_rate / (1 - (1 + monthly_rate) ** (-maturity))
            projected_ead = (
                initial_ead * (1 + monthly_rate) ** (months - 1)
                - payment * ((1 + monthly_rate) ** (months - 1) - 1) / monthly_rate
            )
        projected_ead = np.maximum(projected_ead, 0)

        discount_factor = (1 + annual_rate) ** ((months - 0.5) / 12)
        discounted_exposure = marginal_pd * projected_ead / discount_factor

        first_12 = months <= 12
        lifetime_pd = 1 - (1 - pd_12m) ** (maturity / 12)

        result = {
            "id_loan": loan_id,
            "initial_ead": initial_ead,
            "pd_12m": pd_12m,
            "lifetime_pd": lifetime_pd,
            "remaining_maturity_months": maturity,
            "original_interest_rate": row.orig_int_rt,
            "monthly_hazard": monthly_hazard,
            "lifetime_marginal_pd_sum": marginal_pd.sum(),
        }

        for scenario, lgd in self.lgd:
            result[f"ecl_12m_{scenario}"] = discounted_exposure[first_12].sum() * lgd
            result[f"ecl_lifetime_{scenario}"] = discounted_exposure.sum() * lgd

        years = ((months - 1) // 12) + 1
        annual = []
        for year in np.unique(years):
            mask = years == year
            annual_row = {
                "id_loan": loan_id,
                "year": int(year),
                "marginal_pd": marginal_pd[mask].sum(),
                "average_projected_ead": projected_ead[mask].mean(),
            }
            for scenario, lgd in self.lgd:
                annual_row[f"ecl_{scenario}"] = discounted_exposure[mask].sum() * lgd
            annual.append(annual_row)

        return result, annual

    def _run_checks(self, loan_df: pd.DataFrame):
        scenarios = [name for name, _ in self.lgd]

        if not (loan_df[f"ecl_12m_{scenarios[0]}"] <= loan_df[f"ecl_lifetime_{scenarios[0]}"] + 1e-8).all():
            raise ValueError("12-month ECL exceeds lifetime ECL for at least one loan.")

        for i in range(len(scenarios) - 1):
            s1, s2 = scenarios[i], scenarios[i + 1]
            if not (loan_df[f"ecl_lifetime_{s1}"] <= loan_df[f"ecl_lifetime_{s2}"]).all():
                raise ValueError(f"LGD scenario ordering violated: {s1} vs {s2}.")

        diff = (loan_df["lifetime_marginal_pd_sum"] - loan_df["lifetime_pd"]).abs().max()
        if not np.isclose(diff, 0, atol=RECONCILIATION_TOLERANCE):
            raise ValueError(f"Marginal PD reconciliation failed. Max diff: {diff:.2e}")

        logger.info("All ECL validation checks passed.")

    def _portfolio_summary(self, loan_df: pd.DataFrame, total_ead: float) -> pd.DataFrame:
        rows = []
        for scenario, lgd in self.lgd:
            ecl_12m = loan_df[f"ecl_12m_{scenario}"].sum()
            ecl_lifetime = loan_df[f"ecl_lifetime_{scenario}"].sum()
            rows.append({
                "Scenario": scenario,
                "LGD": lgd,
                "Portfolio_EAD": total_ead,
                "ECL_12m": ecl_12m,
                "ECL_Lifetime": ecl_lifetime,
                "ECL_12m_pct_EAD": ecl_12m / total_ead,
                "ECL_Lifetime_pct_EAD": ecl_lifetime / total_ead,
            })
        return pd.DataFrame(rows)

    def _validate(self, portfolio: pd.DataFrame):
        required = ["id_loan", "ead", "pd_12m", "mths_remng", "orig_int_rt"]
        missing = [c for c in required if c not in portfolio.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if portfolio["id_loan"].duplicated().any():
            raise ValueError("Duplicate loan IDs.")
        if portfolio[required].isna().any().any():
            raise ValueError("Missing values in required ECL inputs.")
        if not portfolio["pd_12m"].between(0, 1).all():
            raise ValueError("PD must be between 0 and 1.")
        if (portfolio["ead"] <= 0).any():
            raise ValueError("EAD must be positive.")
        if (portfolio["mths_remng"] <= 0).any():
            raise ValueError("Remaining maturity must be positive.")
