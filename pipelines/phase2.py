"""
Phase II pipeline — Freddie Mac ECL framework.

Steps:
    1. Load origination + performance data
    2. Build 12-month default target
    3. Train PD model (GroupShuffleSplit by loan)
    4. Score September 2021 portfolio snapshot
    5. Estimate EAD
    6. Project Lifetime PD
    7. Calculate ECL under three LGD scenarios

Usage:
    python -m pipelines.phase2
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_freddie_mac_origination, load_freddie_mac_performance
from src.data.validators import assert_binary_target, assert_no_duplicate_ids, merge_check
from src.models.pd_model import Phase2PDModel
from src.models.ead import EADEstimator
from src.models.lgd import LGDScenarios
from src.models.lifetime_pd import LifetimePDProjector
from src.ecl.calculator import ECLCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_target(perf: pd.DataFrame, dpd_threshold: int = 3) -> pd.DataFrame:
    """
    Build 12-month forward-looking default target.
    Default = 90+ DPD (delq_numeric >= threshold) or REO status.
    """
    perf = perf.copy()
    perf["delq_sts"] = perf["delq_sts"].astype("string").str.strip()
    perf["delq_numeric"] = pd.to_numeric(perf["delq_sts"], errors="coerce")
    perf["period"] = pd.to_numeric(perf["period"], errors="coerce")
    perf = perf.dropna(subset=["id_loan", "period"]).copy()

    year = (perf["period"] // 100).astype(int)
    month = (perf["period"] % 100).astype(int)
    perf["month_index"] = year * 12 + month

    perf["default_flag"] = (
        (perf["delq_numeric"] >= dpd_threshold) | (perf["delq_sts"] == "RA")
    ).astype(int)

    perf = perf.sort_values(["id_loan", "month_index"])

    rows = []
    for loan_id, group in perf.groupby("id_loan"):
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            horizon = group[
                (group["month_index"] > row["month_index"]) &
                (group["month_index"] <= row["month_index"] + 12)
            ]
            target = int(horizon["default_flag"].any())
            rows.append({
                "id_loan": loan_id,
                "period": int(row["period"]),
                "month_index": int(row["month_index"]),
                "target_12m": target,
            })

    return pd.DataFrame(rows)


def run(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    p2 = cfg["phase2"]
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    lgd_cfg = cfg["lgd_scenarios"]

    output_dir = Path(paths["outputs"]) / "phase2"
    model_dir = Path(paths["models"]) / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────
    orig = load_freddie_mac_origination(paths["freddie_mac_dir"])
    perf = load_freddie_mac_performance(paths["freddie_mac_dir"])

    # ── 2. Build target ───────────────────────────────────────
    logger.info("Building 12-month default target...")
    target_df = build_target(perf, dpd_threshold=p2["default_dpd_threshold"])
    logger.info("Target built: %d observations, default rate: %.2f%%",
                len(target_df), target_df["target_12m"].mean() * 100)

    # ── 3. Build modeling dataset ─────────────────────────────
    assert_no_duplicate_ids(orig["id_loan"])
    dataset = target_df.merge(orig, on="id_loan", how="left", validate="many_to_one",
                              indicator=True)
    merge_check(dataset, "target", "origination")
    dataset = dataset.drop(columns=["_merge"])

    features = p2["numerical_features"] + p2["categorical_features"]
    target_col = p2["target"]
    group_col = p2["group_column"]

    for col in p2["numerical_features"]:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    for col in p2["categorical_features"]:
        dataset[col] = dataset[col].fillna("MISSING").astype(str)

    dataset = dataset.dropna(subset=[target_col, group_col]).copy()
    assert_binary_target(dataset[target_col], target_col)

    X = dataset[features].copy()
    y = dataset[target_col].astype(int)
    groups = dataset[group_col]

    # ── 4. Train PD model ─────────────────────────────────────
    model = Phase2PDModel(
        numerical_features=p2["numerical_features"],
        categorical_features=p2["categorical_features"],
        max_iter=model_cfg["max_iter"],
        solver=model_cfg["solver"],
        test_size=p2["test_size"],
        random_state=p2["random_state"],
    )

    train_idx, test_idx = model.split(X, y, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

    model.fit(X_train, y_train, g_train)
    metrics = model.evaluate(X_test, y_test, g_test)

    logger.info("Validation metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    pd.DataFrame([metrics]).to_csv(output_dir / "pd_validation_metrics.csv", index=False)

    # ── 5. Train final model on full data → score snapshot ────
    logger.info("Training final model on full dataset...")
    final_model = Phase2PDModel(
        numerical_features=p2["numerical_features"],
        categorical_features=p2["categorical_features"],
        max_iter=model_cfg["max_iter"],
        solver=model_cfg["solver"],
    )
    final_model.fit(X, y, groups)
    final_model.save(model_dir)

    snapshot = dataset[dataset["period"].astype(str) == p2["snapshot_period"]].copy()
    if snapshot.empty:
        raise ValueError(f"No data for snapshot period {p2['snapshot_period']}")
    assert_no_duplicate_ids(snapshot["id_loan"])

    snapshot["pd_12m"] = final_model.predict_proba(snapshot[features])
    logger.info("Portfolio snapshot: %d loans, avg PD: %.4f%%",
                len(snapshot), snapshot["pd_12m"].mean() * 100)

    # ── 6. Merge performance for EAD fields ───────────────────
    perf_snap = perf[perf["period"].astype(str) == p2["snapshot_period"]].copy()
    for col in ["curr_act_upb", "loan_age", "mths_remng"]:
        perf_snap[col] = pd.to_numeric(perf_snap[col], errors="coerce")

    portfolio = snapshot.merge(
        perf_snap[["id_loan", "period", "curr_act_upb", "loan_age", "mths_remng"]],
        on=["id_loan", "period"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    merge_check(portfolio, "snapshot", "performance")
    portfolio = portfolio.drop(columns=["_merge"])

    # ── 7. EAD ────────────────────────────────────────────────
    ead = EADEstimator()
    portfolio = ead.estimate(portfolio)
    ead.summary(portfolio).to_csv(output_dir / "ead_summary.csv", index=False)

    # ── 8. Lifetime PD ────────────────────────────────────────
    projector = LifetimePDProjector()
    portfolio = projector.project(portfolio)
    projector.summary(portfolio).to_csv(output_dir / "lifetime_pd_summary.csv", index=False)

    # ── 9. ECL ────────────────────────────────────────────────
    lgd = LGDScenarios(lgd_cfg)
    calculator = ECLCalculator(lgd)

    portfolio["orig_int_rt"] = pd.to_numeric(portfolio["orig_int_rt"], errors="coerce")
    results = calculator.calculate(portfolio)

    results["loan"].to_csv(output_dir / "ecl_loan_level.csv", index=False)
    results["annual"].to_csv(output_dir / "ecl_annual_breakdown.csv", index=False)
    results["summary"].to_csv(output_dir / "ecl_scenario_summary.csv", index=False)

    logger.info("\nECL Scenario Summary:\n%s", results["summary"].to_string(index=False))
    logger.info("Phase II pipeline complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    run()
