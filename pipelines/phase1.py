"""
Phase I pipeline — Loan Default PD modeling.

Loads the Loan Default dataset, applies WOE transformation,
trains a logistic regression PD model, evaluates it and exports
the model artifact with metadata.

Usage:
    python -m pipelines.phase1
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_loan_default
from src.data.validators import assert_binary_target
from src.models.pd_model import Phase1PDModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    p1 = cfg["phase1"]
    paths = cfg["paths"]
    model_cfg = cfg["model"]

    # ── 1. Load ──────────────────────────────────────────────
    df = load_loan_default(paths["loan_default_csv"])

    # ── 2. Clean ─────────────────────────────────────────────
    df["credit_type"] = df["credit_type"].astype(str).str.strip().str.upper()
    df = df[~df["credit_type"].isin(p1["exclude_credit_types"])].copy()

    df[p1["target"]] = pd.to_numeric(df[p1["target"]], errors="coerce")
    df = df[df[p1["target"]].isin([0, 1])].copy()
    df[p1["target"]] = df[p1["target"]].astype(int)

    for col in p1["numerical_features"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["LTV"].le(p1["ltv_max"]) | df["LTV"].isna()].copy()

    assert_binary_target(df[p1["target"]], p1["target"])

    features = p1["numerical_features"] + p1["categorical_features"]
    X = df[features].copy()
    y = df[p1["target"]].copy()

    # ── 3. Train / Test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=p1["test_size"],
        random_state=p1["random_state"],
        stratify=y,
    )
    logger.info("Train: %d | Test: %d | Train default rate: %.2f%%",
                len(X_train), len(X_test), y_train.mean() * 100)

    # ── 4. Train ──────────────────────────────────────────────
    model = Phase1PDModel(
        numerical_features=p1["numerical_features"],
        categorical_features=p1["categorical_features"],
        max_iter=model_cfg["max_iter"],
        solver=model_cfg["solver"],
        woe_max_bins=p1["woe_max_bins"],
        woe_smoothing=p1["woe_smoothing"],
    )
    model.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info("Train AUC: %.4f | Gini: %.4f", train_metrics["roc_auc"], train_metrics["gini"])
    logger.info("Test  AUC: %.4f | Gini: %.4f | Brier: %.4f",
                test_metrics["roc_auc"], test_metrics["gini"], test_metrics["brier_score"])

    logger.info("\nIV Table:\n%s", model.woe_transformer_.iv_table().to_string())
    logger.info("\nCoefficients:\n%s", model.coefficients().to_string(index=False))

    # ── 6. Save ───────────────────────────────────────────────
    output_dir = Path(paths["models"]) / "phase1"
    model.save(output_dir)

    metrics_df = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "test", **test_metrics},
    ])
    metrics_df.to_csv(output_dir / "phase1_pd_metrics.csv", index=False)
    logger.info("Phase I pipeline complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    run()
