"""
train_model.py
--------------
One-time script: trains the risk scoring model and saves it as a sklearn
Pipeline to models/underwriting_risk_model.joblib.

Run once before starting the app:
    python train_model.py

MODEL DESIGN — GradientBoostingRegressor on a continuous risk score:
  Classification on hard binary labels (default/no-default) produces
  near-binary probabilities on synthetic data — every borderline applicant
  either scores 0.000 or 1.000, making the "Medium" risk band useless.

  Root cause: synthetic data is linearly separable (each risk factor is a
  clean threshold). XGBoost / GradientBoostingClassifier learns these
  thresholds perfectly and assigns probability 0 or 1 to everything.

  Fix: train a regressor on a CONTINUOUS risk score instead of a classifier
  on binary labels. The target is a weighted combination of normalised risk
  factors with Gaussian noise — this forces the model to output real-valued
  scores across [0, 1], and borderline inputs produce scores in 0.35–0.65.

  At inference time, score_applicant() clips the regressor output to [0, 1]
  and maps to Low/Medium/High using the same thresholds as before.

  PR-AUC is replaced by MAE (mean absolute error) as the evaluation metric
  since this is now a regression problem.

Feature vector (order must match FEATURE_NAMES in utils/risk_model.py):
  [cibil_score, foir, loan_to_income_ratio, doc_confidence,
   active_loans_count, payment_history_months]

References:
  sklearn Pipeline persistence: https://scikit-learn.org/stable/model_persistence.html
  GradientBoostingRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
  joblib persistence: https://joblib.readthedocs.io/en/latest/persistence.html
"""
import pathlib
import logging

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MODEL_DIR = pathlib.Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "underwriting_risk_model.joblib"

FEATURE_NAMES = [
    "cibil_score",
    "foir",
    "loan_to_income_ratio",
    "doc_confidence",
    "active_loans_count",
    "payment_history_months",
]


def generate_training_data(n: int = 10000, seed: int = RANDOM_STATE):
    """
    Generate synthetic training data with a continuous risk score target.

    Each feature is normalised to [0, 1] with domain-appropriate direction
    (high CIBIL = low risk, high FOIR = high risk, etc.) and combined as a
    weighted sum. Gaussian noise (σ=0.06) simulates real-world uncertainty
    and prevents the model from learning a perfectly sharp boundary.

    Feature weights:
      cibil_score        0.30  — strongest predictor of repayment
      foir               0.25  — over-leveraged borrowers default more
      loan_to_income     0.20  — loan size relative to ability to repay
      doc_confidence     0.10  — low-quality docs signal hidden risk
      active_loans       0.10  — too many concurrent obligations
      payment_history    0.05  — thin history = unknown risk
    """
    rng = np.random.default_rng(seed)

    cibil = rng.integers(300, 900, n).astype(float)
    foir = rng.uniform(0.10, 0.90, n)
    lti = rng.uniform(0.5, 8.0, n)
    doc_conf = rng.uniform(0.3, 1.0, n)
    active_loans = rng.integers(0, 8, n).astype(float)
    payment_months = rng.integers(0, 120, n).astype(float)

    X = np.column_stack([cibil, foir, lti, doc_conf, active_loans, payment_months])

    # Continuous risk score: weighted sum of normalised risk factors
    risk_raw = (
        0.30 * np.clip((750 - cibil) / 450, 0, 1)            # cibil: high = low risk
        + 0.25 * np.clip((foir - 0.20) / 0.60, 0, 1)         # foir: high = high risk
        + 0.20 * np.clip((lti - 1.0) / 6.0, 0, 1)            # lti: high = high risk
        + 0.10 * np.clip((0.8 - doc_conf) / 0.5, 0, 1)       # doc: low conf = higher risk
        + 0.10 * np.clip((active_loans - 1) / 6, 0, 1)       # active loans: many = higher risk
        + 0.05 * np.clip((36 - payment_months) / 36, 0, 1)   # payment: short = higher risk
    )

    # Add Gaussian noise — simulates real-world uncertainty, prevents over-fitting
    noise = rng.normal(0, 0.06, n)
    y = np.clip(risk_raw + noise, 0.0, 1.0)

    logger.info(
        "Generated %d samples | Risk score mean=%.3f std=%.3f | "
        "Low(<0.35): %.1f%%  Medium(0.35-0.65): %.1f%%  High(>=0.65): %.1f%%",
        n, y.mean(), y.std(),
        (y < 0.35).mean() * 100,
        ((y >= 0.35) & (y < 0.65)).mean() * 100,
        (y >= 0.65).mean() * 100,
    )
    return X, y


def build_pipeline() -> Pipeline:
    """
    Build sklearn Pipeline: StandardScaler → GradientBoostingRegressor.

    min_samples_leaf=15 prevents overfitting on the noisy target and
    forces the model to produce smooth, non-binary predictions.
    """
    regressor = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def train_and_save():
    logger.info("Generating training data…")
    X, y = generate_training_data(n=10000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    logger.info("Training GradientBoostingRegressor pipeline…")
    pipeline.fit(X_train, y_train)

    preds = np.clip(pipeline.predict(X_test), 0, 1)
    mae = mean_absolute_error(y_test, preds)
    logger.info("MAE on hold-out: %.4f (target: <0.06)", mae)

    # Verify probability distribution on borderline inputs
    rng = np.random.default_rng(55)
    X_border = np.column_stack([
        rng.integers(550, 750, 500).astype(float),
        rng.uniform(0.40, 0.72, 500),
        rng.uniform(2.5, 6.0, 500),
        rng.uniform(0.45, 0.80, 500),
        rng.integers(1, 6, 500).astype(float),
        rng.integers(6, 48, 500).astype(float),
    ])
    border_scores = np.clip(pipeline.predict(X_border), 0, 1)
    pct_medium = ((border_scores >= 0.35) & (border_scores < 0.65)).mean() * 100
    logger.info("Borderline zone %% Medium label: %.1f%% (target: >15%%)", pct_medium)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Model saved → %s", MODEL_PATH)

    return pipeline


if __name__ == "__main__":
    train_and_save()
