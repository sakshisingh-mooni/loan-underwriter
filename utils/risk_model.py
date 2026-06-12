"""
utils/risk_model.py
-------------------
Pluggable risk scoring model interface.

MODEL TYPE — GradientBoostingRegressor (continuous risk score):
  The model predicts a continuous risk score in [0, 1] rather than a
  binary classification probability. This prevents near-binary outputs
  on synthetic data and ensures the "Medium" risk band is populated.

  score_applicant() clips the raw regressor output to [0, 1] and maps
  to Low / Medium / High using fixed thresholds:
    < 0.35 → Low
    0.35 – 0.65 → Medium
    >= 0.65 → High

  Fallback: if models/underwriting_risk_model.joblib is missing, builds a
  minimal inline pipeline so the app starts without needing train_model.py
  first.

SYNTHETIC DATA — SCOPE AND LIMITATIONS:
  The model in this project is trained on 10,000 programmatically generated
  samples. The data generation function in train_model.py embeds domain
  knowledge (feature weights, risk thresholds) directly, which means the
  model is learning a noisy approximation of a hand-crafted formula rather
  than patterns in real loan performance data.

  This is an intentional architectural choice for a portfolio project, not
  a production model. The specific limitations are:

    1. Feature weights are hard-coded in generate_training_data() rather
       than learned from historical loan performance. In production you
       would train on the NBFC's own loan book: approved applications,
       repayment histories, and default outcomes.

    2. The "risk score" target is a weighted sum with Gaussian noise, not
       an actual default probability. A real model's target is a binary
       outcome: did this borrower default within 90 days (NPL definition
       per RBI Master Directions for NBFCs)?

    3. The model cannot learn non-linearities or interactions that only
       exist in real data (e.g. the correlation between employment type
       and payment discipline across economic cycles).

  PRODUCTION REPLACEMENT PATH:
    The architecture is deliberately decoupled so swapping the model
    requires no changes to the graph or agent code:

      Step 1: Obtain historical loan data from the NBFC's LOS (Loan
              Origination System) — approved applications with 12+ month
              repayment history and a binary default label.

      Step 2: Feature-engineer the same six features used here
              (cibil_score, foir, lti, doc_confidence, active_loans,
              payment_months) from the raw LOS data.

      Step 3: Train any sklearn estimator (XGBoost, LightGBM, or a
              calibrated RandomForestClassifier) on the real data.
              Wrap it in a sklearn Pipeline with StandardScaler as the
              first step (or a ColumnTransformer for heterogeneous types).

      Step 4: joblib.dump(your_pipeline, "models/underwriting_risk_model.joblib")

      Step 5: Restart the app. This module loads the new file automatically.
              No other code changes required.

    The only interface contract: the final Pipeline step must expose either
      .predict(X) returning a float in [0, 1]   (use for regressors)
      .predict_proba(X)[:, 1]                   (use for classifiers)
    Both are handled transparently by _is_regressor() and score_applicant().

  WHY GRADIENTBOOSTINGREGRESSOR HERE:
    Gradient boosting on a continuous target avoids the near-binary
    probability collapse that ensemble classifiers produce on linearly
    separable synthetic data. Every borderline applicant gets a score
    in the Medium band (0.35–0.65) rather than 0.000 or 1.000.
    This makes the Risk Scorer output informative for downstream agents
    during demo and testing.

Feature vector (order matters — must match train_model.py FEATURE_NAMES):
  [cibil_score, foir, loan_to_income_ratio, doc_confidence,
   active_loans_count, payment_history_months]

To swap in your own model:
  1. Ensure it is a sklearn Pipeline saved with joblib.
  2. The final step must expose either:
       .predict(X) → float in [0, 1]   (regressor)
       .predict_proba(X)[:, 1]         (classifier)
  3. Run: joblib.dump(your_pipeline, "models/underwriting_risk_model.joblib")
  4. Restart the app — this module loads it automatically.

References:
  joblib persistence: https://joblib.readthedocs.io/en/latest/persistence.html
  sklearn Pipeline:   https://scikit-learn.org/stable/model_persistence.html
  RBI NPA definition: https://www.rbi.org.in/Scripts/FAQView.aspx?Id=96
"""
import pathlib
import logging

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "underwriting_risk_model.joblib"

FEATURE_NAMES = [
    "cibil_score",
    "foir",
    "loan_to_income_ratio",
    "doc_confidence",
    "active_loans_count",
    "payment_history_months",
]

# Module-level cache — populated on first call to score_applicant().
# Deferred to avoid running GradientBoostingRegressor.fit() (the fallback)
# on the import thread, which would block Streamlit's first page load.
# Reference: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
_PIPELINE: Pipeline | None = None
_IS_REGRESSOR: bool | None = None


def _is_regressor(pipeline: Pipeline) -> bool:
    """
    Return True if the pipeline's final estimator is a regressor.
    Regressors use predict(); classifiers use predict_proba().

    Reference: https://scikit-learn.org/stable/glossary.html#term-regressor
    """
    from sklearn.base import is_regressor
    final_step = pipeline.steps[-1][1]
    return is_regressor(final_step)


def _load_or_build_pipeline() -> Pipeline:
    """
    Load the saved Pipeline from disk, or build a minimal fallback.
    Logs a warning if the trained file is missing so the developer
    knows to run train_model.py.
    """
    if MODEL_PATH.exists():
        logger.info("[RiskModel] Loading trained model from %s", MODEL_PATH)
        return joblib.load(MODEL_PATH)

    logger.warning(
        "[RiskModel] %s not found. Run `python train_model.py` to train and "
        "save the model. Using fallback inline pipeline for now.",
        MODEL_PATH,
    )
    return _build_fallback_pipeline()


def _build_fallback_pipeline() -> Pipeline:
    """
    Minimal fallback regressor — same continuous-score approach as train_model.py.
    Trains on 400 samples in ~1 second. Only used when
    underwriting_risk_model.joblib is absent.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n = 400

    cibil = rng.integers(300, 900, n).astype(float)
    foir = rng.uniform(0.10, 0.90, n)
    lti = rng.uniform(0.5, 8.0, n)
    doc_conf = rng.uniform(0.3, 1.0, n)
    active_loans = rng.integers(0, 8, n).astype(float)
    payment_months = rng.integers(0, 120, n).astype(float)

    X = np.column_stack([cibil, foir, lti, doc_conf, active_loans, payment_months])

    y = np.clip(
        0.30 * np.clip((750 - cibil) / 450, 0, 1)
        + 0.25 * np.clip((foir - 0.20) / 0.60, 0, 1)
        + 0.20 * np.clip((lti - 1.0) / 6.0, 0, 1)
        + 0.10 * np.clip((0.8 - doc_conf) / 0.5, 0, 1)
        + 0.10 * np.clip((active_loans - 1) / 6, 0, 1)
        + 0.05 * np.clip((36 - payment_months) / 36, 0, 1)
        + rng.normal(0, 0.06, n),
        0.0, 1.0,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.08,
            min_samples_leaf=5, random_state=42,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def _get_pipeline() -> tuple[Pipeline, bool]:
    """
    Return the cached (pipeline, is_regressor) pair, loading it on first call.

    Thread safety: in CPython, module-level assignments are protected by the
    GIL, so a concurrent first call results in the pipeline being built twice
    at worst — not corruption. Acceptable for this use case.
    """
    global _PIPELINE, _IS_REGRESSOR
    if _PIPELINE is None:
        _PIPELINE = _load_or_build_pipeline()
        _IS_REGRESSOR = _is_regressor(_PIPELINE)
    return _PIPELINE, _IS_REGRESSOR  # type: ignore[return-value]


def score_applicant(
    cibil_score: int,
    foir: float,
    loan_to_income_ratio: float,
    doc_confidence: float,
    active_loans_count: int,
    payment_history_months: int,
) -> tuple[float, str, list[str]]:
    """
    Score a loan applicant using the risk model Pipeline.

    Returns:
        risk_score (float): continuous risk score 0.0 (safe) – 1.0 (high risk)
        risk_label (str):   "Low" | "Medium" | "High"
        features_used (list[str]): top-3 features by model importance

    Handles both regressor pipelines (predict()) and classifier pipelines
    (predict_proba()[:, 1]) transparently via _is_regressor().

    Thresholds:
        < 0.35  → Low    (approve)
        0.35–0.65 → Medium (refer for human review)
        >= 0.65 → High   (reject)
    """
    pipeline, is_reg = _get_pipeline()

    features = np.array([[
        cibil_score,
        foir,
        loan_to_income_ratio,
        doc_confidence,
        active_loans_count,
        payment_history_months,
    ]])

    if is_reg:
        raw = float(pipeline.predict(features)[0])
    else:
        raw = float(pipeline.predict_proba(features)[0][1])

    risk_prob = float(np.clip(raw, 0.0, 1.0))

    if risk_prob < 0.35:
        label = "Low"
    elif risk_prob < 0.65:
        label = "Medium"
    else:
        label = "High"

    # Feature importances from the final estimator
    final_estimator = pipeline.steps[-1][1]
    # Unwrap CalibratedClassifierCV if needed
    # Reference: https://scikit-learn.org/stable/modules/calibration.html
    if hasattr(final_estimator, "estimator"):
        final_estimator = final_estimator.estimator
    if hasattr(final_estimator, "feature_importances_"):
        importances = final_estimator.feature_importances_
        top_indices = np.argsort(importances)[-3:][::-1]
        features_used = [FEATURE_NAMES[i] for i in top_indices]
    else:
        features_used = FEATURE_NAMES[:3]

    logger.info(
        "[RiskModel] score=%.3f label=%s top_features=%s",
        risk_prob, label, features_used,
    )
    return risk_prob, label, features_used
