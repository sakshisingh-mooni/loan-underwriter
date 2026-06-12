#!/bin/bash
# startup_api.sh
# --------------
# Azure App Service startup script for the FastAPI REST layer (api.py).
# Set as the "Startup Command" on the API App Service instance:
#   App Service → Settings → Configuration → General settings → Startup Command
#
# This script is SEPARATE from startup.sh (which runs the Streamlit UI).
# Both scripts are deployed to the same App Service Plan; each runs on its own
# App Service instance. See AZURE_DEPLOY.md § "REST API — Separate App Service"
# for the full deployment steps.
#
# Port constraint:
#   Azure App Service (Linux) exposes port 8000 externally regardless of what
#   internal port the app binds to, as long as the process listens on 0.0.0.0.
#   The PORT environment variable is set by the App Service runtime; uvicorn
#   reads it via the shell substitution below.
#   Reference: https://learn.microsoft.com/en-us/azure/app-service/configure-language-python
#
# Model training:
#   api.py imports graph.py which imports risk_scorer.py which loads the sklearn
#   model on first inference. If the .joblib is absent, utils/risk_model.py builds
#   a minimal fallback inline (~1 second). For cold-start predictability, we train
#   the full model here exactly as startup.sh does for the Streamlit app.
#   To skip training (faster cold start): commit models/underwriting_risk_model.joblib
#   to git and remove the models/*.joblib line from .gitignore.
#
# uvicorn workers:
#   The B2 plan has 2 vCPUs. The --workers flag is intentionally omitted here
#   because uvicorn's built-in single-worker mode is simpler and sufficient for
#   a portfolio/demo workload. The underwriting graph is I/O-bound (LLM API calls),
#   not CPU-bound, so a single uvicorn worker with async handling is adequate.
#   For a production NBFC deployment: replace uvicorn with gunicorn + UvicornWorker:
#     gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} api:app
#   Reference: https://www.uvicorn.org/deployment/#gunicorn

set -e

MODEL_PATH="models/underwriting_risk_model.joblib"

echo "[startup_api] Checking for trained model at $MODEL_PATH..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "[startup_api] Model not found — training now (takes ~30s)..."
    python train_model.py
    echo "[startup_api] Model training complete."
else
    echo "[startup_api] Model found — skipping training."
fi

# PORT is injected by Azure App Service at runtime (default 8000).
# Reference: https://learn.microsoft.com/en-us/azure/app-service/reference-app-settings
PORT="${PORT:-8000}"

echo "[startup_api] Starting FastAPI/uvicorn on 0.0.0.0:${PORT}..."
python -m uvicorn api:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-level info
