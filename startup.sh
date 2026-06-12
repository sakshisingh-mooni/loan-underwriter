#!/bin/bash
# startup.sh
# ----------
# Azure App Service startup script.
# Set as the "Startup Command" in App Service → Settings → Configuration → General settings.
#
# Azure App Service exposes ports 8000 and 443 only.
# --server.address 0.0.0.0 is required so Streamlit binds to all interfaces.
#
# Model training: run once before the app starts. If the .joblib file already
# exists (e.g. baked into the Docker image or committed to the repo), this is
# a no-op. The file check below is the only guard — train_model.py always
# writes when invoked, so we only invoke it when the model is absent.
#
# Reference: https://learn.microsoft.com/en-us/answers/questions/1470782/
#            how-to-deploy-a-streamlit-application-on-azure-app

set -e

MODEL_PATH="models/underwriting_risk_model.joblib"

echo "[startup] Checking for trained model at $MODEL_PATH..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "[startup] Model not found — training now (takes ~30s)..."
    python train_model.py
    echo "[startup] Model training complete."
else
    echo "[startup] Model found — skipping training."
fi

echo "[startup] Starting Streamlit on port 8000..."
python -m streamlit run app.py \
    --server.port 8000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false
    # --server.enableXsrfProtection is left at its default (true).
    # If you see 403s from Azure's load-balancer health checks, add:
    #   --server.enableXsrfProtection false
    # Disabling it removes CSRF protection on all form submissions, so only
    # do this if you are certain the 403s originate from the health check.
