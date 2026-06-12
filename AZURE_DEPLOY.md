# Azure Deployment Guide — Loan Underwriting AI

## Prerequisites
- Azure CLI installed and logged in (`az login`)
- Python 3.11 virtual environment working locally
- All 5 environment variables from `.env.example` ready

---

## Step 1 — Resource group and App Service Plan

```bash
# Choose your region (centralindia is closest to Bhopal)
REGION="centralindia"
RG="rg-loan-underwriter"
PLAN="plan-loan-underwriter"
APP="loan-underwriter-ai"          # must be globally unique

az group create --name $RG --location $REGION

# B2 minimum: B1 (1.75 GB RAM) is tight for Streamlit + sklearn inference
# + LLM API call buffering under concurrent load. B2 (3.5 GB) gives headroom.
az appservice plan create \
    --name $PLAN \
    --resource-group $RG \
    --sku B2 \
    --is-linux

az webapp create \
    --name $APP \
    --resource-group $RG \
    --plan $PLAN \
    --runtime "PYTHON:3.11"
```

---

## Step 2 — PostgreSQL Flexible Server (required for HITL in production)

MemorySaver is in-process RAM. A container restart (which Azure does for
patching, cold starts, and scaling) wipes all checkpoints, meaning any
in-flight HITL Refer cases are lost and the resume will fail. PostgresSaver
persists checkpoints in Azure PostgreSQL across restarts.

```bash
PG_SERVER="pg-loan-underwriter"
PG_DB="underwriting"
PG_USER="loanadmin"
PG_PASS="<choose a strong password>"

az postgres flexible-server create \
    --name $PG_SERVER \
    --resource-group $RG \
    --location $REGION \
    --admin-user $PG_USER \
    --admin-password $PG_PASS \
    --sku-name Standard_B1ms \
    --storage-size 32 \
    --version 15 \
    --public-access 0.0.0.0

az postgres flexible-server db create \
    --database-name $PG_DB \
    --server-name $PG_SERVER \
    --resource-group $RG

# Allow App Service outbound IPs to reach PostgreSQL
# (simpler: allow Azure services)
az postgres flexible-server firewall-rule create \
    --name AllowAzureServices \
    --resource-group $RG \
    --server-name $PG_SERVER \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0
```

---

## Step 3 — Set environment variables on App Service

```bash
az webapp config appsettings set \
    --name $APP \
    --resource-group $RG \
    --settings \
        GROQ_API_KEY="<your-groq-key>" \
        LANGFUSE_PUBLIC_KEY="<your-langfuse-public-key>" \
        LANGFUSE_SECRET_KEY="<your-langfuse-secret-key>" \
        LANGFUSE_HOST="https://cloud.langfuse.com" \
        APP_ENV="production" \
        DATABASE_URL="postgresql://${PG_USER}:${PG_PASS}@${PG_SERVER}.postgres.database.azure.com:5432/${PG_DB}?sslmode=require"
```

---

## Step 4 — Set startup command

Azure App Service runs Python apps with Gunicorn by default, which does not
work with Streamlit. The startup.sh script handles model training and launches
Streamlit directly on port 8000.

```bash
az webapp config set \
    --name $APP \
    --resource-group $RG \
    --startup-file "startup.sh"
```

---

## Step 5 — Deploy code

```bash
# From the project root (loan_underwriter/)
az webapp up \
    --name $APP \
    --resource-group $RG \
    --runtime "PYTHON:3.11" \
    --sku B2
```

Or use VS Code Azure extension: right-click the project folder →
"Deploy to Web App" → select your App Service.

---

## Step 6 — Verify deployment

```bash
# Stream live logs
az webapp log tail --name $APP --resource-group $RG

# App URL
echo "https://${APP}.azurewebsites.net"
```

Open the URL. You should see:
- "✅ Langfuse tracing active" (if keys are correct)
- Three sample applicant buttons in the sidebar

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langfuse` | requirements.txt not installed | Check build logs, ensure runtime is PYTHON:3.11 |
| `EnvironmentError: Missing GROQ_API_KEY` | App settings not set | Re-run Step 3 |
| HITL resume fails after overnight | MemorySaver wiped by restart | Ensure `APP_ENV=production` and `DATABASE_URL` are set |
| `psycopg.OperationalError` | PostgreSQL firewall | Re-run the firewall rule in Step 2 |
| `invalid connection option "+psycopg"` | Wrong DATABASE_URL format | Use `postgresql://...` not `postgresql+psycopg://...` — the `+psycopg` prefix is SQLAlchemy-only; psycopg3 uses libpq URIs |
| `ModuleNotFoundError: psycopg_pool` | psycopg-pool not installed | Ensure `psycopg-pool>=3.2.0` is in requirements.txt |
| Streamlit not loading | Wrong port | Confirm startup.sh uses `--server.port 8000` |
| Cold start takes 40+ seconds | Model training on startup | Commit `models/underwriting_risk_model.joblib` to git (remove from .gitignore) to skip training |

---

## REST API — Separate App Service Deployment

### Why a separate App Service?

Azure App Service (Linux) binds to a single port per instance (port 8000 by
default, exposed externally on 443). Running Streamlit and uvicorn on the same
instance is not supported: Streamlit occupies port 8000, and there is no
secondary external port to serve uvicorn.

The correct pattern is two App Service **instances** on the **same App Service
Plan**. Multiple web apps on a single plan share compute resources (the B2 plan
has 2 vCPUs and 3.5 GB RAM) but each app gets its own hostname, startup
command, and environment variables. You pay for the plan once; additional apps
on the same plan have no extra hosting cost.

Reference: https://learn.microsoft.com/en-us/azure/app-service/overview-hosting-plans

### Step A — Create the API App Service (same plan, new app)

```bash
# Reuse the variables set in Step 1 (RG, PLAN, PG_SERVER, PG_DB, PG_USER, PG_PASS)
API_APP="loan-underwriter-api"    # must be globally unique; different from $APP

az webapp create \
    --name $API_APP \
    --resource-group $RG \
    --plan $PLAN \
    --runtime "PYTHON:3.11"
```

### Step B — Set environment variables on the API App Service

The API app needs the same environment variables as the Streamlit app.
PostgresSaver keeps HITL checkpoints alive across the two instances because
both read from the same Azure PostgreSQL database via `thread_id`.

```bash
az webapp config appsettings set \
    --name $API_APP \
    --resource-group $RG \
    --settings \
        GROQ_API_KEY="<your-groq-key>" \
        LANGFUSE_PUBLIC_KEY="<your-langfuse-public-key>" \
        LANGFUSE_SECRET_KEY="<your-langfuse-secret-key>" \
        LANGFUSE_HOST="https://cloud.langfuse.com" \
        APP_ENV="production" \
        DATABASE_URL="postgresql://${PG_USER}:${PG_PASS}@${PG_SERVER}.postgres.database.azure.com:5432/${PG_DB}?sslmode=require"
```

### Step C — Set the startup command

`startup_api.sh` trains the risk model if absent, then starts uvicorn on the
port Azure injects via the `PORT` environment variable (default 8000).

```bash
az webapp config set \
    --name $API_APP \
    --resource-group $RG \
    --startup-file "startup_api.sh"
```

### Step D — Deploy code

```bash
az webapp up \
    --name $API_APP \
    --resource-group $RG \
    --runtime "PYTHON:3.11" \
    --sku B2
```

### Step E — Verify

```bash
# Stream live logs
az webapp log tail --name $API_APP --resource-group $RG

API_URL="https://${API_APP}.azurewebsites.net"

# Health check
curl "${API_URL}/health"
# Expected: {"status":"ok","graph_nodes":["document_parser","bureau_fetcher",...]}

# Submit a test application
curl -X POST "${API_URL}/underwrite" \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_name": "Arjun Mehta",
    "applicant_age": 32,
    "annual_income": 1200000,
    "loan_amount_requested": 3000000,
    "loan_purpose": "Home Purchase",
    "employment_type": "Salaried",
    "existing_obligations": 10000,
    "document_text": "Salary certificate — Arjun Mehta. Annual CTC: 12,00,000.",
    "cibil_score_override": 760
  }'

# OpenAPI / Swagger UI (browser)
echo "Swagger UI: ${API_URL}/docs"
```

### Troubleshooting (API app)

| Symptom | Cause | Fix |
|---|---|---|
| `uvicorn: command not found` | `uvicorn[standard]` not installed | Confirm `uvicorn[standard]>=0.29.0` is in `requirements.txt` |
| `{"detail":"Graph execution failed: ..."}` | Missing env vars on API app | Re-run Step B; check `az webapp config appsettings list --name $API_APP` |
| HITL `/resume` returns 500 after restart | MemorySaver wiped | Confirm `APP_ENV=production` and `DATABASE_URL` are set on the API app |
| `Application Error` on cold start | Model training timed out | Commit `models/underwriting_risk_model.joblib` to git to skip training on startup |
| CORS errors from a frontend calling the API | CORS not configured | Add `fastapi.middleware.cors.CORSMiddleware` to `api.py` with your allowed origins |
