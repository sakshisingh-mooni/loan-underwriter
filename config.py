"""
config.py
---------
Centralised configuration with lazy initialization.

FIX: The original config called Config.from_env() at module level, meaning
any import of this module would raise EnvironmentError if keys were missing.
This broke unit tests, CI pipelines, and cold-start tooling.

The fix: `cfg` is now a module-level lazy proxy. The first access triggers
validation; subsequent accesses return the cached instance.

Usage (unchanged from original):
    from config import cfg
    cfg.groq_api_key
    cfg.langfuse_public_key

References:
  - Python dataclasses docs: https://docs.python.org/3/library/dataclasses.html
  - python-dotenv: https://pypi.org/project/python-dotenv/
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # API keys
    groq_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str

    # LLM settings
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500

    # Graph control
    max_revisions: int = 2

    # Loan product constants (simplified RBI/NBFC guidelines)
    max_foir: float = 0.55              # 55% — standard NBFC FOIR limit
    min_cibil_score: int = 650
    max_loan_to_income_ratio: float = 5.0
    min_age: int = 21
    max_age: int = 65

    # Application environment — controls checkpointer selection
    # Set to "production" in Azure to use PostgresSaver
    app_env: str = "development"

    # PostgreSQL connection string — required when app_env == "production"
    database_url: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        required = {
            "GROQ_API_KEY": "Get free key at console.groq.com",
            "LANGFUSE_PUBLIC_KEY": "Get free key at cloud.langfuse.com → Project Settings",
            "LANGFUSE_SECRET_KEY": "Get free key at cloud.langfuse.com → Project Settings",
        }
        missing = [
            f"  {k} — {hint}"
            for k, hint in required.items()
            if not os.environ.get(k)
        ]
        if missing:
            raise EnvironmentError(
                "Missing required environment variables:\n"
                + "\n".join(missing)
                + "\n\nCopy .env.example to .env and fill in your keys."
            )

        app_env = os.environ.get("APP_ENV", "development")
        database_url = os.environ.get("DATABASE_URL", "")

        if app_env == "production" and not database_url:
            raise EnvironmentError(
                "APP_ENV=production requires DATABASE_URL to be set.\n"
                "Set DATABASE_URL to your Azure PostgreSQL connection string."
            )

        return cls(
            groq_api_key=os.environ["GROQ_API_KEY"],
            langfuse_public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            langfuse_secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            langfuse_host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            app_env=app_env,
            database_url=database_url,
        )


# ── Lazy singleton ─────────────────────────────────────────────────────────────
# Validated on first access, not at import time.
# This means unit tests can import any module without needing .env to exist.
#
# Pattern: __getattr__ on the module is called for any name not found in the
# module's __dict__. We cache the result back into __dict__ so it's only
# constructed once.
#
# Reference: https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access

_cfg_instance: Config | None = None


def _get_cfg() -> Config:
    global _cfg_instance
    if _cfg_instance is None:
        _cfg_instance = Config.from_env()
    return _cfg_instance


def __getattr__(name: str):
    if name == "cfg":
        return _get_cfg()
    raise AttributeError(f"module 'config' has no attribute {name!r}")
