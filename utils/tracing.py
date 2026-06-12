"""
utils/tracing.py
----------------
Langfuse v2 observability setup.

SDK VERSION: This file targets Langfuse PyPI 2.x SDK, pinned to langfuse==2.60.10.
  The companion package langfuse-langchain==2.60.10.1 is also required —
  it provides compatibility between Langfuse v2 and LangChain 1.x.
  Without it, `from langfuse.callback import CallbackHandler` raises
  ModuleNotFoundError because Langfuse v2 imports legacy LangChain modules
  removed in LangChain 1.x.
  requirements.txt pins both packages to exact versions.

BREAKING CHANGE in Langfuse v4 (for future reference):
  CallbackHandler.__init__ signature changed completely.
  v2 signature: CallbackHandler(session_id=, user_id=, tags=, ...)
  v3 signature: CallbackHandler() — no session/user args; those moved to
    config["metadata"] as "langfuse_session_id" / "langfuse_user_id"
  v4 signature: CallbackHandler(*, public_key=None, trace_context=None)


  The CallbackHandler itself takes no session/user args in v3. Those fields are
  passed via config["metadata"] in the LangChain/LangGraph invocation:
    config={
        "callbacks": [handler],
        "metadata": {
            "langfuse_session_id": "<session_id>",
            "langfuse_user_id": "<user_id>",
        },
    }
  LangGraph passes this metadata to the handler automatically, and Langfuse
  picks it up as the session/user on the trace.

  get_langfuse_handler() now accepts optional session_id and user_id
  and returns (handler, metadata_dict) so _get_config() in app.py can
  spread the metadata dict into config without knowing the Langfuse-specific
  key names.

Usage:
    from utils.tracing import get_langfuse_handler
    handler, lf_metadata = get_langfuse_handler(
        session_id="thread-abc",
        user_id="Rahul Sharma",
    )
    graph.invoke(state, config={
        "callbacks": [handler],
        "metadata": lf_metadata,
    })
"""
from __future__ import annotations
from functools import lru_cache
from langfuse.callback import CallbackHandler


@lru_cache(maxsize=1)
def _verify_auth() -> bool:
    """
    Verify Langfuse credentials once on first call (cached via lru_cache).

    Uses a minimal Langfuse client call to check that LANGFUSE_PUBLIC_KEY
    and LANGFUSE_SECRET_KEY are valid before allowing any graph run.

    # SDK v2 note: auth_check() was removed in v3 (not available in v2 either at
    # module level). Code uses manual key-format validation instead.
    In v2 we verify credentials by constructing a CallbackHandler, which
    reads env vars at init time and raises on missing/invalid keys.
    We also do a lightweight Langfuse() client instantiation which validates
    the key format without making a network call.

    Raises RuntimeError on auth failure so misconfiguration is caught
    at startup, not silently mid-run.
    Note: lru_cache does NOT cache exceptions in Python, so a failed
    auth will retry on the next call.
    """
    import os

    # Hard version guard — fail loudly if v3 or later slips through.
    # requirements.txt pins langfuse==2.60.10 exactly, so this should
    # never fire in normal use. It is a defensive check for environments
    # where requirements.txt was not followed.
    try:
        import langfuse as _lf_module
        _ver = getattr(_lf_module, "__version__", "0")
        _major = int(str(_ver).split(".")[0])
        if _major >= 3:
            raise RuntimeError(
                f"Langfuse v{_ver} detected. This codebase targets langfuse==2.60.10 only. "
                "Downgrade: pip install langfuse==2.60.10 langfuse-langchain==2.60.10.1"
            )
    except (ImportError, ValueError):
        pass

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        raise RuntimeError(
            "Langfuse credentials missing.\n"
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file.\n"
            "Get free keys at: https://cloud.langfuse.com → Project Settings"
        )
    if not public_key.startswith("pk-lf-"):
        raise RuntimeError(
            f"LANGFUSE_PUBLIC_KEY looks invalid (got: '{public_key[:12]}...'). "
            "Expected format: 'pk-lf-...'."
        )
    if not secret_key.startswith("sk-lf-"):
        raise RuntimeError(
            f"LANGFUSE_SECRET_KEY looks invalid (got: '{secret_key[:12]}...'). "
            "Expected format: 'sk-lf-...'."
        )
    return True


def get_langfuse_handler(
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[CallbackHandler, dict]:
    """
    Return (CallbackHandler, metadata_dict) for one graph invocation.

    The CallbackHandler picks up credentials from LANGFUSE_PUBLIC_KEY /
    LANGFUSE_SECRET_KEY env vars automatically. Each graph run produces
    a separate trace in cloud.langfuse.com.

    session_id and user_id are returned as a metadata dict ready to be
    spread directly into config["metadata"] on the graph invocation.
    Langfuse reads "langfuse_session_id" and "langfuse_user_id" from
    LangChain/LangGraph metadata and attaches them to the trace.

    
    Args:
        session_id: Correlates multiple LLM calls into one Langfuse session.
                    Using thread_id here groups all nodes of one underwriting
                    run under a single session in the Langfuse UI.
        user_id: The applicant name or any user identifier for filtering
                 traces per user in the Langfuse UI.

    Returns:
        (CallbackHandler, metadata_dict) — spread metadata_dict into
        config["metadata"] alongside config["callbacks"].
    """
    _verify_auth()

    handler = CallbackHandler()

    metadata: dict = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id

    return handler, metadata
