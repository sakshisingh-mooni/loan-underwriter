# conftest.py
# -----------
# Presence of this file at the project root causes pytest to add the project
# root to sys.path automatically via its rootdir detection.
# Reference: https://docs.pytest.org/en/stable/explanation/pythonpath.html
#
# Env stubs: mock.patch.dict("sys.modules", ...) in test_rule_engine.py restores
# sys.modules to its pre-block state when the context manager exits.
# This removes agents.rule_engine and config from sys.modules. The autouse
# patch_cfg fixture then re-imports agents.rule_engine fresh, which re-executes
# `from config import cfg` at module level, triggering Config.from_env().
# Without a .env file, this raises EnvironmentError and crashes all 16 tests.
#
# Fix: set stub values before any test runs. os.environ.setdefault() means
# real keys (if present) always take precedence — this never overwrites a
# legitimate .env value.
# Reference: https://docs.python.org/3/library/os.html#os.environ.setdefault
import os
os.environ.setdefault("GROQ_API_KEY", "test-stub")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test-stub")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test-stub")