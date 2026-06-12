"""
utils/llm.py
------------
Cached LLM factory. Import get_llm() wherever you need the model.
Using functools.lru_cache ensures a single ChatGroq instance is created
regardless of how many agents import this — avoids redundant object creation.
"""
from functools import lru_cache
from langchain_groq import ChatGroq
from config import cfg


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Return the shared ChatGroq LLM instance.
    Temperature 0.1 — low for deterministic underwriting decisions.
    """
    return ChatGroq(
        model=cfg.llm_model,
        api_key=cfg.groq_api_key,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
    )
