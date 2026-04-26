"""
app/services/research_engine.py
Live web research using DuckDuckGo Search.
Import is guarded — module loads safely even if duckduckgo_search is not installed.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
    import warnings
    warnings.filterwarnings("ignore", module="duckduckgo_search")
    _DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    _DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed — web research disabled.")

import time

_CACHE = {}

def get_live_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for ingredient/brand health info to provide context to the LLM.
    Uses a 2-tier cache: L1 (In-Memory) -> L2 (Persistent DB).
    """
    if not _DDGS_AVAILABLE:
        return "Web research is currently unavailable."
    
    # Tier 1: L1 In-Memory (fastest)
    if query in _CACHE and time.time() - _CACHE[query][0] < 3600:
        return _CACHE[query][1]

    # Tier 2: L2 Persistent DB (Phase 2 Hardening)
    try:
        from app.models.db import get_research_cache, set_research_cache
        db_res = get_research_cache(query)
        if db_res:
            _CACHE[query] = (time.time(), db_res)
            return db_res
    except Exception as e:
        logger.debug("L2 Cache lookup failed: %s", e)

    # Fallback: Live Web Search
    try:
        with DDGS() as ddgs:
            results = [
                f"{r['title']}: {r['body']}"
                for r in ddgs.text(query, max_results=max_results)
            ]

        if not results:
            result_str = "No recent web data found for this specific query."
        else:
            result_str = "\n---\n".join(results)

        # Update both caches
        _CACHE[query] = (time.time(), result_str)
        try: set_research_cache(query, result_str)
        except: pass
        
        return result_str

    except Exception as e:
        logger.warning("Live search failed: %s", e)
        return "Web research is currently unavailable."
