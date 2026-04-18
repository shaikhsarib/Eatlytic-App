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
    Includes simple in-memory caching to stay within DDG rate limits.
    """
    if not _DDGS_AVAILABLE:
        return "Web research is currently unavailable (duckduckgo_search not installed)."
    
    # Simple TTL cache (24 hours) to prevent DDG blocking
    if query in _CACHE and time.time() - _CACHE[query][0] < 86400:
        return _CACHE[query][1]

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

        _CACHE[query] = (time.time(), result_str)
        return result_str

    except Exception as e:
        logger.warning("Live search failed: %s", e)
        return "Web research is currently unavailable."
