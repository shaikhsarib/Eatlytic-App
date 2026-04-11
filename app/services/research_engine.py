"""
app/services/research_engine.py
Live web research using DuckDuckGo Search.
Import is guarded — module loads safely even if duckduckgo_search is not installed.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    _DDGS_AVAILABLE = False
    logger.warning("duckduckgo_search not installed — web research disabled.")

def get_live_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for ingredient/brand health info to provide context to the LLM.
    Returns a safe fallback string if the package is not installed.
    """
    if not _DDGS_AVAILABLE:
        return "Web research is currently unavailable (duckduckgo_search not installed)."
    try:
        with DDGS() as ddgs:
            results = [
                f"{r['title']}: {r['body']}"
                for r in ddgs.text(query, max_results=max_results)
            ]

        if not results:
            return "No recent web data found for this specific query."

        return "\n---\n".join(results)
    except Exception as e:
        logger.warning("Live search failed: %s", e)
        return "Web research is currently unavailable."
