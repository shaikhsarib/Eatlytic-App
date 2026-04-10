"""
app/services/research_engine.py
Live web research using DuckDuckGo Search.
"""

import logging
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

def get_live_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for ingredient/brand health info to provide context to the LLM.
    """
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
