import time
import json
import logging
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock dependencies to avoid actual API/DB calls during benchmark
from app.services.research_engine import get_live_search, _CACHE

def benchmark_cache_performance():
    query = "Nutrition facts for Amul Butter"
    
    # 1. Clear caches
    _CACHE.clear()
    
    # 2. Simulate First Run (Cold Cache - Web Search Mocked)
    with patch('app.services.research_engine.DDGS') as mock_ddgs:
        # Mock DDG results
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.text.return_value = [
            {"title": "Amul Butter Info", "body": "100g contains 722kcal, 80g Fat."}
        ]
        
        start_time = time.time()
        res1 = get_live_search(query)
        cold_latency = (time.time() - start_time) * 1000
        logger.info(f"Cold Search Latency: {cold_latency:.2f}ms")
        assert "722kcal" in res1

    # 3. Simulate Second Run (Level 1 Cache - Memory)
    start_time = time.time()
    res2 = get_live_search(query)
    l1_latency = (time.time() - start_time) * 1000
    logger.info(f"L1 (Memory) Cache Latency: {l1_latency:.2f}ms")
    assert res1 == res2

    # 4. Simulate Third Run (Level 2 Cache - DB Mocked)
    _CACHE.clear() # Clear memory to force DB lookup
    with patch('app.models.db.get_research_cache') as mock_get_db:
        mock_get_db.return_value = res1 # Simulate DB find
        
        start_time = time.time()
        res3 = get_live_search(query)
        l2_latency = (time.time() - start_time) * 1000
        logger.info(f"L2 (DB) Cache Latency: {l2_latency:.2f}ms")
        assert res1 == res3

if __name__ == "__main__":
    logger.info("--- Starting Phase 2 Latency Benchmarking ---")
    benchmark_cache_performance()
    logger.info("--- Benchmarking Complete ---")
