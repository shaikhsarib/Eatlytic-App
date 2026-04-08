"""
flush_cache.py
Utility script to completely wipe the ai_cache and ocr_cache tables.
Use this when stale cache is causing zeroed-out nutrient results.
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.db import db_conn, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def flush_cache():
    """Clear all cached OCR and AI results."""
    # Ensure tables exist
    init_db()

    logger.info("Flushing OCR and AI caches...")

    with db_conn() as conn:
        # Count existing entries
        ocr_count = conn.execute("SELECT COUNT(*) FROM ocr_cache").fetchone()[0]
        ai_count = conn.execute("SELECT COUNT(*) FROM ai_cache").fetchone()[0]

        logger.info(
            f"Found {ocr_count} OCR cache entries and {ai_count} AI cache entries"
        )

        # Delete all entries
        conn.execute("DELETE FROM ocr_cache")
        conn.execute("DELETE FROM ai_cache")

    logger.info("✅ Cache flushed successfully!")
    logger.info("All future scans will use fresh OCR and AI analysis.")


if __name__ == "__main__":
    flush_cache()
