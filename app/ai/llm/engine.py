"""
app/ai/llm/engine.py
─────────────────────────────────────────────────────────────────────────────
AI Cognitive Core - LLM Engine layer for Eatlytic.
Stateless AI functions only (LLM helper routines and label recovery).
Provides backward-compatible re-exports for the service orchestration functions.
"""

import logging
import asyncio
from app.ai.llm.client import call_llm, parse_llm_response

logger = logging.getLogger(__name__)

async def recover_label_with_ai(raw_text: str) -> dict:
    if not raw_text or len(raw_text) < 20:
        return {"is_valid": False, "clean_text": ""}
    sample = raw_text[:1000]
    prompt = f"""
The following was OCR-extracted from a food product image.
Does it contain a nutrition table or ingredient list?

TEXT:
{sample}

Rules:
1. VALID if you see ANY of: Fat, Energy, Calories, Sugar, Protein, Serving, Saturated,
   Carbohydrate, Sodium, or a list of chemical/food ingredient names.
2. INVALID only if it is PURELY marketing text with zero numbers or nutrient words.
3. When in doubt → respond VALID.

Return ONLY this JSON object:
{{"status": "VALID" | "INVALID", "cleaned_text": "<relevant text or empty>"}}
"""
    try:
        raw = await asyncio.to_thread(call_llm, prompt, 800)
        res = parse_llm_response(raw)
        return {
            "is_valid": res.get("status") == "VALID",
            "clean_text": res.get("cleaned_text") or raw_text
        }
    except Exception as e:
        logger.error("AI recovery failed: %s", e)
        return {"is_valid": False, "clean_text": ""}


def __getattr__(name: str):
    if name in (
        "unified_analyze_flow",
        "find_db_product_match",
        "upsert_food_product",
        "build_offline_match_response"
    ):
        import app.services.scan_orchestrator as scan_orchestrator
        return getattr(scan_orchestrator, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

