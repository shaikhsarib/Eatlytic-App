"""
app/routes/additive_db.py
Eatlytic Proprietary Additive Intelligence API.

Exposes the verified additive database for frontend lookup,
search, and ingredient scanning endpoints.
"""
import logging
from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/additive-db", tags=["additive-db"])


def _get_additive_service():
    """Lazy import to handle missing additives.json gracefully."""
    try:
        from app.services.additive_db import (
            lookup,
            scan_ingredients,
            get_ingredient_risk_summary,
            get_all_additives,
            get_additive_by_id,
        )
        return lookup, scan_ingredients, get_ingredient_risk_summary, get_all_additives, get_additive_by_id
    except Exception as e:
        logger.error("Additive DB not available: %s", e)
        return None, None, None, None, None


@router.get("/search")
async def search_additive(
    q: str = Query("", min_length=2, description="Ingredient name, alias, INS code, or E-number"),
    limit: int = Query(10, le=50),
):
    """
    Search the proprietary additive database by name, alias, INS code, or E-number.
    Example: /additive-db/search?q=MSG or /additive-db/search?q=E621
    """
    lookup, _, _, get_all_additives, _ = _get_additive_service()
    if not lookup:
        return JSONResponse({"error": "Additive database unavailable"}, status_code=503)

    if not q or len(q.strip()) < 2:
        return JSONResponse({"results": [], "count": 0})

    result = lookup(q.strip())
    if result:
        return JSONResponse({"results": [result], "count": 1, "source": "exact_match"})

    # Fuzzy fallback: scan all additives for partial name match
    q_lower = q.lower()
    all_additives = get_all_additives()
    fuzzy_results = [
        a for a in all_additives
        if q_lower in a["name"].lower()
        or any(q_lower in alias.lower() for alias in a.get("aliases", []))
    ][:limit]

    return JSONResponse({
        "results": fuzzy_results,
        "count": len(fuzzy_results),
        "source": "fuzzy_match"
    })


@router.get("/ingredient/{additive_id}")
async def get_additive(additive_id: str):
    """
    Get a specific additive record by its ID (e.g. E621, FSSAI-001).
    """
    _, _, _, _, get_additive_by_id = _get_additive_service()
    if not get_additive_by_id:
        return JSONResponse({"error": "Additive database unavailable"}, status_code=503)

    record = get_additive_by_id(additive_id.upper())
    if not record:
        raise HTTPException(status_code=404, detail=f"Additive '{additive_id}' not found in database")
    return JSONResponse(record)


class ScanIngredientsRequest(BaseModel):
    ingredients_text: str
    persona: str = "general"


@router.post("/scan")
async def scan_ingredients_endpoint(request: Request, payload: ScanIngredientsRequest):
    """
    Scan a full ingredients list against the verified additive database.
    Returns matched additives grouped by safety tier.

    This is the core intelligence endpoint — powers the ingredient spotlight
    in the scan results.
    """
    _, _, get_ingredient_risk_summary, _, _ = _get_additive_service()
    if not get_ingredient_risk_summary:
        return JSONResponse({"error": "Additive database unavailable"}, status_code=503)

    if not payload.ingredients_text or len(payload.ingredients_text.strip()) < 3:
        return JSONResponse({"error": "ingredients_text is too short"}, status_code=400)

    result = get_ingredient_risk_summary(
        ingredients_text=payload.ingredients_text,
        persona=payload.persona,
    )
    return JSONResponse(result)


@router.get("/stats")
async def additive_db_stats():
    """
    Return database statistics: total additives, breakdown by safety tier and category.
    """
    _, _, _, get_all_additives, _ = _get_additive_service()
    if not get_all_additives:
        return JSONResponse({"error": "Additive database unavailable"}, status_code=503)

    all_additives = get_all_additives()
    total = len(all_additives)
    by_tier = {"SAFE": 0, "CAUTION": 0, "AVOID": 0}
    by_category = {}
    by_fssai = {}

    for a in all_additives:
        tier = a.get("safety_tier", "UNKNOWN")
        by_tier[tier] = by_tier.get(tier, 0) + 1

        cat = a.get("category", "other")
        by_category[cat] = by_category.get(cat, 0) + 1

        fssai = a.get("fssai_status", "unknown")
        by_fssai[fssai] = by_fssai.get(fssai, 0) + 1

    return JSONResponse({
        "total_additives": total,
        "by_safety_tier": by_tier,
        "by_category": by_category,
        "by_fssai_status": by_fssai,
        "moat_status": (
            "🔴 Early stage (<100)" if total < 100 else
            "🟡 Growing (100-300)" if total < 300 else
            "🟢 Defensible (300+)"
        ),
        "database_version": "1.0.0",
    })
