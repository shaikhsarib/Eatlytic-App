"""
app/routes/food_db.py
Proprietary food database endpoints.
"""
import logging
from fastapi import APIRouter, Request, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from app.models.db import db_conn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/food-db", tags=["food-db"])


@router.get("/search")
async def search_food_db(
    request: Request,
    q      : str = Query("", min_length=2),
    limit  : int = Query(10, le=50),
):
    if not q or len(q.strip()) < 2:
        return JSONResponse({"products": [], "source": "none"})

    with db_conn() as conn:
        rows = conn.execute(
            """SELECT name,brand,category,calories_100g,protein_100g,carbs_100g,
                      fat_100g,sodium_100g,fiber_100g,sugar_100g,
                      eatlytic_score,verified,scan_count
               FROM food_products
               WHERE (name LIKE ? OR brand LIKE ?) AND verified=1
               ORDER BY scan_count DESC, eatlytic_score DESC
               LIMIT ?""",
            (f"%{q}%", f"%{q}%", limit)
        ).fetchall()

    if rows:
        products = [dict(r) for r in rows]
        return JSONResponse({"products": products, "source": "eatlytic_db",
                             "count": len(products)})

    try:
        import httpx
        async with httpx.AsyncClient(timeout=8) as hc:
            resp = await hc.get(
                "https://world.openfoodfacts.org/cgi/search.pl",
                params={"search_terms": q, "action": "process", "json": 1,
                        "page_size": limit,
                        "fields": "product_name,brands,nutriments,categories_tags"}
            )
        data     = resp.json()
        products = []
        for p in data.get("products", []):
            n = p.get("nutriments", {})
            products.append({
                "name"         : p.get("product_name", ""),
                "brand"        : p.get("brands", ""),
                "category"     : ", ".join((p.get("categories_tags") or [])[:2]),
                "calories_100g": round(n.get("energy-kcal_100g", 0), 1),
                "protein_100g" : round(n.get("proteins_100g", 0), 1),
                "carbs_100g"   : round(n.get("carbohydrates_100g", 0), 1),
                "fat_100g"     : round(n.get("fat_100g", 0), 1),
                "sodium_100g"  : round(n.get("sodium_100g", 0) * 1000, 1),
                "fiber_100g"   : round(n.get("fiber_100g", 0), 1),
                "sugar_100g"   : round(n.get("sugars_100g", 0), 1),
                "eatlytic_score": 0,
                "verified"     : 0,
                "source"       : "openfoodfacts",
            })
        return JSONResponse({"products": products, "source": "openfoodfacts",
                             "count": len(products)})
    except Exception as exc:
        logger.warning("OpenFoodFacts fallback failed: %s", exc)
        return JSONResponse({"products": [], "source": "unavailable"})


@router.get("/product/{product_id}")
async def get_product(request: Request, product_id: int):
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM food_products WHERE id=?", (product_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")
    return JSONResponse(dict(row))


@router.post("/verify/{product_id}")
async def verify_product(
    request    : Request,
    product_id : int,
    admin_token: str = Form(...),
):
    import os
    _expected = os.environ.get("ADMIN_TOKEN")
    if not _expected:
        raise HTTPException(status_code=500, detail="Server misconfiguration: ADMIN_TOKEN not set.")
    import hmac as _hmac_mod
    if not _hmac_mod.compare_digest(admin_token, _expected):
        raise HTTPException(status_code=403, detail="Invalid admin token")

    with db_conn() as conn:
        row = conn.execute("SELECT id FROM food_products WHERE id=?", (product_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Product not found")
        conn.execute(
            "UPDATE food_products SET verified=1, verified_at=datetime('now'), verified_by='admin' WHERE id=?",
            (product_id,)
        )
    return JSONResponse({"verified": True, "product_id": product_id})


@router.get("/stats")
async def db_stats(request: Request):
    with db_conn() as conn:
        total    = conn.execute("SELECT COUNT(*) FROM food_products").fetchone()[0]
        verified = conn.execute("SELECT COUNT(*) FROM food_products WHERE verified=1").fetchone()[0]
        top_cats = conn.execute(
            """SELECT category, COUNT(*) c FROM food_products
               WHERE category != '' GROUP BY category ORDER BY c DESC LIMIT 5"""
        ).fetchall()
        popular  = conn.execute(
            """SELECT name, brand, scan_count, eatlytic_score
               FROM food_products ORDER BY scan_count DESC LIMIT 10"""
        ).fetchall()

    return JSONResponse({
        "total_products"   : total,
        "verified_products": verified,
        "verification_rate": f"{round(verified/total*100, 1)}%" if total else "0%",
        "top_categories"   : [{"category": r[0], "count": r[1]} for r in top_cats],
        "most_scanned"     : [dict(r) for r in popular],
        "moat_status"     : (
            "🔴 Early stage (<1K)" if total < 1000 else
            "🟡 Growing (1K-10K)"  if total < 10000 else
            "🟢 Defensible (10K+)"
        ),
    })
