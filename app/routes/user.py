import os
import logging
import datetime
from fastapi import APIRouter, Request, Response, Form, HTTPException, Body
from app.models.db import (
    get_device_history,
    get_scan_by_id,
    delete_user_data,
    db_conn,
    add_meal_log,
    get_daily_macro_totals
)
from app.utils import get_device_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["user"])

FREE_SCAN_LIMIT = int(os.environ.get("FREE_SCAN_LIMIT", "10"))

@router.get("/api/v1/meal-log")
async def get_meal_log(request: Request, response: Response):
    device_key = get_device_key(request, response)
    return get_daily_macro_totals(device_key)

@router.post("/api/v1/meal-log")
async def post_meal_log(request: Request, response: Response, meal_data: dict = Body(...)):
    device_key = get_device_key(request, response)
    add_meal_log(device_key, meal_data)
    return {"status": "logged", "totals": get_daily_macro_totals(device_key)}


@router.post("/api/v1/report-error")
async def report_scan_error(
    request: Request,
    response: Response,
    scan_id: int = Form(...),
    note: str = Form(""),
):
    """Accept user feedback about inaccurate scan results."""
    device_key = get_device_key(request, response)
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO scan_reports (scan_id, device_key, note, reported_at) "
                "VALUES (?, ?, ?, ?)",
                (scan_id, device_key, note[:500], datetime.datetime.utcnow().isoformat()),
            )
    except Exception as e:
        logger.warning("report_scan_error db write failed (table may not exist): %s", e)
    return {"status": "reported", "message": "Thank you — our team will review this scan."}

@router.get("/api/v1/history")
async def get_history(request: Request, response: Response):
    device_key = get_device_key(request, response)
    return get_device_history(device_key)

@router.delete("/api/v1/user/delete")
async def erase_user_data(request: Request, response: Response):
    device_key = get_device_key(request, response)
    delete_user_data(device_key)
    return {"status": "erased", "message": "All your data has been permanently deleted."}

@router.get("/scan-status")
@router.get("/scan-quota")
async def scan_status(request: Request, response: Response):
    device_key = get_device_key(request, response)
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM devices WHERE device_key=?", (device_key,)).fetchone()
    
    if not row:
        return {"scans_used": 0, "scans_remaining": FREE_SCAN_LIMIT, "is_pro": False, "limit": FREE_SCAN_LIMIT}
    
    u = dict(row)
    if u["month"] != month_key:
        return {"scans_used": 0, "scans_remaining": 9999 if u["is_pro"] else FREE_SCAN_LIMIT, "is_pro": bool(u["is_pro"]), "limit": FREE_SCAN_LIMIT}
    
    used = u["scan_count"]
    return {
        "scans_used": used,
        "scans_remaining": 9999 if u["is_pro"] else max(0, FREE_SCAN_LIMIT - used),
        "is_pro": bool(u["is_pro"]),
        "limit": FREE_SCAN_LIMIT
    }

@router.post("/api/v1/duel")
async def product_duel(request: Request, scan_a_id: int = Form(...), scan_b_id: int = Form(...)):
    from app.services.duel_service import run_duel
    prod_a = get_scan_by_id(scan_a_id)
    prod_b = get_scan_by_id(scan_b_id)
    if not prod_a or not prod_b:
        raise HTTPException(status_code=404, detail="One or more products not found in history.")
    return run_duel(prod_a, prod_b, persona=prod_a.get("persona", "general"))
