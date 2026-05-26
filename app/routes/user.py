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
    get_daily_macro_totals,
    create_user,
    authenticate_user,
    create_session,
    get_session,
    delete_session,
    sync_local_history,
    create_organization,
    get_org_by_admin,
    generate_api_key,
    get_org_api_keys,
    revoke_api_key
)
from app.utils import get_device_key

logger = logging.getLogger(__name__)
router = APIRouter(tags=["user"])

FREE_SCAN_LIMIT = int(os.environ.get("FREE_SCAN_LIMIT", "10"))

# ── USER AUTHENTICATION ENDPOINTS ──────────────────────────────────────
@router.post("/api/v1/auth/register")
async def api_register(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(""),
    device_key: str = Form(None)
):
    try:
        user_id = create_user(
            email=email.strip().lower(),
            password=password,
            name=name.strip()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    
    dev_key = device_key or request.cookies.get("device_key")
    if dev_key:
        sync_local_history(dev_key, user_id)
        
    token = create_session(user_id, device_hint=request.headers.get("user-agent", ""))
    response.set_cookie(key="session_token", value=token, max_age=2592000, httponly=True)
    
    return {
        "status": "registered",
        "token": token,
        "user": {
            "id": user_id,
            "email": email.strip().lower(),
            "name": name.strip(),
            "is_pro": False
        }
    }

@router.post("/api/v1/auth/login")
async def api_login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    device_key: str = Form(None)
):
    user_data = authenticate_user(email.strip().lower(), password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    
    user_id = user_data["id"]
    dev_key = device_key or request.cookies.get("device_key")
    if dev_key:
        sync_local_history(dev_key, user_id)
        
    token = create_session(user_id, device_hint=request.headers.get("user-agent", ""))
    response.set_cookie(key="session_token", value=token, max_age=2592000, httponly=True)
    
    return {
        "status": "logged_in",
        "token": token,
        "user": {
            "id": user_id,
            "email": user_data["email"],
            "name": user_data["name"],
            "is_pro": bool(user_data["is_pro"])
        }
    }

@router.post("/api/v1/auth/logout")
async def api_logout(request: Request, response: Response):
    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if token:
        delete_session(token)
    response.delete_cookie(key="session_token")
    return {"status": "logged_out", "message": "Successfully logged out."}

@router.get("/api/v1/auth/session")
async def api_session(request: Request):
    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="No session token found.")
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return {
        "status": "active",
        "user": {
            "id": session["user_id"],
            "email": session["email"],
            "name": session["name"],
            "is_pro": bool(session["is_pro"]),
            "persona": session["persona"],
            "language": session["language"]
        }
    }

# ── USER HISTORIES & MEAL LOGS ─────────────────────────────────────────
@router.get("/api/v1/meal-log")
async def get_meal_log(request: Request, response: Response):
    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if token:
        session = get_session(token)
        if session:
            date_str = datetime.date.today().isoformat()
            with db_conn() as conn:
                row = conn.execute("""
                    SELECT SUM(calories) as total_cal, SUM(protein) as total_pro, 
                           SUM(carbs) as total_car, SUM(fat) as total_fat
                    FROM daily_logs
                    WHERE user_id = ? AND log_date = ?
                """, (session["user_id"], date_str)).fetchone()
            if row and row["total_cal"] is not None:
                return {
                    "calories": row["total_cal"],
                    "protein": row["total_pro"],
                    "carbs": row["total_car"],
                    "fat": row["total_fat"]
                }
            return {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
            
    device_key = get_device_key(request, response)
    return get_daily_macro_totals(device_key)

@router.post("/api/v1/meal-log")
async def post_meal_log(request: Request, response: Response, meal_data: dict = Body(...)):
    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    user_id = None
    if token:
        session = get_session(token)
        if session:
            user_id = session["user_id"]
            
    device_key = get_device_key(request, response)
    date_str = datetime.date.today().isoformat()
    
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO daily_logs (device_key, user_id, log_date, meal_name, calories, protein, carbs, fat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            device_key, user_id, date_str, meal_data.get("product_name", "Meal"),
            meal_data.get("calories", 0), meal_data.get("protein", 0),
            meal_data.get("carbs", 0), meal_data.get("fat", 0)
        ))
        
    if user_id:
        with db_conn() as conn:
            row = conn.execute("""
                SELECT SUM(calories) as total_cal, SUM(protein) as total_pro, 
                       SUM(carbs) as total_car, SUM(fat) as total_fat
                FROM daily_logs
                WHERE user_id = ? AND log_date = ?
            """, (user_id, date_str)).fetchone()
        if row and row["total_cal"] is not None:
            return {
                "status": "logged",
                "totals": {
                    "calories": row["total_cal"],
                    "protein": row["total_pro"],
                    "carbs": row["total_car"],
                    "fat": row["total_fat"]
                }
            }
            
    return {"status": "logged", "totals": get_daily_macro_totals(device_key)}

@router.post("/api/v1/report-error")
async def report_scan_error(
    request: Request,
    response: Response,
    scan_id: int = Form(...),
    note: str = Form(""),
):
    device_key = get_device_key(request, response)
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO scan_reports (scan_id, device_key, note, reported_at) "
                "VALUES (?, ?, ?, ?)",
                (scan_id, device_key, note[:500], datetime.datetime.utcnow().isoformat()),
            )
    except Exception as e:
        logger.warning("report_scan_error db write failed: %s", e)
    return {"status": "reported", "message": "Thank you — our team will review this scan."}

@router.get("/api/v1/history")
async def get_history(request: Request, response: Response):
    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if token:
        session = get_session(token)
        if session:
            with db_conn() as conn:
                rows = conn.execute("""
                    SELECT id, product_name, score, verdict, scanned_at, brand, category
                    FROM scans 
                    WHERE user_id = ? 
                    ORDER BY scanned_at DESC 
                    LIMIT 15
                """, (session["user_id"],)).fetchall()
            return [dict(r) for r in rows]
            
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

# ── DEVELOPER PORTAL ENDPOINTS ───────────────────────────────────────
@router.post("/api/v1/developer/organization")
async def api_create_organization(request: Request, name: str = Form(None)):
    if not name:
        try:
            body = await request.json()
            name = body.get("name")
        except Exception:
            pass
    if not name:
        raise HTTPException(status_code=422, detail="Missing parameter: name")

    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    
    admin_id = session["user_id"]
    existing = get_org_by_admin(admin_id)
    if existing:
        return {"status": "exists", "organization": existing}
    
    org_id = create_organization(admin_id, name.strip())
    return {
        "status": "created",
        "organization": {
            "id": org_id,
            "name": name.strip(),
            "admin_id": admin_id,
            "plan": "business"
        }
    }

@router.get("/api/v1/developer/dashboard")
async def api_get_developer_dashboard(request: Request):
    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    
    admin_id = session["user_id"]
    org = get_org_by_admin(admin_id)
    if not org:
        return {"status": "no_organization"}
    
    keys = get_org_api_keys(org["id"])
    return {
        "status": "success",
        "organization": org,
        "api_keys": keys
    }

@router.post("/api/v1/developer/keys")
async def api_generate_api_key(request: Request, client_name: str = Form(None), plan: str = Form("business")):
    if not client_name:
        try:
            body = await request.json()
            client_name = body.get("client_name")
            plan = body.get("plan", plan)
        except Exception:
            pass
    if not client_name:
        raise HTTPException(status_code=422, detail="Missing parameter: client_name")

    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    
    admin_id = session["user_id"]
    org = get_org_by_admin(admin_id)
    if not org:
        raise HTTPException(status_code=400, detail="You must register an organization first.")
    
    api_key = generate_api_key(client_name.strip(), org["id"], plan)
    return {
        "status": "generated",
        "api_key": api_key,
        "client_name": client_name.strip(),
        "plan": plan
    }

@router.post("/api/v1/developer/keys/revoke")
async def api_revoke_api_key(request: Request, api_key: str = Form(None)):
    if not api_key:
        try:
            body = await request.json()
            api_key = body.get("api_key")
        except Exception:
            pass
    if not api_key:
        raise HTTPException(status_code=422, detail="Missing parameter: api_key")

    token = request.headers.get("X-Eatlytic-Session") or request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required.")
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    
    admin_id = session["user_id"]
    org = get_org_by_admin(admin_id)
    if not org:
        raise HTTPException(status_code=400, detail="Access denied.")
    
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE api_key = ? AND organization_id = ?", (api_key, org["id"])).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="API Key not found or belongs to another organization.")
    
    revoke_api_key(api_key)
    return {"status": "revoked", "message": "API Key has been successfully deactivated."}

