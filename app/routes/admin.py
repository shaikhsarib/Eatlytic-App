from fastapi import APIRouter, Request, HTTPException, Depends, Form
from app.models.db import get_unverified_scans, apply_correction, db_conn
from app.utils import verify_admin
import os
import secrets
import datetime

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(verify_admin)])

@router.get("/unverified")
async def list_unverified():
    return get_unverified_scans()

@router.post("/correct/{scan_id}")
async def apply_scan_correction(scan_id: int, correction: dict):
    apply_correction(scan_id, correction)
    return {"status": "corrected", "scan_id": scan_id}

@router.post("/create-api-key")
async def create_api_key_endpoint(client_name: str = Form(...), plan: str = Form("business")):
    key = "eak_" + secrets.token_urlsafe(32)
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys(api_key, client_name, plan, scans_this_month, month, active) VALUES(?,?,?,?,?,1)",
            (key, client_name, plan, 0, month_key),
        )
    return {"api_key": key, "client": client_name, "plan": plan}
