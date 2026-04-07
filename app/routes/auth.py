"""
app/routes/auth.py
Auth endpoints: OTP request, OTP verify, logout, profile.
"""
import logging
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import JSONResponse
from app.services.auth import (
    send_email_otp, verify_email_otp,
    create_session, revoke_session, get_user_from_token,
)
from app.models.db import db_conn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/request-otp")
async def request_otp(email: str = Form(...)):
    otp = send_email_otp(email)
    return JSONResponse({
        "sent"   : True,
        "message": "OTP sent. Check your email.",
        # Note: _dev_otp removed. OTP is only logged server-side.
    })


@router.post("/verify-otp")
async def verify_otp(request: Request, email: str = Form(...), otp: str = Form(...)):
    user = verify_email_otp(email, otp)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired OTP")
    device_hint = request.headers.get("user-agent", "")[:100]
    token       = create_session(user["id"], device_hint)
    return JSONResponse({
        "token"   : token,
        "user_id" : user["id"],
        "email"   : user.get("email", ""),
        "is_pro"  : bool(user.get("is_pro", 0)),
        "message" : "Login successful",
    })


@router.post("/logout")
async def logout(request: Request):
    auth  = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
    if token:
        revoke_session(token)
    return JSONResponse({"logged_out": True})


@router.get("/me")
async def get_me(request: Request):
    auth  = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
    user  = get_user_from_token(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse({
        "user_id"    : user["id"],
        "email"      : user.get("email", ""),
        "name"       : user.get("name", ""),
        "is_pro"     : bool(user.get("is_pro", 0)),
        "streak_days": user.get("streak_days", 0),
        "persona"    : user.get("persona", "General Adult"),
        "language"   : user.get("language", "en"),
    })


@router.put("/profile")
async def update_profile(
    request : Request,
    name    : str   = Form(""),
    persona : str   = Form("General Adult"),
    language: str   = Form("en"),
    tdee    : float = Form(2000),
):
    auth  = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
    user  = get_user_from_token(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET name=?, persona=?, language=?, tdee=? WHERE id=?",
            (name, persona, language, tdee, user["id"])
        )
    return JSONResponse({"updated": True})
