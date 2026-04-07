"""
app/routes/payments.py
Payment endpoints: create Razorpay order, verify payment, status.
"""
import logging
import hashlib
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import JSONResponse
from app.services.auth import get_user_from_token
from app.models.db import db_conn
from app.services.payments import create_order, activate_pro_after_payment, get_payment_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/payments", tags=["payments"])


def _get_user_or_device(request: Request) -> tuple:
    auth       = request.headers.get("Authorization", "")
    token      = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
    user       = get_user_from_token(token) if token else None
    user_id    = user["id"] if user else None
    ip         = request.client.host if request.client else "unknown"
    ua         = request.headers.get("user-agent", "")
    device_key = hashlib.md5(f"{ip}:{ua}".encode()).hexdigest()[:16]
    return user_id, device_key


@router.post("/create-order")
async def create_order_endpoint(request: Request):
    user_id, device_key = _get_user_or_device(request)
    if not user_id:
        raise HTTPException(status_code=401,
            detail="Login required to create payment order. POST /auth/request-otp first.")
    try:
        order = create_order(user_id, device_key)
        return JSONResponse(order)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("create_order failed: %s", exc)
        raise HTTPException(status_code=500, detail="Payment service error")


@router.post("/verify")
async def verify_payment(
    request              : Request,
    razorpay_order_id   : str = Form(...),
    razorpay_payment_id : str = Form(...),
    razorpay_signature  : str = Form(...),
):
    try:
        result = activate_pro_after_payment(
            razorpay_order_id, razorpay_payment_id, razorpay_signature
        )
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("verify_payment failed: %s", exc)
        raise HTTPException(status_code=500, detail="Payment verification failed")


@router.get("/status/{order_id}")
async def payment_status(request: Request, order_id: str):
    try:
        status = get_payment_status(order_id)
        if not status:
            raise HTTPException(status_code=404, detail="Order not found")
        return JSONResponse(status)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
