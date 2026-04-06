"""
app/services/payments.py
Razorpay payment integration.
"""
import os
import logging
import hashlib
import hmac
import datetime
from app.models.db import db_conn
UTC = datetime.timezone.utc

logger = logging.getLogger(__name__)

RAZORPAY_KEY_ID     = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")
PRO_AMOUNT_PAISE    = 19900


def create_order(user_id: str, device_key: str = "") -> dict:
    """Create a Razorpay order and return order details."""
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise RuntimeError("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET env vars required")
    try:
        import razorpay
        client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    except ImportError:
        raise RuntimeError("razorpay package not installed")

    order = client.order.create({
        "amount"   : PRO_AMOUNT_PAISE,
        "currency" : "INR",
        "receipt"  : f"eat_{user_id[:8]}_{datetime.datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "notes"    : {"user_id": user_id, "product": "eatlytic_pro"},
    })
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO payments(user_id,device_key,razorpay_order_id,amount_paise,status) VALUES(?,?,?,?,?)",
            (user_id, device_key, order["id"], PRO_AMOUNT_PAISE, "created")
        )
    return {
        "order_id": order["id"],
        "amount"  : PRO_AMOUNT_PAISE,
        "currency": "INR",
        "key_id"  : RAZORPAY_KEY_ID,
    }


def verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Verify Razorpay HMAC signature."""
    if not RAZORPAY_KEY_SECRET:
        raise RuntimeError("RAZORPAY_KEY_SECRET not set — cannot verify payment signature.")
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def activate_pro_after_payment(order_id: str, payment_id: str, signature: str) -> dict:
    """Verify payment and activate Pro for user."""
    if not verify_signature(order_id, payment_id, signature):
        raise ValueError("Invalid payment signature — possible tampering")
    expires = (datetime.datetime.now(UTC) + datetime.timedelta(days=31)).isoformat()
    with db_conn() as conn:
        row = conn.execute(
            "SELECT user_id, device_key FROM payments WHERE razorpay_order_id=?", (order_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Order {order_id} not found")
        user_id    = row["user_id"]
        device_key = row["device_key"]
        conn.execute(
            "UPDATE payments SET razorpay_payment_id=?,razorpay_signature=?,status='paid',paid_at=datetime('now') WHERE razorpay_order_id=?",
            (payment_id, signature, order_id)
        )
        if user_id:
            conn.execute(
                "UPDATE users SET is_pro=1, pro_expires=? WHERE id=?",
                (expires, user_id)
            )
        if device_key:
            conn.execute(
                "UPDATE devices SET is_pro=1 WHERE device_key=?",
                (device_key,)
            )
    return {"success": True, "user_id": user_id, "expires": expires}


def get_payment_status(order_id: str) -> dict | None:
    """Get payment status for an order."""
    with db_conn() as conn:
        row = conn.execute(
            "SELECT razorpay_order_id, razorpay_payment_id, status, amount_paise, currency FROM payments WHERE razorpay_order_id=?",
            (order_id,)
        ).fetchone()
    return dict(row) if row else None
