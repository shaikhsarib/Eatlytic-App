"""
app/services/auth.py
Auth service: OTP management, sessions, user management, scan quota, streaks.
"""
import os
import hmac
import logging
import secrets
import datetime
import threading
import uuid
import json
from app.database.connection import db_conn
UTC = datetime.timezone.utc

logger = logging.getLogger(__name__)

SESSION_TTL_DAYS = 30
FREE_SCAN_LIMIT  = int(os.environ.get("FREE_SCAN_LIMIT", "10"))
OTP_MAX_ATTEMPTS = 5        # lockout after 5 wrong guesses
OTP_EXPIRY_MINS  = 10


def send_email_otp(email: str) -> str:
    """Generate and store a 6-digit OTP. Thread-safe & Persistent."""
    key = email.lower().strip()
    now = datetime.datetime.now(UTC)
    otp = str(secrets.randbelow(900000) + 100000)
    expires = now + datetime.timedelta(minutes=OTP_EXPIRY_MINS)
    
    # Store in persistent SQLite cache
    with db_conn() as conn:
        conn.execute("DELETE FROM ai_cache WHERE cache_key = ?", (f"otp:{key}",))
        conn.execute(
            "INSERT OR REPLACE INTO ai_cache(cache_key, result_json) VALUES(?,?)",
            (f"otp:{key}", json.dumps({
                "otp": otp,
                "expires": expires.isoformat(),
                "attempts": 0
            }))
        )
    logger.info("OTP issued for %s (dev mode — remove in prod)", email)
    return otp


def verify_email_otp(email: str, otp: str):
    """Verify OTP. Returns user on success, None on failure. Thread-safe & Persistent with lockout."""
    key = email.lower().strip()
    now = datetime.datetime.now(UTC)
    
    with db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT result_json FROM ai_cache WHERE cache_key=?", (f"otp:{key}",)).fetchone()
        if not row:
            return None
        
        try:
            entry = json.loads(row["result_json"])
        except Exception:
            return None
            
        stored = entry.get("otp")
        expires_str = entry.get("expires")
        attempts = entry.get("attempts", 0)
        
        try:
            expires = datetime.datetime.fromisoformat(expires_str)
        except Exception:
            conn.execute("DELETE FROM ai_cache WHERE cache_key=?", (f"otp:{key}",))
            return None
            
        if now > expires:
            conn.execute("DELETE FROM ai_cache WHERE cache_key=?", (f"otp:{key}",))
            return None
            
        if attempts >= OTP_MAX_ATTEMPTS:
            conn.execute("DELETE FROM ai_cache WHERE cache_key=?", (f"otp:{key}",))
            logger.warning("OTP brute-force lockout for %s after %d attempts", email, attempts)
            return None
            
        # Timing-safe compare
        if not hmac.compare_digest(stored, otp.strip()):
            entry["attempts"] = attempts + 1
            conn.execute(
                "UPDATE ai_cache SET result_json=? WHERE cache_key=?",
                (json.dumps(entry), f"otp:{key}")
            )
            return None
            
        conn.execute("DELETE FROM ai_cache WHERE cache_key=?", (f"otp:{key}",))
        
    return _get_or_create_user(email=email)


def _get_or_create_user(email=None, phone=None, name=""):
    if not email and not phone:
        raise ValueError("email or phone required")
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=?" if email else "SELECT * FROM users WHERE phone=?",
            (email or phone,)
        ).fetchone()
        if row:
            return dict(row)
        uid = str(uuid.uuid4())
        conn.execute("INSERT INTO users(id,email,phone,name) VALUES(?,?,?,?)",
                     (uid, email, phone, name))
        return {"id": uid, "email": email, "phone": phone, "name": name,
                "is_pro": 0, "streak_days": 0, "scan_count_month": 0}


def create_session(user_id: str, device_hint: str = "") -> str:
    """Create a session token for a user."""
    token   = "eat_" + secrets.token_urlsafe(40)
    expires = (datetime.datetime.now(UTC) + datetime.timedelta(days=SESSION_TTL_DAYS)).isoformat()
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO sessions(token,user_id,expires_at,device_hint) VALUES(?,?,?,?)",
            (token, user_id, expires, device_hint)
        )
    return token


def revoke_session(token: str) -> None:
    """Revoke/delete a session token."""
    with db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))


def get_user_from_token(token: str):
    """Look up user by session token (returns None if invalid/expired)."""
    if not token:
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON s.user_id=u.id "
            "WHERE s.token=? AND s.expires_at>datetime('now')",
            (token,)
        ).fetchone()
    return dict(row) if row else None


def check_and_increment_scan_user(user_id: str, increment: bool = True) -> dict:
    """
    Check if user has remaining scan quota and increment.
    Returns quota status dict.
    """
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT is_pro, scan_month, scan_count_month FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not row:
            return {"allowed": False, "scans_used": 0, "scans_remaining": 0, "is_pro": False}
        if row["scan_month"] != month_key:
            conn.execute("UPDATE users SET scan_month=?, scan_count_month=0 WHERE id=?",
                         (month_key, user_id))
            count = 0
        else:
            count = row["scan_count_month"]
        if row["is_pro"]:
            if increment:
                conn.execute("UPDATE users SET scan_count_month=scan_count_month+1 WHERE id=?", (user_id,))
            return {"allowed": True, "scans_used": count + (1 if increment else 0), "scans_remaining": 9999, "is_pro": True}
        if count >= FREE_SCAN_LIMIT:
            return {"allowed": False, "scans_used": count, "scans_remaining": 0, "is_pro": False}
        if increment:
            conn.execute("UPDATE users SET scan_count_month=scan_count_month+1 WHERE id=?", (user_id,))
        new = count + (1 if increment else 0)
        return {"allowed": True, "scans_used": new, "scans_remaining": FREE_SCAN_LIMIT - new, "is_pro": False}


def update_streak_user(user_id: str) -> None:
    """Update scan streak for a user."""
    today     = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    with db_conn() as conn:
        row = conn.execute(
            "SELECT streak_days, last_scan_date FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not row or row["last_scan_date"] == today:
            return
        streak = (row["streak_days"] + 1) if row["last_scan_date"] == yesterday else 1
        conn.execute(
            "UPDATE users SET streak_days=?, last_scan_date=? WHERE id=?",
            (streak, today, user_id)
        )
