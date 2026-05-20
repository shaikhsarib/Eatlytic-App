import os
import hmac
import hashlib
import uuid
import secrets
import logging
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# --- SECURITY CONFIG ---
COOKIE_SECRET = os.environ.get("COOKIE_SECRET")
if not COOKIE_SECRET:
    COOKIE_SECRET = secrets.token_urlsafe(32)

def sign_value(value: str) -> str:
    """Create a signature for a value."""
    return hmac.new(COOKIE_SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()

def get_device_key(request: Request, response: Response = None) -> str:
    """
    Get or set a secure, signed device ID in a cookie.
    """
    cookie_val: str = request.cookies.get("eatlytic_did")
    
    if cookie_val and ":" in cookie_val:
        try:
            did, sig = cookie_val.split(":", 1)
            if hmac.compare_digest(sign_value(did), sig):
                return did
        except ValueError:
            pass

    # No valid cookie — generate a new one
    new_did: str = str(uuid.uuid4())
    if response:
        sig: str = sign_value(new_did)
        response.set_cookie(
            "eatlytic_did", 
            f"{new_did}:{sig}", 
            httponly=True, 
            secure=True, 
            samesite="strict", 
            max_age=365*86400 # 1 year
        )
    return new_did

def sanitize_text(text: str, max_len: int = 5000) -> str:
    if not text: return ""
    # Remove null bytes and limit length
    clean: str = text.replace("\x00", "").strip()
    return clean[:max_len]

def verify_admin(request: Request) -> bool:
    ADMIN_TOKEN: str = os.environ.get("ADMIN_TOKEN", "")
    token: str = request.headers.get("X-Admin-Token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required.")
    return True
