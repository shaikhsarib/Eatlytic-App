"""
app/services/auth.py
B2B API Key verification and authorization layer.
"""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.models.db import verify_api_key, increment_api_scan

API_KEY_NAME = "X-Eatlytic-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_b2b_client(api_key: str = Security(api_key_header)):
    """
    Dependency to validate B2B API keys.
    Usage: @app.post("/api/v1/analyze", dependencies=[Depends(get_b2b_client)])
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing. Use X-Eatlytic-Key header.",
        )
    
    key_data = verify_api_key(api_key)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API Key.",
        )
    
    # Track usage (optional: add rate limiting logic here)
    increment_api_scan(api_key)
    
    return key_data
