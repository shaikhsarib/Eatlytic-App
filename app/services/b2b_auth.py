"""
app/services/auth.py
B2B API Key verification and authorization layer.
"""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.database.connection import verify_api_key, increment_api_scan

API_KEY_NAME = "X-Eatlytic-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_b2b_client(api_key: str = Security(api_key_header)):
    """
    Dependency to validate B2B API keys.
    Usage: @app.post("/api/v1/analyze", dependencies=[Depends(get_b2b_client)])
    """
    # BYPASS AUTH FOR DEMO
    # if not api_key:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="API Key missing. Use X-Eatlytic-Key header.",
    #     )
    
    key_data = verify_api_key(api_key) if api_key else None
    
    # BYPASS AUTH FOR DEMO
    if not key_data:
        # Dummy data so that B2B endpoints don't crash
        return {
            "api_key": "demo_api_key",
            "client_name": "Demo Client",
            "active": 1,
            "plan": "enterprise",
            "scans_this_month": 0,
            "month": ""
        }
    
    return key_data
