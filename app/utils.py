"""
app/utils.py
Legacy transitional import stub for app.core.security.
Allows backward compatibility for external tools or untracked legacy modules.
"""
from app.core.security import (
    sign_value,
    get_device_key,
    sanitize_text,
    verify_admin,
    COOKIE_SECRET,
)
