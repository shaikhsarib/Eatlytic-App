import json
import logging
import datetime
from app.database.connection import db_conn, _supabase, parse_utc_iso

logger = logging.getLogger(__name__)

def check_and_increment_scan(device_key: str, limit: int = 10, increment: bool = True) -> dict:
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM devices WHERE device_key=?", (device_key,)).fetchone()
        if not row:
            conn.execute("INSERT INTO devices(device_key, month, scan_count) VALUES(?,?,0)", (device_key, month_key))
            u = {"is_pro": 0, "month": month_key, "scan_count": 0, "pro_expires": None}
        else:
            u = dict(row)

        if u.get("is_pro") == 1:
            exp = u.get("pro_expires")
            if exp and parse_utc_iso(exp) < datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None):
                conn.execute("UPDATE devices SET is_pro=0 WHERE device_key=?", (device_key,))
                u["is_pro"] = 0

        if u["month"] != month_key:
            conn.execute("UPDATE devices SET month=?, scan_count=0 WHERE device_key=?", (month_key, device_key))
            u["scan_count"] = 0

        if u["is_pro"]:
            if increment: conn.execute("UPDATE devices SET scan_count=scan_count+1 WHERE device_key=?", (device_key,))
            return {"allowed": True, "scans_used": u["scan_count"] + (1 if increment else 0), "scans_remaining": 9999, "is_pro": True}

        if u["scan_count"] >= limit:
            return {"allowed": False, "scans_used": u["scan_count"], "scans_remaining": 0, "is_pro": False}

        if increment: conn.execute("UPDATE devices SET scan_count=scan_count+1 WHERE device_key=?", (device_key,))
        new_c = u["scan_count"] + (1 if increment else 0)
        return {"allowed": True, "scans_used": new_c, "scans_remaining": max(0, limit - new_c), "is_pro": False}

def purge_old_records():
    """DPDP Article 12: Delete inactive records after 90 days."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
    with db_conn() as conn:
        conn.execute("DELETE FROM devices WHERE last_scan_date < ? AND is_pro = 0", (cutoff,))
        conn.execute("DELETE FROM ai_cache WHERE created_at < datetime('now', '-30 days')")
    logger.info("DPDP cleanup executed.")

def delete_user_data(device_key: str):
    """Right to erasure."""
    with db_conn() as conn:
        conn.execute("DELETE FROM devices WHERE device_key=?", (device_key,))
        conn.execute("DELETE FROM scans WHERE device_key=?", (device_key,))
    if _supabase:
        try: _supabase.table("devices").delete().eq("device_key", device_key).execute()
        except: pass
    logger.info("User data erased for %s", device_key[:8])

def get_unverified_scans(limit: int = 50):
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM scans WHERE verified=0 ORDER BY scanned_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

def apply_correction(scan_id: int, correction: dict):
    with db_conn() as conn:
        conn.execute("UPDATE scans SET verified=1, correction_json=? WHERE id=?", (json.dumps(correction), scan_id))

def save_scan(device_key: str, data: dict, user_id: str = None) -> int:
    """Save a scan result to the database."""
    with db_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO scans (
                device_key, user_id, product_name, score, verdict,
                calories, protein, carbs, fat, sodium, fiber, sugar,
                persona, language, analysis_json, metadata_json,
                barcode, brand, category
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            device_key, user_id, data.get("product_name", "Unknown"),
            data.get("score", 0), data.get("verdict", ""),
            data.get("calories", 0), data.get("protein", 0),
            data.get("carbs", 0), data.get("fat", 0),
            data.get("sodium", 0), data.get("fiber", 0), data.get("sugar", 0),
            data.get("persona", "general"), data.get("language", "en"),
            json.dumps(data.get("analysis_json", data)),
            json.dumps(data.get("metadata", {})),
            data.get("barcode"), data.get("brand"), data.get("category")
        ))
        return cursor.lastrowid

def get_device_history(device_key: str, limit: int = 15):
    """Retrieve recent scans for a specific device."""
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT id, product_name, score, verdict, scanned_at, brand, category
            FROM scans 
            WHERE device_key = ? 
            ORDER BY scanned_at DESC 
            LIMIT ?
        """, (device_key, limit)).fetchall()
    return [dict(r) for r in rows]

def get_scan_by_id(scan_id: int):
    """Retrieve full scan details by ID."""
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if not row: return None
    d = dict(row)
    for field in ["analysis_json", "metadata_json", "correction_json"]:
        if d.get(field):
            try: d[field] = json.loads(d[field])
            except: d[field] = {}
    return d

def add_meal_log(device_key: str, data: dict):
    """Log a meal to the daily diary."""
    date_str = datetime.date.today().isoformat()
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO daily_logs (device_key, log_date, meal_name, calories, protein, carbs, fat)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            device_key, date_str, data.get("product_name", "Meal"),
            data.get("calories", 0), data.get("protein", 0),
            data.get("carbs", 0), data.get("fat", 0)
        ))

def get_daily_macro_totals(device_key: str):
    """Aggregate macros for today."""
    date_str = datetime.date.today().isoformat()
    with db_conn() as conn:
        row = conn.execute("""
            SELECT SUM(calories) as total_cal, SUM(protein) as total_pro, 
                   SUM(carbs) as total_car, SUM(fat) as total_fat
            FROM daily_logs
            WHERE device_key = ? AND log_date = ?
        """, (device_key, date_str)).fetchone()
    if not row or row["total_cal"] is None:
        return {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    return {
        "calories": row["total_cal"],
        "protein": row["total_pro"],
        "carbs": row["total_car"],
        "fat": row["total_fat"]
    }

def hash_password(password: str) -> str:
    import hashlib
    import os
    salt = os.urandom(16).hex()
    pbkdf_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return f"{salt}:{pbkdf_hash}"

def verify_password(stored_password_hash: str, password: str) -> bool:
    import hashlib
    if not stored_password_hash or ":" not in stored_password_hash:
        return False
    salt, original_hash = stored_password_hash.split(":", 1)
    pbkdf_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return pbkdf_hash == original_hash

def create_user(email: str, password: str, name: str = "", persona: str = "General Adult", language: str = "en") -> str:
    import uuid
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO users (id, email, password_hash, name, persona, language)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, email, pw_hash, name, persona, language))
    return user_id

def authenticate_user(email: str, password: str) -> dict:
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        return None
    user_dict = dict(row)
    if verify_password(user_dict.get("password_hash"), password):
        user_dict.pop("password_hash", None)
        return user_dict
    return None

def create_session(user_id: str, device_hint: str = "") -> str:
    import os
    token = os.urandom(24).hex()
    expires_at = (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(days=30)).isoformat()
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO sessions (token, user_id, expires_at, device_hint)
            VALUES (?, ?, ?, ?)
        """, (token, user_id, expires_at, device_hint))
    return token

def get_session(token: str) -> dict:
    with db_conn() as conn:
        row = conn.execute("""
            SELECT s.*, u.email, u.name, u.is_pro, u.persona, u.language, u.tdee
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token,)).fetchone()
    if not row:
        return None
    session_dict = dict(row)
    if parse_utc_iso(session_dict["expires_at"]) < datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None):
        delete_session(token)
        return None
    return session_dict

def delete_session(token: str) -> None:
    with db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

def sync_local_history(device_key: str, user_id: str) -> None:
    with db_conn() as conn:
        conn.execute("UPDATE scans SET user_id = ? WHERE device_key = ? AND user_id IS NULL", (user_id, device_key))
        conn.execute("UPDATE daily_logs SET user_id = ? WHERE device_key = ? AND user_id IS NULL", (user_id, device_key))
        conn.execute("UPDATE devices SET user_id = ? WHERE device_key = ?", (user_id, device_key))
    try:
        sync_local_history_supabase(device_key, user_id)
    except Exception as e:
        logger.warning("Failed calling sync_local_history_supabase: %s", e)

def sync_local_history_supabase(device_key: str, user_id: str) -> None:
    if not _supabase:
        return
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _sync():
        _supabase.table("devices").update({"user_id": user_id}).eq("device_key", device_key).execute()
        _supabase.table("scans").update({"user_id": user_id}).eq("device_key", device_key).execute()
        
    try:
        _sync()
        logger.info("Successfully synced device %s history to Supabase for user %s", device_key[:8], user_id[:8])
    except Exception as e:
        logger.warning("Failed to sync history to Supabase: %s", e)
