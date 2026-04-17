"""
app/models/db.py
Database schema, connection management, and initialisation.
Uses Supabase for production (concurrent-safe cache + device tracking).
SQLite retained as local dev fallback when Supabase env vars are absent.
"""

import os
import json
import sqlite3
import logging
import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ── Supabase client (production) ──────────────────────────────────────
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
_supabase = None
if _SUPABASE_URL and _SUPABASE_KEY:
    from supabase import create_client, Client
    _supabase: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    logger.info("Supabase client initialised: %s", _SUPABASE_URL)
else:
    logger.warning("Supabase env vars missing — using SQLite fallback")

# ── SQLite fallback (local dev only) ─────────────────────────────────
DATA_DIR = os.path.join(os.getcwd(), "data")
DB_FILE = os.path.join(DATA_DIR, "eatlytic.db")
os.makedirs(DATA_DIR, exist_ok=True)

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

@contextmanager
def db_conn():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    with db_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                email        TEXT UNIQUE,
                phone        TEXT UNIQUE,
                name         TEXT DEFAULT '',
                created_at   TEXT DEFAULT (datetime('now')),
                is_pro       INTEGER DEFAULT 0,
                pro_expires  TEXT,
                scan_count_month INTEGER DEFAULT 0,
                scan_month   TEXT DEFAULT '',
                streak_days  INTEGER DEFAULT 0,
                last_scan_date TEXT DEFAULT '',
                tdee         REAL DEFAULT 0,
                persona      TEXT DEFAULT 'General Adult',
                language     TEXT DEFAULT 'en',
                onboarding_done INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS organizations (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                plan         TEXT DEFAULT 'business',
                admin_id     TEXT REFERENCES users(id),
                created_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                api_key            TEXT PRIMARY KEY,
                client_name        TEXT NOT NULL,
                organization_id    TEXT REFERENCES organizations(id),
                plan               TEXT DEFAULT 'business',
                scans_this_month   INTEGER DEFAULT 0,
                month              TEXT DEFAULT '',
                active             INTEGER DEFAULT 1,
                created_at         TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS devices (
                device_key      TEXT PRIMARY KEY,
                user_id         TEXT REFERENCES users(id),
                created_at      TEXT DEFAULT (datetime('now')),
                is_pro          INTEGER DEFAULT 0,
                pro_expires     TEXT,
                month           TEXT DEFAULT '',
                scan_count      INTEGER DEFAULT 0,
                streak_days     INTEGER DEFAULT 0,
                last_scan_date  TEXT DEFAULT '',
                persona         TEXT DEFAULT 'General Adult',
                language        TEXT DEFAULT 'en',
                tdee            REAL DEFAULT 0,
                onboarding_done INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS scans (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT REFERENCES users(id),
                device_key    TEXT,
                product_name  TEXT DEFAULT 'Unknown',
                score         INTEGER DEFAULT 0,
                verdict       TEXT DEFAULT '',
                calories      REAL DEFAULT 0,
                protein       REAL DEFAULT 0,
                carbs         REAL DEFAULT 0,
                fat           REAL DEFAULT 0,
                sodium        REAL DEFAULT 0,
                fiber         REAL DEFAULT 0,
                sugar         REAL DEFAULT 0,
                persona       TEXT DEFAULT '',
                language      TEXT DEFAULT 'en',
                scanned_at    TEXT DEFAULT (datetime('now')),
                analysis_json TEXT DEFAULT '{}',
                metadata_json TEXT DEFAULT '{}',
                correction_json TEXT DEFAULT '{}',
                verified      INTEGER DEFAULT 0,
                verified_by   TEXT DEFAULT NULL,
                verified_at   TEXT DEFAULT NULL,
                barcode       TEXT DEFAULT NULL,
                brand         TEXT DEFAULT NULL,
                category      TEXT DEFAULT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scans_user     ON scans(user_id);
            CREATE INDEX IF NOT EXISTS idx_scans_device  ON scans(device_key);
            CREATE INDEX IF NOT EXISTS idx_scans_date    ON scans(scanned_at);
            CREATE INDEX IF NOT EXISTS idx_scans_product ON scans(product_name);

            CREATE TABLE IF NOT EXISTS ocr_cache (
                cache_key   TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ai_cache (
                cache_key   TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS image_fingerprints (
                hash_key    TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            
            -- Keep other tables for consistency
            CREATE TABLE IF NOT EXISTS daily_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT REFERENCES users(id),
                device_key  TEXT,
                log_date    TEXT NOT NULL,
                meal_name   TEXT DEFAULT '',
                calories    REAL DEFAULT 0,
                logged_at   TEXT DEFAULT (datetime('now'))
            );
        """)
        # Migrations for existing DBs
        try: conn.execute("ALTER TABLE scans ADD COLUMN metadata_json TEXT DEFAULT '{}'")
        except: pass
        try: conn.execute("ALTER TABLE scans ADD COLUMN correction_json TEXT DEFAULT '{}'")
        except: pass
    logger.info("Database ready: %s", DB_FILE)

# ── Retrieval Helpers ──────────────────────────────────────────────────
def get_ocr_cache(key: str):
    if _supabase:
        try:
            res = _supabase.table("ocr_cache").select("*").eq("cache_key", key).execute()
            if res.data: return json.loads(res.data[0]["result_json"])
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM ocr_cache WHERE cache_key=?", (key,)).fetchone()
    return json.loads(row["result_json"]) if row else None

def set_ocr_cache(key: str, value: dict):
    if _supabase:
        try: _supabase.table("ocr_cache").upsert({"cache_key": key, "result_json": json.dumps(value)}).execute()
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO ocr_cache(cache_key,result_json) VALUES(?,?)", (key, json.dumps(value)))

def get_image_fingerprint_match(hash_key: str):
    if _supabase:
        try:
            res = _supabase.table("image_fingerprints").select("*").eq("hash_key", hash_key).execute()
            if res.data: return json.loads(res.data[0]["result_json"])
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM image_fingerprints WHERE hash_key=?", (hash_key,)).fetchone()
    return json.loads(row["result_json"]) if row else None

def set_image_fingerprint(hash_key: str, value: dict):
    if _supabase:
        try: _supabase.table("image_fingerprints").upsert({"hash_key": hash_key, "result_json": json.dumps(value)}).execute()
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO image_fingerprints(hash_key, result_json) VALUES(?,?)", (hash_key, json.dumps(value)))


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
            if exp and exp < datetime.datetime.utcnow().isoformat():
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

def verify_api_key(api_key: str) -> dict:
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE api_key=? AND active=1", (api_key,)).fetchone()
    return dict(row) if row else None

def increment_api_scan(api_key: str) -> None:
    mo = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute("UPDATE api_keys SET scans_this_month = CASE WHEN month = ? THEN scans_this_month + 1 ELSE 1 END, month = ? WHERE api_key = ?", (mo, mo, api_key))

# ── DPDP Compliance & Admin ───────────────────────────────────────────
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
    # Parse JSON fields
    for field in ["analysis_json", "metadata_json", "correction_json"]:
        if d.get(field):
            try: d[field] = json.loads(d[field])
            except: d[field] = {}
    return d
