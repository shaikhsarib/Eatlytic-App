"""
app/database/connection.py
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

def parse_utc_iso(iso_str: str) -> datetime.datetime:
    """Safely parse timezone-aware and naive ISO strings, returning a naive UTC datetime."""
    try:
        # standard ISO format might have "Z" instead of "+00:00"
        s = iso_str.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        # Fallback to current time if parsing fails
        return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

def _supabase_execute_with_retry(operation_fn):
    """Executes a Supabase query/mutation with tenacity-backed exponential backoff."""
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _execute():
        return operation_fn()
        
    return _execute()

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
                password_hash TEXT,
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

            CREATE TABLE IF NOT EXISTS research_cache (
                query       TEXT PRIMARY KEY,
                result      TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            
            -- Keep other tables for consistency
            CREATE TABLE IF NOT EXISTS sessions (
                token       TEXT PRIMARY KEY,
                user_id     TEXT REFERENCES users(id),
                expires_at  TEXT NOT NULL,
                device_hint TEXT DEFAULT ''
            );
            
            CREATE TABLE IF NOT EXISTS daily_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT REFERENCES users(id),
                device_key  TEXT,
                log_date    TEXT NOT NULL,
                meal_name   TEXT DEFAULT '',
                calories    REAL DEFAULT 0,
                protein     REAL DEFAULT 0,
                carbs       REAL DEFAULT 0,
                fat         REAL DEFAULT 0,
                logged_at   TEXT DEFAULT (datetime('now'))
            );
            
            CREATE TABLE IF NOT EXISTS food_products (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL,
                brand        TEXT DEFAULT '',
                category     TEXT DEFAULT '',
                barcode      TEXT UNIQUE,
                calories_100g REAL DEFAULT 0,
                protein_100g  REAL DEFAULT 0,
                carbs_100g    REAL DEFAULT 0,
                fat_100g      REAL DEFAULT 0,
                sodium_100g   REAL DEFAULT 0,
                fiber_100g    REAL DEFAULT 0,
                sugar_100g    REAL DEFAULT 0,
                eatlytic_score INTEGER DEFAULT 0,
                verified      INTEGER DEFAULT 0,
                verified_by   TEXT DEFAULT NULL,
                verified_at   TEXT DEFAULT NULL,
                scan_count    INTEGER DEFAULT 0,
                created_at    TEXT DEFAULT (datetime('now')),
                updated_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_fp_barcode ON food_products(barcode);
            CREATE INDEX IF NOT EXISTS idx_fp_name    ON food_products(name);

            CREATE TABLE IF NOT EXISTS scan_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id     INTEGER,
                device_key  TEXT,
                note        TEXT DEFAULT '',
                reported_at TEXT DEFAULT (datetime('now')),
                resolved    INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_reports_scan ON scan_reports(scan_id);

            CREATE TABLE IF NOT EXISTS food_products_extra (
                id              INTEGER PRIMARY KEY,
                sat_fat_100g    REAL DEFAULT 0,
                ingredients_raw TEXT DEFAULT '',
                source          TEXT DEFAULT 'llm_scan'
            );

            CREATE TABLE IF NOT EXISTS payments (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             TEXT,
                device_key          TEXT,
                razorpay_order_id   TEXT UNIQUE,
                razorpay_payment_id TEXT,
                razorpay_signature  TEXT,
                amount_paise        INTEGER,
                currency            TEXT DEFAULT 'INR',
                status              TEXT,
                paid_at             TEXT,
                created_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS dietitians (
                code            TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                email           TEXT NOT NULL,
                dietitian_key   TEXT NOT NULL,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS patient_cohorts (
                device_key      TEXT NOT NULL,
                dietitian_code  TEXT REFERENCES dietitians(code) ON DELETE CASCADE,
                created_at      TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (device_key, dietitian_code)
            );

            CREATE TABLE IF NOT EXISTS genomic_profiles (
                device_key      TEXT PRIMARY KEY,
                genetic_snps    TEXT,
                biomarkers      TEXT,
                updated_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS cgm_readings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                device_key      TEXT NOT NULL,
                glucose_mgdl    REAL NOT NULL,
                recorded_at     TEXT NOT NULL,
                sensor_state    TEXT DEFAULT 'active',
                created_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (device_key) REFERENCES devices(device_key) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_cgm_device_time ON cgm_readings (device_key, recorded_at DESC);
        """)
        # Migrations for existing DBs — only ignore "duplicate column" OperationalErrors
        try: conn.execute("ALTER TABLE scans ADD COLUMN metadata_json TEXT DEFAULT '{}'")
        except sqlite3.OperationalError: pass
        try: conn.execute("ALTER TABLE scans ADD COLUMN correction_json TEXT DEFAULT '{}'")
        except sqlite3.OperationalError: pass
        try: conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        except sqlite3.OperationalError: pass
        try: conn.execute("ALTER TABLE daily_logs ADD COLUMN user_id TEXT")
        except sqlite3.OperationalError: pass
        try: conn.execute("ALTER TABLE daily_logs ADD COLUMN protein REAL DEFAULT 0")
        except sqlite3.OperationalError: pass
        try: conn.execute("ALTER TABLE daily_logs ADD COLUMN carbs REAL DEFAULT 0")
        except sqlite3.OperationalError: pass
        try: conn.execute("ALTER TABLE daily_logs ADD COLUMN fat REAL DEFAULT 0")
        except sqlite3.OperationalError: pass
        # Seed a default B2B Key for local development and live telemetry demo
        try:
            mo = datetime.date.today().isoformat()[:7]
            conn.execute("""
                INSERT OR IGNORE INTO api_keys (api_key, client_name, plan, scans_this_month, active, month)
                VALUES ('eatlytic_live_testkey123', 'Eatlytic Dev Portal Live Demo', 'business', 142, 1, ?)
            """, (mo,))
        except Exception as e:
            logger.error("Failed to seed default B2B dev key: %s", e)

    logger.info("Database ready: %s", DB_FILE)

def get_ai_cache(key: str):
    if _supabase:
        try:
            res = _supabase_execute_with_retry(lambda: _supabase.table("ai_cache").select("*").eq("cache_key", key).execute())
            if res.data: return json.loads(res.data[0]["result_json"])
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM ai_cache WHERE cache_key=?", (key,)).fetchone()
    return json.loads(row["result_json"]) if row else None

def set_ai_cache(key: str, value: dict):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("ai_cache").upsert({"cache_key": key, "result_json": json.dumps(value)}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO ai_cache(cache_key,result_json) VALUES(?,?)", (key, json.dumps(value)))

# ── Retrieval Helpers ──────────────────────────────────────────────────
def get_ocr_cache(key: str):
    if _supabase:
        try:
            res = _supabase_execute_with_retry(lambda: _supabase.table("ocr_cache").select("*").eq("cache_key", key).execute())
            if res.data: return json.loads(res.data[0]["result_json"])
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM ocr_cache WHERE cache_key=?", (key,)).fetchone()
    return json.loads(row["result_json"]) if row else None

def set_ocr_cache(key: str, value: dict):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("ocr_cache").upsert({"cache_key": key, "result_json": json.dumps(value)}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO ocr_cache(cache_key,result_json) VALUES(?,?)", (key, json.dumps(value)))

_bktree = None
_bktree_lock = None

def _get_bktree():
    global _bktree, _bktree_lock
    if _bktree_lock is None:
        import threading
        _bktree_lock = threading.Lock()
    if _bktree is None:
        with _bktree_lock:
            if _bktree is None:
                from app.ai.perception.bk_tree import BKTree
                tree = BKTree()
                with db_conn() as c:
                    rows = c.execute("SELECT hash_key, result_json FROM image_fingerprints").fetchall()
                for row in rows:
                    tree.insert(row["hash_key"], row["result_json"])
                _bktree = tree
    return _bktree

def get_image_fingerprint_match(hash_key: str):
    if not hash_key:
        return None
    # 1. Try exact lookup first (O(1))
    with db_conn() as c:
        row = c.execute("SELECT result_json FROM image_fingerprints WHERE hash_key=?", (hash_key,)).fetchone()
    if row:
        return json.loads(row["result_json"])
        
    # 2. BK-Tree sub-linear Hamming search (replaces O(n) table scan)
    tree = _get_bktree()
    matches = tree.search(hash_key, max_distance=6)
    if matches:
        matches.sort(key=lambda x: x[0])
        return json.loads(matches[0][1])
    return None

def set_image_fingerprint(hash_key: str, value: dict):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("image_fingerprints").upsert({"hash_key": hash_key, "result_json": json.dumps(value)}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO image_fingerprints(hash_key, result_json) VALUES(?,?)", (hash_key, json.dumps(value)))
    # Synchronize BK-Tree
    tree = _get_bktree()
    tree.insert(hash_key, json.dumps(value))

# ── Research Cache (Phase 2 Latency Optimization) ─────────────────────
def get_research_cache(query: str):
    if _supabase:
        try:
            res = _supabase_execute_with_retry(lambda: _supabase.table("research_cache").select("*").eq("query", query).execute())
            if res.data: return res.data[0]["result"]
        except Exception: pass
    with db_conn() as c:
        row = c.execute("SELECT result FROM research_cache WHERE query=?", (query,)).fetchone()
    return row["result"] if row else None

def set_research_cache(query: str, result: str):
    if _supabase:
        try: _supabase_execute_with_retry(lambda: _supabase.table("research_cache").upsert({"query": query, "result": result}).execute())
        except: pass
    with db_conn() as c:
        c.execute("INSERT OR REPLACE INTO research_cache(query, result) VALUES(?,?)", (query, result))


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

# ── User Authentication & Password Hashing ─────────────────────────────
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
        # strip password_hash for safety before returning
        user_dict.pop("password_hash", None)
        return user_dict
    return None

def create_session(user_id: str, device_hint: str = "") -> str:
    import os
    import datetime
    token = os.urandom(24).hex()
    expires_at = (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(days=30)).isoformat()
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO sessions (token, user_id, expires_at, device_hint)
            VALUES (?, ?, ?, ?)
        """, (token, user_id, expires_at, device_hint))
    return token

def get_session(token: str) -> dict:
    import datetime
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
    """
    Consolidates stateless history records on user registration/login.
    Syncs SQLite local cache buffers up to the Supabase cloud cluster with retries.
    """
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

# ── B2B Developer Organization & API Keys ─────────────────────────────
def create_organization(admin_id: str, name: str) -> str:
    import uuid
    org_id = str(uuid.uuid4())
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO organizations (id, name, admin_id)
            VALUES (?, ?, ?)
        """, (org_id, name, admin_id))
    return org_id

def get_org_by_admin(admin_id: str) -> dict:
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM organizations WHERE admin_id = ?", (admin_id,)).fetchone()
    return dict(row) if row else None

def generate_api_key(client_name: str, organization_id: str, plan: str = "business") -> str:
    import os
    api_key = f"eatlytic_live_{os.urandom(16).hex()}"
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO api_keys (api_key, client_name, organization_id, plan, active)
            VALUES (?, ?, ?, ?, 1)
        """, (api_key, client_name, organization_id, plan))
    return api_key

def get_org_api_keys(organization_id: str) -> list:
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM api_keys WHERE organization_id = ? ORDER BY created_at DESC", (organization_id,)).fetchall()
    return [dict(r) for r in rows]

def revoke_api_key(api_key: str) -> None:
    with db_conn() as conn:
        conn.execute("UPDATE api_keys SET active = 0 WHERE api_key = ?", (api_key,))


# ── Genomic Profiles (Phase 5 AI Personalized Medicine) ────────────────
def save_genomic_profile(device_key: str, genetic_snps: dict, biomarkers: dict) -> None:
    with db_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO genomic_profiles (device_key, genetic_snps, biomarkers, updated_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (device_key, json.dumps(genetic_snps), json.dumps(biomarkers)))

def get_genomic_profile(device_key: str) -> dict:
    with db_conn() as conn:
        row = conn.execute("SELECT genetic_snps, biomarkers FROM genomic_profiles WHERE device_key = ?", (device_key,)).fetchone()
    if not row:
        return None
    return {
        "device_key": device_key,
        "genetic_snps": json.loads(row["genetic_snps"]) if row["genetic_snps"] else {},
        "biomarkers": json.loads(row["biomarkers"]) if row["biomarkers"] else {}
    }

