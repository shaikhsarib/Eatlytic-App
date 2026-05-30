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

# ── Re-exports from split database modules (Onion Clean Architecture) ──
from app.database.cache import (
    get_ai_cache,
    set_ai_cache,
    get_ocr_cache,
    set_ocr_cache,
    get_image_fingerprint_match,
    set_image_fingerprint,
    get_research_cache,
    set_research_cache
)

from app.database.users import (
    check_and_increment_scan,
    purge_old_records,
    delete_user_data,
    get_unverified_scans,
    apply_correction,
    save_scan,
    get_device_history,
    get_scan_by_id,
    add_meal_log,
    get_daily_macro_totals,
    hash_password,
    verify_password,
    create_user,
    authenticate_user,
    create_session,
    get_session,
    delete_session,
    sync_local_history,
    sync_local_history_supabase
)

from app.database.api_keys import (
    verify_api_key,
    increment_api_scan,
    create_organization,
    get_org_by_admin,
    generate_api_key,
    get_org_api_keys,
    revoke_api_key
)

from app.database.genomics import (
    save_genomic_profile,
    get_genomic_profile
)


