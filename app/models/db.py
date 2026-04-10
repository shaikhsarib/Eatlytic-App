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

            CREATE TABLE IF NOT EXISTS sessions (
                token        TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at   TEXT DEFAULT (datetime('now')),
                expires_at   TEXT NOT NULL,
                device_hint  TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS devices (
                device_key      TEXT PRIMARY KEY,
                user_id         TEXT REFERENCES users(id),
                created_at      TEXT DEFAULT (datetime('now')),
                is_pro          INTEGER DEFAULT 0,
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
                sodium      REAL DEFAULT 0,
                fiber       REAL DEFAULT 0,
                sugar       REAL DEFAULT 0,
                source      TEXT DEFAULT 'scan',
                logged_at   TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_daily_user_date ON daily_logs(user_id, log_date);
            CREATE INDEX IF NOT EXISTS idx_daily_dev_date  ON daily_logs(device_key, log_date);

            CREATE TABLE IF NOT EXISTS allergen_profiles (
                device_key  TEXT PRIMARY KEY,
                user_id     TEXT REFERENCES users(id),
                allergens   TEXT DEFAULT '[]',
                conditions  TEXT DEFAULT '[]',
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS food_products (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                barcode         TEXT UNIQUE,
                name            TEXT NOT NULL,
                brand           TEXT DEFAULT '',
                category        TEXT DEFAULT '',
                calories_100g   REAL DEFAULT 0,
                protein_100g    REAL DEFAULT 0,
                carbs_100g      REAL DEFAULT 0,
                fat_100g        REAL DEFAULT 0,
                sodium_100g     REAL DEFAULT 0,
                fiber_100g      REAL DEFAULT 0,
                sugar_100g      REAL DEFAULT 0,
                sat_fat_100g    REAL DEFAULT 0,
                eatlytic_score  INTEGER DEFAULT 0,
                ingredients_raw TEXT DEFAULT '',
                allergens_json  TEXT DEFAULT '[]',
                source          TEXT DEFAULT 'llm_scan',
                scan_count      INTEGER DEFAULT 0,
                verified        INTEGER DEFAULT 0,
                verified_by     TEXT DEFAULT NULL,
                created_at      TEXT DEFAULT (datetime('now')),
                updated_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_food_barcode ON food_products(barcode);
            CREATE INDEX IF NOT EXISTS idx_food_name    ON food_products(name);
            CREATE INDEX IF NOT EXISTS idx_food_brand  ON food_products(brand);

            CREATE TABLE IF NOT EXISTS benchmarks (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name        TEXT NOT NULL,
                ground_truth_json   TEXT NOT NULL,
                llm_output_json     TEXT DEFAULT '{}',
                ocr_text            TEXT DEFAULT '',
                f1_score            REAL DEFAULT 0,
                score_delta         REAL DEFAULT 0,
                field_accuracy      TEXT DEFAULT '{}',
                tested_at           TEXT DEFAULT (datetime('now')),
                model_used          TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS nps_responses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                device_key   TEXT,
                user_id      TEXT REFERENCES users(id),
                score        INTEGER NOT NULL,
                comment      TEXT DEFAULT '',
                submitted_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS payments (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             TEXT REFERENCES users(id),
                device_key          TEXT,
                razorpay_order_id   TEXT UNIQUE,
                razorpay_payment_id TEXT UNIQUE,
                razorpay_signature  TEXT DEFAULT '',
                amount_paise        INTEGER DEFAULT 19900,
                currency            TEXT DEFAULT 'INR',
                status              TEXT DEFAULT 'created',
                plan                TEXT DEFAULT 'pro_monthly',
                created_at          TEXT DEFAULT (datetime('now')),
                paid_at             TEXT DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                api_key            TEXT PRIMARY KEY,
                client_name        TEXT NOT NULL,
                plan               TEXT DEFAULT 'business',
                scans_this_month   INTEGER DEFAULT 0,
                month              TEXT DEFAULT '',
                active             INTEGER DEFAULT 1,
                created_at         TEXT DEFAULT (datetime('now'))
            );

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
        """)
    logger.info("Database ready: %s", DB_FILE)


# ══════════════════════════════════════════════════════════════════════
#  SUPABASE CACHE LAYER (production) with SQLite fallback (dev)
# ══════════════════════════════════════════════════════════════════════


def get_ocr_cache(key: str):
    if _supabase:
        try:
            response = (
                _supabase.table("ocr_cache").select("*").eq("cache_key", key).execute()
            )
            if response.data:
                return json.loads(response.data[0]["result_json"])
        except Exception as e:
            logger.warning("Supabase get_ocr_cache failed: %s", e)
    try:
        with db_conn() as c:
            row = c.execute(
                "SELECT result_json FROM ocr_cache WHERE cache_key=?", (key,)
            ).fetchone()
        return json.loads(row["result_json"]) if row else None
    except Exception:
        return None


def set_ocr_cache(key: str, value: dict):
    if _supabase:
        try:
            _supabase.table("ocr_cache").upsert(
                {"cache_key": key, "result_json": json.dumps(value)}
            ).execute()
            return
        except Exception as e:
            logger.warning(
                "Supabase set_ocr_cache failed, falling back to SQLite: %s", e
            )
    try:
        with db_conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO ocr_cache(cache_key,result_json) VALUES(?,?)",
                (key, json.dumps(value)),
            )
    except Exception as exc:
        logger.warning("set_ocr_cache: %s", exc)


def get_ai_cache(key: str):
    if _supabase:
        try:
            response = (
                _supabase.table("cached_products")
                .select("*")
                .eq("label_hash", key)
                .execute()
            )
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.warning("Supabase get_ai_cache failed: %s", e)
    try:
        with db_conn() as c:
            row = c.execute(
                "SELECT result_json FROM ai_cache WHERE cache_key=? AND created_at > datetime('now', '-30 days')",
                (key,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None
    except Exception:
        return None


def set_ai_cache(key: str, value: dict):
    if _supabase:
        try:
            payload = {"label_hash": key}
            payload.update({k: v for k, v in value.items() if k not in ("label_hash",)})
            _supabase.table("cached_products").upsert(payload).execute()
            return
        except Exception as e:
            logger.warning(
                "Supabase set_ai_cache failed, falling back to SQLite: %s", e
            )
    try:
        with db_conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO ai_cache(cache_key,result_json) VALUES(?,?)",
                (key, json.dumps(value)),
            )
    except Exception as exc:
        logger.warning("set_ai_cache: %s", exc)


def check_and_increment_scan(device_key: str, limit: int = 10, increment: bool = True) -> dict:
    """Consolidated scan limit logic: tracks usage by device_key/month."""
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM devices WHERE device_key=?", (device_key,)
        ).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO devices(device_key, month, scan_count) VALUES(?,?,0)",
                (device_key, month_key),
            )
            u = {"is_pro": 0, "month": month_key, "scan_count": 0}
        else:
            u = dict(row)

        if u["month"] != month_key:
            conn.execute(
                "UPDATE devices SET month=?, scan_count=0 WHERE device_key=?",
                (month_key, device_key),
            )
            u["month"] = month_key
            u["scan_count"] = 0

        if u["is_pro"]:
            if increment:
                conn.execute(
                    "UPDATE devices SET scan_count=scan_count+1 WHERE device_key=?",
                    (device_key,),
                )
            return {
                "allowed": True,
                "scans_used": u["scan_count"] + (1 if increment else 0),
                "scans_remaining": 9999,
                "is_pro": True,
            }

        if u["scan_count"] >= limit:
            return {
                "allowed": False,
                "scans_used": u["scan_count"],
                "scans_remaining": 0,
                "is_pro": False,
            }

        if increment:
            conn.execute(
                "UPDATE devices SET scan_count=scan_count+1 WHERE device_key=?",
                (device_key,),
            )

        new_count = u["scan_count"] + (1 if increment else 0)
        return {
            "allowed": True,
            "scans_used": new_count,
            "scans_remaining": max(0, limit - new_count),
            "is_pro": False,
        }
