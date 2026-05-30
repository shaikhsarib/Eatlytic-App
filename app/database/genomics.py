import json
import logging
from app.database.connection import db_conn

logger = logging.getLogger(__name__)

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
