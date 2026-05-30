import datetime
import logging
from app.database.connection import db_conn

logger = logging.getLogger(__name__)

def verify_api_key(api_key: str) -> dict:
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE api_key=? AND active=1", (api_key,)).fetchone()
    return dict(row) if row else None

def increment_api_scan(api_key: str) -> None:
    mo = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute("UPDATE api_keys SET scans_this_month = CASE WHEN month = ? THEN scans_this_month + 1 ELSE 1 END, month = ? WHERE api_key = ?", (mo, mo, api_key))

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
