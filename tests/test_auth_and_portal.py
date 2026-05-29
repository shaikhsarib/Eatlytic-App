import pytest
import httpx
import sqlite3
import datetime
from main import app
from app.database.connection import db_conn, init_db, verify_password, hash_password

@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = str(tmp_path / "test.db")
    import app.database.connection as db_mod

    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    yield


@pytest.mark.anyio
async def test_user_registration_and_login():
    """Verify signup, login, session token validation, and password hashing correctness."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Test Registration
        reg_res = await ac.post("/api/v1/auth/register", data={
            "email": "dev@eatlytic.com",
            "password": "securepassword123",
            "name": "Alex Dev"
        })
        assert reg_res.status_code == 200
        reg_data = reg_res.json()
        assert reg_data["status"] == "registered"
        assert "token" in reg_data
        assert reg_data["user"]["email"] == "dev@eatlytic.com"
        assert reg_data["user"]["name"] == "Alex Dev"
        
        token = reg_data["token"]
        
        # Verify user exists in DB and has hashed password
        with db_conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email='dev@eatlytic.com'").fetchone()
            assert row is not None
            assert verify_password(row["password_hash"], "securepassword123")
            
        # 2. Test Duplicate Registration
        dup_res = await ac.post("/api/v1/auth/register", data={
            "email": "dev@eatlytic.com",
            "password": "anotherpassword",
            "name": "Alex Duplicate"
        })
        assert dup_res.status_code == 400
        
        # 3. Test Session Validation
        session_res = await ac.get("/api/v1/auth/session", headers={"X-Eatlytic-Session": token})
        assert session_res.status_code == 200
        session_data = session_res.json()
        assert session_data["status"] == "active"
        assert session_data["user"]["email"] == "dev@eatlytic.com"
        
        # 4. Test Invalid Session
        inv_session_res = await ac.get("/api/v1/auth/session", headers={"X-Eatlytic-Session": "invalidtoken"})
        assert inv_session_res.status_code == 401
        
        # 5. Test Logout
        logout_res = await ac.post("/api/v1/auth/logout", headers={"X-Eatlytic-Session": token})
        assert logout_res.status_code == 200
        
        # Verify session is deleted
        with db_conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
            assert row is None
            
        # 6. Test Login
        login_res = await ac.post("/api/v1/auth/login", data={
            "email": "dev@eatlytic.com",
            "password": "securepassword123"
        })
        assert login_res.status_code == 200
        login_data = login_res.json()
        assert login_data["status"] == "logged_in"
        assert "token" in login_data
        
        # 7. Test Login Incorrect Password
        bad_login_res = await ac.post("/api/v1/auth/login", data={
            "email": "dev@eatlytic.com",
            "password": "wrongpassword"
        })
        assert bad_login_res.status_code == 401

@pytest.mark.anyio
async def test_local_history_and_logs_sync():
    """Verify that anonymous scans and daily meal logs are synced to user accounts upon login."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_key = "anonymous_device_123"
        
        # Seed anonymous scans and logs
        with db_conn() as conn:
            conn.execute("""
                INSERT INTO scans (device_key, product_name, score, verdict)
                VALUES (?, 'Anonymous Biscuit', 70, 'HEALTHY')
            """, (device_key,))
            conn.execute("""
                INSERT INTO daily_logs (device_key, log_date, meal_name, calories, protein, carbs, fat)
                VALUES (?, ?, 'Anonymous Meal', 300, 15, 40, 8)
            """, (device_key, datetime.date.today().isoformat()))
            
        # Register a new user with this device key
        reg_res = await ac.post("/api/v1/auth/register", data={
            "email": "syncuser@eatlytic.com",
            "password": "syncpassword123",
            "name": "Sync User",
            "device_key": device_key
        })
        assert reg_res.status_code == 200
        user_id = reg_res.json()["user"]["id"]
        token = reg_res.json()["token"]
        
        # Verify that scans and daily logs are now updated with this user_id
        with db_conn() as conn:
            scan_row = conn.execute("SELECT * FROM scans WHERE device_key=? AND user_id=?", (device_key, user_id)).fetchone()
            assert scan_row is not None
            assert scan_row["product_name"] == "Anonymous Biscuit"
            
            log_row = conn.execute("SELECT * FROM daily_logs WHERE device_key=? AND user_id=?", (device_key, user_id)).fetchone()
            assert log_row is not None
            assert log_row["meal_name"] == "Anonymous Meal"
            
        # Get history endpoint and verify the synced items return
        hist_res = await ac.get("/api/v1/history", headers={"X-Eatlytic-Session": token})
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        assert len(hist_data) >= 1
        assert hist_data[0]["product_name"] == "Anonymous Biscuit"

@pytest.mark.anyio
async def test_developer_portal_and_b2b_api():
    """Verify B2B organization registration, API key creation, key listings, revoking, and key auth validations."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # Register developer
        reg_res = await ac.post("/api/v1/auth/register", data={
            "email": "developer@enterprise.com",
            "password": "devpassword123",
            "name": "Lead Developer"
        })
        assert reg_res.status_code == 200
        token = reg_res.json()["token"]
        
        # 1. Fetch dashboard with no organization
        dash_empty_res = await ac.get("/api/v1/developer/dashboard", headers={"X-Eatlytic-Session": token})
        assert dash_empty_res.status_code == 200
        assert dash_empty_res.json()["status"] == "no_organization"
        
        # 2. Register Developer Organization
        org_res = await ac.post("/api/v1/developer/organization", data={
            "name": "Superfood Innovations Ltd"
        }, headers={"X-Eatlytic-Session": token})
        assert org_res.status_code == 200
        assert org_res.json()["status"] == "created"
        assert org_res.json()["organization"]["name"] == "Superfood Innovations Ltd"
        
        # 3. Fetch dashboard with active organization
        dash_res = await ac.get("/api/v1/developer/dashboard", headers={"X-Eatlytic-Session": token})
        assert dash_res.status_code == 200
        dash_data = dash_res.json()
        assert dash_data["status"] == "success"
        assert dash_data["organization"]["name"] == "Superfood Innovations Ltd"
        assert len(dash_data["api_keys"]) == 0
        
        # 4. Generate B2B API Key
        key_res = await ac.post("/api/v1/developer/keys", data={
            "client_name": "Acme Internal App",
            "plan": "business"
        }, headers={"X-Eatlytic-Session": token})
        assert key_res.status_code == 200
        key_data = key_res.json()
        assert key_data["status"] == "generated"
        assert "api_key" in key_data
        assert key_data["client_name"] == "Acme Internal App"
        
        api_key = key_data["api_key"]
        
        # Verify active in DB
        with db_conn() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE api_key=?", (api_key,)).fetchone()
            assert row is not None
            assert row["active"] == 1
            
        # 5. Fetch dashboard showing keys list
        dash_with_keys_res = await ac.get("/api/v1/developer/dashboard", headers={"X-Eatlytic-Session": token})
        assert dash_with_keys_res.status_code == 200
        keys_list = dash_with_keys_res.json()["api_keys"]
        assert len(keys_list) == 1
        assert keys_list[0]["client_name"] == "Acme Internal App"
        assert keys_list[0]["api_key"] == api_key
        
        # 6. Revoke API Key
        revoke_res = await ac.post("/api/v1/developer/keys/revoke", data={
            "api_key": api_key
        }, headers={"X-Eatlytic-Session": token})
        assert revoke_res.status_code == 200
        assert revoke_res.json()["status"] == "revoked"
        
        # Verify inactive in DB
        with db_conn() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE api_key=?", (api_key,)).fetchone()
            assert row is not None
            assert row["active"] == 0
