import pytest
import httpx
import json
import sqlite3
import datetime
from unittest.mock import patch
from main import app
from app.models.db import db_conn, init_db
import app.models.db as db_mod

@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    yield

@pytest.mark.asyncio
async def test_dietitian_registration_flow():
    """Verify dietitian registration endpoint works and enforces uniqueness."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Register successfully
        resp = await ac.post(
            "/dietitian/register",
            data={
                "name": "Dr. Sarib Mehta",
                "email": "sarib.mehta@dietcare.org",
                "code": "DR_SARIB_45"
            }
        )
        assert resp.status_code == 200
        d = resp.json()
        assert d["status"] == "registered"
        assert d["cohort_code"] == "DR_SARIB_45"
        assert d["dietitian_key"].startswith("diet_")
        assert "dashboard" in d["dashboard_url"]

        # 2. Try registering with duplicate code
        resp_dup = await ac.post(
            "/dietitian/register",
            data={
                "name": "Another Doctor",
                "email": "another@dietcare.org",
                "code": "dr_sarib_45"  # Should be case-insensitive / normalized
            }
        )
        assert resp_dup.status_code == 400
        assert "already registered" in resp_dup.json()["detail"]

@pytest.mark.asyncio
async def test_dietitian_join_cohort_flow():
    """Verify patients can join registered dietitian cohorts."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Register dietitian first
        reg_resp = await ac.post(
            "/dietitian/register",
            data={"name": "Dr. Sarib", "email": "s@diet.org", "code": "SARIB"}
        )
        assert reg_resp.status_code == 200

        # 2. Patient joins registered cohort
        join_resp = await ac.post(
            "/dietitian/join",
            data={"device_key": "patient_device_123", "cohort_code": "sarib"}
        )
        assert join_resp.status_code == 200
        jd = join_resp.json()
        assert jd["status"] == "joined"
        assert jd["cohort_code"] == "SARIB"
        assert jd["dietitian_name"] == "Dr. Sarib"

        # 3. Patient joins non-existent cohort code
        join_err = await ac.post(
            "/dietitian/join",
            data={"device_key": "patient_device_123", "cohort_code": "FAKE_CODE"}
        )
        assert join_err.status_code == 404
        assert "not found" in join_err.json()["detail"].lower()

@pytest.mark.asyncio
async def test_dietitian_dashboard_analytics():
    """Verify dietitian dashboard registers correct telemetry, statistics, and threats."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Setup dietitian and join patient
        reg_resp = await ac.post(
            "/dietitian/register",
            data={"name": "Dr. Sarib", "email": "s@diet.org", "code": "DIET"}
        )
        key = reg_resp.json()["dietitian_key"]
        
        device_key = "device_alice"
        await ac.post("/dietitian/join", data={"device_key": device_key, "cohort_code": "DIET"})

        # 2. Assert dashboard is empty initially
        dash_resp = await ac.get(f"/dietitian/dashboard?key={key}")
        assert dash_resp.status_code == 200
        dash = dash_resp.json()
        assert dash["total_patients"] == 1
        assert dash["total_scans_30d"] == 0
        assert dash["safety_ratios"] == {"Safe": 0.0, "Limit": 0.0, "Avoid": 0.0}
        assert len(dash["glycemic_threat_feed"]) == 0

        # 3. Seed some scans in SQLite for device_alice
        # Scan 1: Safe Food Scan
        # Scan 2: Avoid Glycemic Threat Scan
        with db_conn() as conn:
            # Safe scan (score 8)
            safe_analysis = {
                "product_name": "Almonds",
                "brand": "BadamCorp",
                "score": 8,
                "verdict": "Safe",
                "sugar": 1.2,
                "sodium_mg": 20.0,
                "cons": ["Standard low-sugar product"]
            }
            conn.execute("""
                INSERT INTO scans (device_key, product_name, score, verdict, brand, category, scanned_at, analysis_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (device_key, "Almonds", 8, "Safe", "BadamCorp", "nut", datetime.datetime.utcnow().isoformat(), json.dumps(safe_analysis)))

            # Avoid scan (score 2, glycemic threat)
            avoid_analysis = {
                "product_name": "Treat Jim Jam Biscuits",
                "brand": "Britannia",
                "score": 2,
                "verdict": "Glycemic Threat",
                "sugar": 37.5,
                "sodium_mg": 280.0,
                "cons": ["High in added sugar", "Glycemic threat: Refined wheat flour (Maida) triggers high insulin spikes"]
            }
            conn.execute("""
                INSERT INTO scans (device_key, product_name, score, verdict, brand, category, scanned_at, analysis_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (device_key, "Treat Jim Jam Biscuits", 2, "Glycemic Threat", "Britannia", "biscuit", datetime.datetime.utcnow().isoformat(), json.dumps(avoid_analysis)))

        # 4. Fetch dashboard again and verify clinical aggregation
        dash_resp2 = await ac.get(f"/dietitian/dashboard?key={key}")
        assert dash_resp2.status_code == 200
        dash2 = dash_resp2.json()
        
        assert dash2["total_patients"] == 1
        assert dash2["total_scans_30d"] == 2
        # Safe ratios: 1 Safe (50%), 1 Avoid (50%)
        assert dash2["safety_ratios"]["Safe"] == 50.0
        assert dash2["safety_ratios"]["Avoid"] == 50.0
        
        # Verify glycemic threat feed
        threat_feed = dash2["glycemic_threat_feed"]
        assert len(threat_feed) == 1
        threat = threat_feed[0]
        assert threat["product_name"] == "Treat Jim Jam Biscuits"
        assert threat["brand"] == "Britannia"
        assert threat["sugar_content"] == 37.5
        assert any("Maida" in t for t in threat["threat_triggers"])

        # 5. Fetch dashboard with invalid key
        err_resp = await ac.get("/dietitian/dashboard?key=invalid_key")
        assert err_resp.status_code == 403
        assert "invalid" in err_resp.json()["detail"].lower()
