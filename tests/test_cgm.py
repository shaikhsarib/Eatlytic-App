import pytest
import httpx
import json
import datetime
from unittest.mock import patch
from main import app
from app.database.connection import db_conn, init_db
import app.database.connection as db_mod

@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    yield

@pytest.mark.asyncio
async def test_cgm_ingestion_and_deduplication():
    """Verify that CGM readings sync correctly and duplicate timestamps are ignored."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_key = "device_test_cgm"
        
        # 1. Sync readings first time
        payload = {
            "device_key": device_key,
            "readings": [
                {"glucose_mgdl": 98.0, "recorded_at": "2026-05-29T18:00:00Z", "sensor_state": "active"},
                {"glucose_mgdl": 105.0, "recorded_at": "2026-05-29T18:15:00Z", "sensor_state": "active"}
            ]
        }
        
        resp1 = await ac.post("/cgm/sync", json=payload)
        assert resp1.status_code == 200
        d1 = resp1.json()
        assert d1["status"] == "success"
        assert d1["readings_received"] == 2
        assert d1["new_readings_inserted"] == 2
        
        # 2. Sync again with one duplicate and one new reading
        payload2 = {
            "device_key": device_key,
            "readings": [
                {"glucose_mgdl": 105.0, "recorded_at": "2026-05-29T18:15:00Z", "sensor_state": "active"}, # Duplicate
                {"glucose_mgdl": 112.0, "recorded_at": "2026-05-29T18:30:00Z", "sensor_state": "active"}  # New
            ]
        }
        
        resp2 = await ac.post("/cgm/sync", json=payload2)
        assert resp2.status_code == 200
        d2 = resp2.json()
        assert d2["readings_received"] == 2
        assert d2["new_readings_inserted"] == 1 # Only the new one should insert

@pytest.mark.asyncio
async def test_cgm_clinical_stats():
    """Verify Time-In-Range (TIR) and ADA Estimated HbA1c (eA1c) calculations."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_key = "device_stat_cgm"
        
        # Ingest custom readings to force specific stats:
        # Readings: [60 (Hypo), 80 (Range), 110 (Range), 150 (Hyper)]
        # Total = 4. Average = 100 mg/dL.
        # TIR = 2/4 = 50.0%. Hyper = 25.0%. Hypo = 25.0%.
        # eA1c = (100 + 46.7) / 28.7 = 5.11
        now_str = datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat()
        yesterday_str = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=12)).replace(tzinfo=None).isoformat()
        
        payload = {
            "device_key": device_key,
            "readings": [
                {"glucose_mgdl": 60.0, "recorded_at": yesterday_str, "sensor_state": "active"},
                {"glucose_mgdl": 80.0, "recorded_at": (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=6)).replace(tzinfo=None).isoformat(), "sensor_state": "active"},
                {"glucose_mgdl": 110.0, "recorded_at": (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=4)).replace(tzinfo=None).isoformat(), "sensor_state": "active"},
                {"glucose_mgdl": 150.0, "recorded_at": now_str, "sensor_state": "active"}
            ]
        }
        
        sync_resp = await ac.post("/cgm/sync", json=payload)
        assert sync_resp.status_code == 200
        
        stats_resp = await ac.get(f"/cgm/stats?device_key={device_key}")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        
        assert stats["total_readings_30d"] == 4
        assert stats["average_glucose_mgdl"] == 100.0
        assert stats["time_in_range_percent"] == 50.0
        assert stats["hyperglycemia_percent"] == 25.0
        assert stats["hypoglycemia_percent"] == 25.0
        assert stats["estimated_hba1c"] == 5.11

@pytest.mark.asyncio
async def test_cgm_postprandial_correlations():
    """Verify that product scans are correctly correlated with post-meal glucose spikes."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_key = "device_correlation_cgm"
        
        # 1. Seed device and 2 scans:
        # Scan A: 2026-05-29T10:00:00 (High spike trigger)
        # Scan B: 2026-05-29T14:00:00 (Flat safe scan)
        t_scan_a = "2026-05-29T10:00:00"
        t_scan_b = "2026-05-29T14:00:00"
        
        with db_conn() as conn:
            conn.execute("INSERT OR REPLACE INTO devices (device_key, persona) VALUES (?, ?)", (device_key, "general"))
            
            # Scan A: Sweet Wafer
            conn.execute("""
                INSERT INTO scans (device_key, product_name, score, verdict, brand, category, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (device_key, "Sweet Wafer", 3, "Avoid", "SugarCorp", "biscuit", t_scan_a))
            
            # Scan B: Almond Pack
            conn.execute("""
                INSERT INTO scans (device_key, product_name, score, verdict, brand, category, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (device_key, "Almond Pack", 8, "Safe", "NutCorp", "nut", t_scan_b))
            
        # 2. Ingest CGM readings around Scan A:
        # Baseline (10:00): 90 mg/dL
        # Spike (11:00): 135 mg/dL (+45 mg/dL, trigger!)
        # Ingest CGM readings around Scan B:
        # Baseline (14:00): 95 mg/dL
        # Max post (15:00): 102 mg/dL (+7 mg/dL, safe!)
        payload = {
            "device_key": device_key,
            "readings": [
                {"glucose_mgdl": 90.0, "recorded_at": "2026-05-29T10:00:00", "sensor_state": "active"},
                {"glucose_mgdl": 135.0, "recorded_at": "2026-05-29T11:00:00", "sensor_state": "active"},
                {"glucose_mgdl": 95.0, "recorded_at": "2026-05-29T14:00:00", "sensor_state": "active"},
                {"glucose_mgdl": 102.0, "recorded_at": "2026-05-29T15:00:00", "sensor_state": "active"}
            ]
        }
        
        sync_resp = await ac.post("/cgm/sync", json=payload)
        assert sync_resp.status_code == 200
        
        # 3. Fetch correlations and verify outcomes
        corr_resp = await ac.get(f"/cgm/correlations?device_key={device_key}")
        assert corr_resp.status_code == 200
        data = corr_resp.json()
        
        assert data["total_scans_analyzed_14d"] == 2
        corrs = {c["product_name"]: c for c in data["correlations"]}
        
        # Validate Sweet Wafer (Spike trigger)
        wafer = corrs["Sweet Wafer"]
        assert wafer["baseline_glucose"] == 90.0
        assert wafer["max_post_glucose"] == 135.0
        assert wafer["glucose_spike"] == 45.0
        assert wafer["is_biosensor_trigger"] is True
        assert "SEVERE GLYCEMIC TRIGGER" in wafer["clinical_recommendation"]
        
        # Validate Almond Pack (Safe)
        almond = corrs["Almond Pack"]
        assert almond["baseline_glucose"] == 95.0
        assert almond["max_post_glucose"] == 102.0
        assert almond["glucose_spike"] == 7.0
        assert almond["is_biosensor_trigger"] is False
        assert "Metabolic safe" in almond["clinical_recommendation"]
