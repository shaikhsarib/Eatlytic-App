import pytest
import httpx
import json
import sqlite3
import datetime
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
async def test_personalized_profile_sync_stateless():
    """Verify that uploading genomic profiles statelessly registers to the device context."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_key = "device_genomic_01"
        headers = {"X-Device-Key": device_key}
        
        payload = {
            "genetic_snps": {"rs7903146": "TT", "rs5068": "GG", "rs4988235": "GG"},
            "biomarkers": {"hba1c": 6.2, "fasting_glucose": 110.0}
        }
        
        # 1. Upload genomic profile
        resp = await ac.post("/personalized/profile", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["device_key"] == device_key
        
        # 2. Retrieve profile and assert matching values
        get_resp = await ac.get("/personalized/profile", headers=headers)
        assert get_resp.status_code == 200
        profile = get_resp.json()
        assert profile["device_key"] == device_key
        assert profile["genetic_snps"]["rs7903146"] == "TT"
        assert profile["biomarkers"]["hba1c"] == 6.2

@pytest.mark.asyncio
async def test_genomic_tcf7l2_diabetes_override():
    """Verify that the TCF7L2 Type-2 Diabetes variant dynamically overrides high-GI ingredient warnings."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # User 1: Control (baseline user with empty genomic profile)
        control_headers = {"X-Device-Key": "device_control"}
        
        # User 2: Predisposed (TCF7L2 'TT' Genotype)
        predisposed_headers = {"X-Device-Key": "device_tcf_predisposed"}
        await ac.post(
            "/personalized/profile",
            json={"genetic_snps": {"rs7903146": "TT"}, "biomarkers": {}},
            headers=predisposed_headers
        )
        
        # Scan raw product details containing Maltodextrin (high GI) under general persona (gives high control score)
        scan_payload = {
            "product_name": "Energy Drink Powder",
            "brand": "ActiveCorp",
            "category": "beverage",
            "nutrients": {"sugar": 1.0, "carbs": 5.0, "calories": 30.0},
            "ingredients_raw": "Maltodextrin, minor sugar, natural flavor",
            "persona": "general"
        }
        
        # 1. Run baseline audit on Control device
        ctrl_resp = await ac.post("/personalized/scan", json=scan_payload, headers=control_headers)
        assert ctrl_resp.status_code == 200
        ctrl_data = ctrl_resp.json()
        
        # 2. Run personalized audit on TCF7L2 TT device
        pred_resp = await ac.post("/personalized/scan", json=scan_payload, headers=predisposed_headers)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        
        # 3. Assert dynamic genomic score and verdict overrides
        assert pred_data["clinical_audit"]["verdict"] == "AVOID"
        assert pred_data["safety_tier"] == "Avoid"
        assert pred_data["score"] < ctrl_data["score"]
        
        # Assert specific clinical genomic warning is appended
        reasons_text = " ".join(pred_data["cons"])
        assert "Genomic Risk" in reasons_text
        assert "TCF7L2" in reasons_text
        assert "rs7903146" in reasons_text
        assert "Maltodextrin" in reasons_text

@pytest.mark.asyncio
async def test_genomic_lct_lactose_allergen_override():
    """Verify that LCT GG variant dynamically triggers severe warnings on dairy-derived products."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_lactose_intolerant = "device_lactose_intolerant"
        headers = {"X-Device-Key": device_lactose_intolerant}
        
        await ac.post(
            "/personalized/profile",
            json={"genetic_snps": {"rs4988235": "GG"}, "biomarkers": {}},
            headers=headers
        )
        
        dairy_scan = {
            "product_name": "Protein Cookie",
            "brand": "WheyPro",
            "category": "cookies",
            "nutrients": {"protein": 20.0, "sugar": 2.0, "calories": 250.0},
            "ingredients_raw": "Whey protein isolate, chocolate chips, milk solids",
            "persona": "general"
        }
        
        resp = await ac.post("/personalized/scan", json=dairy_scan, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["clinical_audit"]["verdict"] == "AVOID"
        assert data["safety_tier"] == "Avoid"
        assert data["score"] == 1
        
        reasons_text = " ".join(data["cons"])
        assert "LCT" in reasons_text
        assert "rs4988235" in reasons_text
        assert "Lactase Non-Persistence" in reasons_text

@pytest.mark.asyncio
async def test_genomic_agt_sodium_hypertension_override():
    """Verify that AGT sodium-sensitive hypertension variants dynamically reduce daily sodium thresholds."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_agt = "device_agt"
        headers = {"X-Device-Key": device_agt}
        
        await ac.post(
            "/personalized/profile",
            json={"genetic_snps": {"rs5068": "GG"}, "biomarkers": {}},
            headers=headers
        )
        
        # High sodium product (350mg sodium)
        salt_scan = {
            "product_name": "Instant Ramen Tastemaker",
            "brand": "NoodleCorp",
            "category": "spices",
            "nutrients": {"sodium_mg": 350.0, "sugar": 0.5, "calories": 40.0},
            "ingredients_raw": "Salt, MSG, chili powder",
            "persona": "general"
        }
        
        resp = await ac.post("/personalized/scan", json=salt_scan, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["clinical_audit"]["verdict"] == "AVOID"
        assert data["safety_tier"] == "Avoid"
        
        reasons_text = " ".join(data["cons"])
        assert "AGT" in reasons_text
        assert "rs5068" in reasons_text
        assert "sodium-sensitive" in reasons_text

@pytest.mark.asyncio
async def test_biomarker_hba1c_sweetener_downgrade():
    """Verify that prediabetic/diabetic HbA1c readings downgrade synthetic sweeteners."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        device_diabetic = "device_diabetic"
        headers = {"X-Device-Key": device_diabetic}
        
        await ac.post(
            "/personalized/profile",
            json={"genetic_snps": {}, "biomarkers": {"hba1c": 6.2}},
            headers=headers
        )
        
        sweetener_scan = {
            "product_name": "Diet Cola",
            "brand": "FizzyCorp",
            "category": "beverage",
            "nutrients": {"sugar": 0.0, "calories": 0.0},
            "ingredients_raw": "Carbonated water, acesulfame potassium, aspartame",
            "persona": "diabetic"
        }
        
        resp = await ac.post("/personalized/scan", json=sweetener_scan, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["clinical_audit"]["verdict"] == "AVOID"
        assert data["safety_tier"] == "Avoid"
        
        reasons_text = " ".join(data["cons"])
        assert "Biomarker Risk" in reasons_text
        assert "HbA1c" in reasons_text
        assert "aspartame" in reasons_text
