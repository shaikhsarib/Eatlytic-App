import os
import re
import pytest
import datetime
from unittest.mock import patch
from app.models.db import db_conn, init_db
import app.models.db as db_mod
from app.services.llm.engine import find_db_product_match, unified_analyze_flow

# ── Self-contained fixture to redirect DB to temporary space ──────────────
@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    
    # Seed mock products for tests
    with db_conn() as conn:
        # Seed Amul Butter
        cursor = conn.execute("""
            INSERT INTO food_products (
                name, brand, category, barcode, calories_100g,
                protein_100g, carbs_100g, fat_100g, sodium_100g,
                fiber_100g, sugar_100g, eatlytic_score, verified,
                scan_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)
        """, ("Pasteurised Butter", "Amul", "Dairy", "8901262010018", 722.0, 0.6, 0.0, 80.0, 830.0, 0.0, 0.0, 4))
        amul_id = cursor.lastrowid
        conn.execute("""
            INSERT INTO food_products_extra (id, sat_fat_100g, ingredients_raw, source)
            VALUES (?, ?, ?, ?)
        """, (amul_id, 51.0, "Butter, Common Salt", "verified_seeder"))
        
        # Seed Maggi Noodles
        cursor = conn.execute("""
            INSERT INTO food_products (
                name, brand, category, barcode, calories_100g,
                protein_100g, carbs_100g, fat_100g, sodium_100g,
                fiber_100g, sugar_100g, eatlytic_score, verified,
                scan_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)
        """, ("Maggi Masala Noodles", "Nestle", "Instant Noodles", "8901058895724", 389.0, 8.0, 59.6, 13.5, 1240.0, 2.0, 1.2, 2))
        maggi_id = cursor.lastrowid
        conn.execute("""
            INSERT INTO food_products_extra (id, sat_fat_100g, ingredients_raw, source)
            VALUES (?, ?, ?, ?)
        """, (maggi_id, 7.0, "Refined wheat flour (Maida), palm oil, salt, INS 635, INS 451, Caramel color 150d", "verified_seeder"))

        # Seed Britannia Treat Jim Jam Biscuits
        cursor = conn.execute("""
            INSERT INTO food_products (
                name, brand, category, barcode, calories_100g,
                protein_100g, carbs_100g, fat_100g, sodium_100g,
                fiber_100g, sugar_100g, eatlytic_score, verified,
                scan_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)
        """, ("Treat Jim Jam Biscuits", "Britannia", "Biscuit", "8901063029170", 483.0, 5.0, 73.0, 19.0, 280.0, 1.0, 37.5, 3))
        jimjam_id = cursor.lastrowid
        conn.execute("""
            INSERT INTO food_products_extra (id, sat_fat_100g, ingredients_raw, source)
            VALUES (?, ?, ?, ?)
        """, (jimjam_id, 9.5, "Refined wheat flour (Maida), Sugar, Mixed fruit jam, Edible vegetable oil (Palm), Invert sugar syrup, Butter, Milk solids, Iodised salt, Raising agents [503(ii) & 500(ii)], Emulsifiers, Preservative (223), Acidity regulators, Synthetic food color (INS 122)", "verified_seeder"))

        # Seed Tata Iodised Salt
        cursor = conn.execute("""
            INSERT INTO food_products (
                name, brand, category, barcode, calories_100g,
                protein_100g, carbs_100g, fat_100g, sodium_100g,
                fiber_100g, sugar_100g, eatlytic_score, verified,
                scan_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)
        """, ("Iodised Salt", "Tata", "Condiment", "8901058002313", 0.0, 0.0, 0.0, 0.0, 38700.0, 0.0, 0.0, 6))
        salt_id = cursor.lastrowid
        conn.execute("""
            INSERT INTO food_products_extra (id, sat_fat_100g, ingredients_raw, source)
            VALUES (?, ?, ?, ?)
        """, (salt_id, 0.0, "Edible common salt, Potassium iodate, Anticaking agent (INS 536)", "verified_seeder"))

        # Seed Lays India's Magic Masala Chips
        cursor = conn.execute("""
            INSERT INTO food_products (
                name, brand, category, barcode, calories_100g,
                protein_100g, carbs_100g, fat_100g, sodium_100g,
                fiber_100g, sugar_100g, eatlytic_score, verified,
                scan_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)
        """, ("India's Magic Masala Chips", "Lays", "Snacks", "8901491101837", 555.0, 7.0, 52.0, 35.0, 780.0, 2.5, 2.0, 2))
        chips_id = cursor.lastrowid
        conn.execute("""
            INSERT INTO food_products_extra (id, sat_fat_100g, ingredients_raw, source)
            VALUES (?, ?, ?, ?)
        """, (chips_id, 13.5, "Potato, Edible vegetable oil (Palmolein, Rice bran), Spices & Condiments, Iodised salt, Sugar, Acidity regulators (INS 330, 296), Flavor enhancers (INS 627, 631)", "verified_seeder"))

    yield

# ══ 1. RE-BASED KEYWORD MATCH TESTS ═══════════════════════════════════
class TestDatabaseKeywordMatching:
    def test_barcode_match(self):
        """Should resolve Amul Butter using barcode extract."""
        text = "Scan complete. Product barcode is 8901262010018 on this pack."
        match = find_db_product_match(text)
        assert match is not None
        assert match["brand"] == "Amul"
        assert match["name"] == "Pasteurised Butter"

    def test_fuzzy_keyword_match(self):
        """Should resolve Amul Butter even if names/packaging terms are shuffled."""
        text = "This is a pack of premium Pasteurised Butter by Amul Dairy India."
        match = find_db_product_match(text)
        assert match is not None
        assert match["brand"] == "Amul"
        assert match["name"] == "Pasteurised Butter"

    def test_mismatch_on_different_brand(self):
        """Should return None if brand name is not in the text."""
        text = "This is Pasteurised Butter from Britannia Dairy."
        match = find_db_product_match(text)
        assert match is None

    def test_mismatch_on_insufficient_name_words(self):
        """Should not match if brand matches but name does not have 70% matching words."""
        text = "Amul Pasteurised Cheese Slices are delicious."
        match = find_db_product_match(text)
        assert match is None

# ══ 2. OFFLINE PIPELINE BYPASS TESTS ══════════════════════════════════
class TestOfflinePipelineBypass:
    @pytest.mark.asyncio
    @patch("app.services.llm.engine.call_llm")
    async def test_llm_bypass_on_amul_butter(self, mock_call_llm):
        """Verify that calling unified_analyze_flow with Amul Butter bypasses LLM entirely."""
        ocr_text = "Beautiful Golden Amul Pasteurised Butter per 100g."
        
        # Execute pipeline
        result = await unified_analyze_flow(ocr_text, persona="adult")
        
        # Verify call_llm was never invoked (proving $0 cost!)
        mock_call_llm.assert_not_called()
        
        # Validate exact schema adherence and structural accuracy
        assert result["product_name"] == "Amul Pasteurised Butter"
        assert result["score"] == 4
        assert result["calories"] == 722.0
        assert result["protein"] == 0.6
        assert result["sodium"] == 830.0
        assert result["fat"] == 80.0
        assert result["score_color"] == "#f59e0b" # Orange for caution (Score 4)
        assert result["safety_tier"] == "Limit"
        
        # Check explanation values
        explanation = result["explanation"]
        assert explanation["nova_level"] == 1 # Seeded butter has no UPF markers
        assert "%" in result["eli5_explanation"] or "verified" in result["eli5_explanation"].lower()
        
    @pytest.mark.asyncio
    @patch("app.services.llm.engine.call_llm")
    async def test_offline_additives_on_maggi_noodles(self, mock_call_llm):
        """Verify that Nestle Maggi matches offline and correctly extracts critical INS context."""
        ocr_text = "Enjoy Nestle Maggi Masala Noodles anytime!"
        
        result = await unified_analyze_flow(ocr_text, persona="diabetic")
        
        # Verify LLM bypass
        mock_call_llm.assert_not_called()
        
        # Validate data details
        assert result["product_name"] == "Nestle Maggi Masala Noodles"
        assert result["score"] == 1
        assert result["score_color"] == "#ef4444" # Red for avoid (Score 1)
        assert result["safety_tier"] == "Avoid"
        
        # Ensure offline additives are picked up via explanation_engine
        insights = result["explanation"]["humanized_insights"]
        assert any("635" in ins.lower() for ins in insights) # Disodium 5'-ribonucleotides flavor enhancer
        assert any("451" in ins.lower() for ins in insights) # HUMECTANT humectants humectantsHumectant/Stabilizer
        assert any("150d" in ins.lower() for ins in insights) # Ammonia Caramel color
        
        # Verify persona warnings are correctly integrated
        cons = result["cons"]
        assert any("Diabetics" in c for c in cons) # Maida/Sugar warning for diabetics

    @pytest.mark.asyncio
    @patch("app.services.llm.engine.call_llm")
    async def test_offline_jim_jam_biscuit(self, mock_call_llm):
        """Verify that Britannia Jim Jam matches offline, calculates Atwater calories, and flags diabetic and children warnings."""
        ocr_text = "Scan result: Britannia Treat Jim Jam mixed fruit cream biscuits. Barcode: 8901063029170."
        
        # Test under general adult persona
        result = await unified_analyze_flow(ocr_text, persona="adult")
        
        # Verify LLM bypass
        mock_call_llm.assert_not_called()
        
        # Verify barcode and product name matches
        assert result["product_name"] == "Britannia Treat Jim Jam Biscuits"
        assert result["score"] == 3
        assert result["calories"] == 483.0
        assert result["carbs"] == 73.0
        assert result["sugar"] == 37.5
        assert result["fat"] == 19.0
        assert result["protein"] == 5.0
        assert result["safety_tier"] == "Avoid"
        
        # Verify Atwater calculation validity
        assert result["extraction_confidence"]["atwater_valid"] is True
        
        # Test under Diabetic care profile (score should drop by 5 and have targeted warnings)
        result_diabetic = await unified_analyze_flow(ocr_text, persona="diabetic")
        assert result_diabetic["score"] == 1 # 3 - 5 capped at 1
        
        # Test under Parent/Children profile (must flag INS 122 synthetic dye warning)
        result_child = await unified_analyze_flow(ocr_text, persona="child")
        assert result_child["score"] == 1 # 3 - 5 capped at 1
        
        # Verify warnings include synthetic dye INS 122
        insights = result_child["explanation"]["humanized_insights"]
        assert any("122" in ins for ins in insights) # INS 122 Carmoisine warning

    @pytest.mark.asyncio
    @patch("app.services.llm.engine.call_llm")
    async def test_offline_tata_salt(self, mock_call_llm):
        """Verify that Tata Salt matches offline, handles zero-calorie, and flags sodium and INS 536 warnings."""
        ocr_text = "Seeded product check: Tata Iodised Salt 1kg packet. Barcode is 8901058002313."
        
        result = await unified_analyze_flow(ocr_text, persona="adult")
        mock_call_llm.assert_not_called()
        
        assert result["product_name"] == "Tata Iodised Salt"
        assert result["calories"] == 0.0
        assert result["score"] == 6
        assert result["sodium"] == 38700.0
        assert result["safety_tier"] == "Limit"
        
        # Check that anticaking agent INS 536 is flagged
        insights = result["explanation"]["humanized_insights"]
        assert any("536" in ins for ins in insights) # INS 536 Potassium Ferrocyanide warning

    @pytest.mark.asyncio
    @patch("app.services.llm.engine.call_llm")
    async def test_offline_lays_chips(self, mock_call_llm):
        """Verify that Lays Chips matches offline, calculates Atwater calories, and flags high fat and INS flavor enhancers."""
        ocr_text = "Scan: Lays India's Magic Masala Potato Chips. Barcode 8901491101837."
        
        result = await unified_analyze_flow(ocr_text, persona="adult")
        mock_call_llm.assert_not_called()
        
        assert result["product_name"] == "Lays India's Magic Masala Chips"
        assert result["score"] == 2
        assert result["calories"] == 555.0
        assert result["fat"] == 35.0
        assert result["sodium"] == 780.0
        assert result["safety_tier"] == "Avoid"
        assert result["extraction_confidence"]["atwater_valid"] is True
        
        # Check that flavor enhancers INS 627 and 631 are flagged
        insights = result["explanation"]["humanized_insights"]
        assert any("627" in ins or "631" in ins for ins in insights) # Flavor enhancers INS 627, 631
