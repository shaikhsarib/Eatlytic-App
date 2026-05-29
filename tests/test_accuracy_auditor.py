import pytest
from unittest.mock import patch, MagicMock
from app.database.connection import db_conn, init_db
import app.database.connection as db_mod
from app.services.explanation_engine import verify_atwater_math
from app.ai.llm.engine import find_db_product_match, unified_analyze_flow

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
    yield

class TestAccuracyAuditor:
    def test_verify_atwater_math_valid(self):
        """Test Atwater Calorie Math for accurate product data."""
        # Treat Jim Jam: Carbs: 73g, Protein: 5g, Fat: 19g -> Cal: (73*4) + (5*4) + (19*9) = 292 + 20 + 171 = 483 kcal
        nutrients = {
            "calories": 483.0,
            "carbs": 73.0,
            "protein": 5.0,
            "fat": 19.0
        }
        res = verify_atwater_math(nutrients)
        assert res["is_atwater_valid"] is True
        assert res["calculated_cal"] == 483.0
        assert res["declared_cal"] == 483.0
        assert res["variance"] == 0.0

    def test_verify_atwater_math_variance_within_tolerance(self):
        """Test that within 10% or +/- 20 kcal deviation, it is marked valid."""
        # Say, declared 470 kcal instead of 483 kcal
        nutrients = {
            "calories": 470.0,
            "carbs": 73.0,
            "protein": 5.0,
            "fat": 19.0
        }
        res = verify_atwater_math(nutrients)
        assert res["is_atwater_valid"] is True
        assert res["variance"] == 13.0

    def test_verify_atwater_math_invalid(self):
        """Test that excessive variance flags an audit failure."""
        nutrients = {
            "calories": 200.0,  # Far off from 483.0
            "carbs": 73.0,
            "protein": 5.0,
            "fat": 19.0
        }
        res = verify_atwater_math(nutrients)
        assert res["is_atwater_valid"] is False
        assert res["variance"] == 283.0

    def test_verify_atwater_math_zero_calories(self):
        """Test zero calories (like salt) passes successfully."""
        nutrients = {
            "calories": 0.0,
            "carbs": 0.0,
            "protein": 0.0,
            "fat": 0.0
        }
        res = verify_atwater_math(nutrients)
        assert res["is_atwater_valid"] is True
        assert res["calculated_cal"] == 0.0

    @pytest.mark.asyncio
    async def test_offline_database_match_flags(self):
        """Test database lookup injects eatlytic_database and atwater_audit."""
        ocr_text = "This biscuit is a Britannia Treat Jim Jam with barcode 8901063029170."
        
        # Test unified_analyze_flow triggers local match
        res = await unified_analyze_flow(ocr_text)
        assert "error" not in res
        assert res["data_source"] == "eatlytic_database"
        assert "atwater_audit" in res
        assert res["atwater_audit"]["is_atwater_valid"] is True
        assert res["atwater_audit"]["calculated_cal"] == 483.0
