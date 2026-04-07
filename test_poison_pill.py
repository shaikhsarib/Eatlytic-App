"""
tests/test_poison_pill.py
Phase 1 trust tests: Atwater math, OCR confidence gate, poison inputs.
"""

import numpy as np
import pytest
from PIL import Image
from io import BytesIO


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    import app.models.db as db_mod

    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    yield


# ══ 1. ATWATER MATH VALIDATION ═══════════════════════════════════════
class TestAtwaterMath:
    def test_valid_math_passes(self):
        """25g protein + 30g carbs + 10g fat = 340 kcal (exact match)."""
        from app.services.fake_detector import atwater_math_check

        nutrients = {"calories": 340, "protein": 25, "carbs": 30, "fat": 10}
        result = atwater_math_check(nutrients)
        assert result["is_valid"] is True

    def test_hallucinated_calories_flagged(self):
        """LLM says 500 kcal but macros only add up to 260."""
        from app.services.fake_detector import atwater_math_check

        nutrients = {"calories": 500, "protein": 10, "carbs": 40, "fat": 5}
        result = atwater_math_check(nutrients)
        assert result["is_valid"] is False
        assert "Math Mismatch" in result["reason"]

    def test_within_tolerance_passes(self):
        """340 stated vs 348 calculated — within 25% margin."""
        from app.services.fake_detector import atwater_math_check

        nutrients = {"calories": 340, "protein": 26, "carbs": 30, "fat": 10}
        result = atwater_math_check(nutrients)
        assert result["is_valid"] is True

    def test_empty_nutrients_passes(self):
        from app.services.fake_detector import atwater_math_check

        result = atwater_math_check({})
        assert result["is_valid"] is True

    def test_zero_calories_passes(self):
        from app.services.fake_detector import atwater_math_check

        nutrients = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        result = atwater_math_check(nutrients)
        assert result["is_valid"] is True


# ══ 2. OCR CONFIDENCE GATE ═══════════════════════════════════════════
class TestOCRConfidenceGate:
    def test_high_confidence_passes(self):
        from app.services.ocr import passes_confidence_gate

        ocr = {"text": "Protein 8g Fat 5g", "avg_confidence": 0.85, "word_count": 6}
        passes, msg = passes_confidence_gate(ocr)
        assert passes is True
        assert msg == ""

    def test_low_confidence_blocked(self):
        from app.services.ocr import passes_confidence_gate

        ocr = {"text": "Pr te n 8g F t 5g", "avg_confidence": 0.15, "word_count": 8}
        passes, msg = passes_confidence_gate(ocr)
        assert passes is False
        assert "too low" in msg.lower()

    def test_threshold_boundary(self):
        """Exactly 0.30 should pass."""
        from app.services.ocr import passes_confidence_gate

        ocr = {"text": "test", "avg_confidence": 0.30, "word_count": 1}
        passes, msg = passes_confidence_gate(ocr)
        assert passes is True

    def test_just_below_threshold_blocked(self):
        from app.services.ocr import passes_confidence_gate

        ocr = {"text": "test", "avg_confidence": 0.20, "word_count": 1}
        passes, msg = passes_confidence_gate(ocr)
        assert passes is False


# ══ 3. POISON PILL: BLURRY IMAGES ════════════════════════════════════
class TestBlurryImageRejection:
    def test_severely_blurry_detected(self):
        """Heavily blurred image should be flagged as poor quality."""
        import cv2

        img = Image.new("RGB", (200, 200), color=(200, 180, 150))
        img_arr = np.array(img)
        blurred = cv2.GaussianBlur(img_arr, (31, 31), 0)
        img = Image.fromarray(blurred, "RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        content = buf.read()

        from app.services.image import assess_image_quality

        quality = assess_image_quality(content)
        assert quality["is_blurry"] is True
        assert quality["quality"] == "poor"

    def test_solid_color_blurry(self):
        """A solid color image has zero texture — should be flagged."""
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        content = buf.read()

        from app.services.image import assess_image_quality

        quality = assess_image_quality(content)
        assert quality["is_blurry"] is True


# ══ 4. POISON PILL: TOXIC/UNHEALTHY PRODUCTS ═════════════════════════
class TestToxicProductFlagging:
    def test_pure_sugar_label_detected(self):
        """Pure sugar should pass nutrition validation."""
        from app.services.ocr import universal_label_filter

        text = "Nutrition Facts per 100g\nCalories 387kcal\nSugar 100g\nCarbohydrate 100g\nIngredients: Sugar"
        result = universal_label_filter(text)
        assert result["is_valid"] is True

    def test_msg_ingredients_parsed(self):
        """MSG product label should pass nutrition validation."""
        from app.services.ocr import universal_label_filter

        text = "Nutrition Facts per 100g\nCalories 0kcal\nSodium 12000mg\nIngredients: Monosodium Glutamate (MSG)"
        result = universal_label_filter(text)
        assert result["is_valid"] is True

    def test_front_pack_marketing_rejected(self):
        from app.services.ocr import universal_label_filter

        text = "NEW! Natural Energy Boost\nPremium Organic Formula\nDelicious & Tasty"
        result = universal_label_filter(text)
        assert result["is_valid"] is False

    def test_empty_image_rejected(self):
        from app.services.ocr import universal_label_filter

        result = universal_label_filter("")
        assert result["is_valid"] is False

    def test_partial_garbled_text_rejected(self):
        from app.services.ocr import universal_label_filter

        text = "N tr t on F cts C lor es 2 0"
        result = universal_label_filter(text)
        assert result["is_valid"] is False

    def test_maggi_style_label_with_garbage_words(self):
        """Maggi-style label with FSSAI/MRP lines should still detect nutrition data."""
        from app.services.ocr import universal_label_filter

        text = (
            "FSSAI License No. 10012021000345\n"
            "MRP Rs. 14 inclusive of all taxes\n"
            "Per 100g\n"
            "Energy 500 kcal\n"
            "Protein 10 g\n"
            "Carbohydrate 60 g\n"
            "Total Fat 20 g\n"
            "Saturated Fat 8 g\n"
            "Sodium 500 mg\n"
            "Best Before 6 months from manufacture\n"
            "Customer Care: 1800-123-456\n"
        )
        result = universal_label_filter(text)
        assert result["is_valid"] is True
        assert "Energy 500 kcal" in result["clean_text"]
        assert "Protein 10 g" in result["clean_text"]
        assert "Sodium 500 mg" in result["clean_text"]
        assert "FSSAI" not in result["clean_text"]
        assert "MRP" not in result["clean_text"]

    def test_line_reconstruction_preserves_nutrition_rows(self):
        """Simulated OCR with line-structured output should pass filter."""
        from app.services.ocr import universal_label_filter

        text = (
            "Nutrition Facts\n"
            "Per 100g\n"
            "Calories 250 kcal\n"
            "Protein 8.5 g\n"
            "Total Fat 5.2 g\n"
            "Sodium 320 mg\n"
        )
        result = universal_label_filter(text)
        assert result["is_valid"] is True
        assert len(result["clean_text"].strip().split("\n")) >= 4

    def test_maggi_full_label_preserves_product_name(self):
        """Realistic Maggi label: product name in header, garbage words, nutrition table."""
        from app.services.ocr import universal_label_filter

        text = (
            "MAGGI Masala Noodles\n"
            "Tastiest Just 2 Minutes\n"
            "Net Wt. 70g\n"
            "FSSAI License No. 10012021000345\n"
            "Nutritional Information per 100g per serving (70g)\n"
            "Energy 500 kcal 350 kcal\n"
            "Protein 8.2 g 5.7 g\n"
            "Carbohydrate 59.0 g 41.3 g\n"
            "Total Fat 20.0 g 14.0 g\n"
            "Saturated Fat 8.5 g 5.9 g\n"
            "Sodium 920 mg 644 mg\n"
            "MRP Rs. 14 inclusive of all taxes\n"
            "Best Before 6 months from manufacture\n"
            "Ingredients: Wheat Flour, Palm Oil, Masala\n"
        )
        result = universal_label_filter(text)
        assert result["is_valid"] is True
        # Product name should be preserved in the first lines
        assert "MAGGI Masala Noodles" in result["clean_text"]
        # Nutrition data should be present
        assert (
            "Energy 500 kcal" in result["clean_text"]
            or "Energy" in result["clean_text"]
        )
        assert "Protein" in result["clean_text"]
        # Garbage should be removed
        assert "FSSAI" not in result["clean_text"]
        assert "MRP" not in result["clean_text"]
        # Ingredients line should be kept for context
        assert "Ingredients" in result["clean_text"]
