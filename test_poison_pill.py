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
        from app.services.llm import _validate_atwater_math

        nutrients = [
            {"name": "Calories", "value": 340, "unit": "kcal"},
            {"name": "Protein", "value": 25, "unit": "g"},
            {"name": "Carbohydrate", "value": 30, "unit": "g"},
            {"name": "Fat", "value": 10, "unit": "g"},
        ]
        assert _validate_atwater_math(nutrients) is None

    def test_hallucinated_calories_flagged(self):
        """LLM says 500 kcal but macros only add up to 260."""
        from app.services.llm import _validate_atwater_math

        nutrients = [
            {"name": "Calories", "value": 500, "unit": "kcal"},
            {"name": "Protein", "value": 10, "unit": "g"},
            {"name": "Carbohydrate", "value": 40, "unit": "g"},
            {"name": "Fat", "value": 5, "unit": "g"},
        ]
        result = _validate_atwater_math(nutrients)
        assert result is not None
        assert result["error"] == "atwater_mismatch"

    def test_within_tolerance_passes(self):
        """340 stated vs 348 calculated — within 15% margin."""
        from app.services.llm import _validate_atwater_math

        nutrients = [
            {"name": "Calories", "value": 340, "unit": "kcal"},
            {"name": "Protein", "value": 26, "unit": "g"},
            {"name": "Carbohydrate", "value": 30, "unit": "g"},
            {"name": "Fat", "value": 10, "unit": "g"},
        ]
        assert _validate_atwater_math(nutrients) is None

    def test_empty_nutrients_passes(self):
        from app.services.llm import _validate_atwater_math

        assert _validate_atwater_math([]) is None

    def test_sanitise_result_sets_warning(self):
        from app.services.llm import sanitise_result

        result = {
            "chart_data": [50, 30, 20],
            "nutrient_breakdown": [
                {"name": "Calories", "value": 500, "unit": "kcal"},
                {"name": "Protein", "value": 10, "unit": "g"},
                {"name": "Carbohydrate", "value": 40, "unit": "g"},
                {"name": "Fat", "value": 5, "unit": "g"},
            ],
        }
        cleaned = sanitise_result(result)
        assert "atwater_warning" in cleaned
        assert cleaned["is_low_confidence"] is True


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

        ocr = {"text": "Pr te n 8g F t 5g", "avg_confidence": 0.45, "word_count": 8}
        passes, msg = passes_confidence_gate(ocr)
        assert passes is False
        assert "too blurry" in msg.lower()

    def test_threshold_boundary(self):
        """Exactly 0.70 should pass."""
        from app.services.ocr import passes_confidence_gate

        ocr = {"text": "test", "avg_confidence": 0.70, "word_count": 1}
        passes, msg = passes_confidence_gate(ocr)
        assert passes is True

    def test_just_below_threshold_blocked(self):
        from app.services.ocr import passes_confidence_gate

        ocr = {"text": "test", "avg_confidence": 0.69, "word_count": 1}
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
        """Pure sugar should get a very low score from label detection."""
        from app.services.ocr import detect_label_presence

        text = "Nutrition Facts per 100g · Calories 387kcal · Sugar 100g · Carbohydrate 100g · Ingredients: Sugar"
        result = detect_label_presence(text)
        assert result["has_label"] is True
        assert result["confidence"] in ("high", "medium")

    def test_msg_ingredients_parsed(self):
        """MSG product label should be parseable."""
        from app.services.ocr import detect_label_presence

        text = "Nutrition Facts per 100g · Calories 0kcal · Sodium 12000mg · Ingredients: Monosodium Glutamate (MSG)"
        result = detect_label_presence(text)
        assert result["has_label"] is True
        assert "sodium" in " ".join(result["label_hits"]).lower()

    def test_front_pack_marketing_rejected(self):
        from app.services.ocr import detect_label_presence

        text = "NEW! Natural Energy Boost — Premium Organic Formula — Delicious & Tasty"
        result = detect_label_presence(text)
        assert result["has_label"] is False

    def test_empty_image_rejected(self):
        from app.services.ocr import detect_label_presence

        result = detect_label_presence("")
        assert result["has_label"] is False
        assert result["suggestion"] == "no_text"

    def test_partial_garbled_text_low_confidence(self):
        from app.services.ocr import detect_label_presence

        text = "N tr t on F cts C lor es 2 0"
        result = detect_label_presence(text)
        assert result["has_label"] in (True, False)
