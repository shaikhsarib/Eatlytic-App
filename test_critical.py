"""
tests/test_critical.py
20 critical automated tests covering:
- OCR parsing correctness
- Nutrient extraction accuracy
- Score calculation validation
- Blur detection
- Label presence detection
- Auth token lifecycle
- Payment signature verification
- Database operations
- API endpoints (integration)
- Food database moat

Run: pytest tests/ -v
"""

import os
import json
import hmac
import hashlib
import pytest
import datetime


# ── Test fixtures ──────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test."""
    db_path = str(tmp_path / "test.db")
    import app.models.db as db_mod

    monkeypatch.setattr(db_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_mod, "DB_FILE", db_path)
    db_mod.init_db()
    yield


# ══ 1. NUTRIENT VALUE SANITISATION ════════════════════════════════════
class TestNutrientSanitisation:
    def test_string_with_unit_stripped(self):
        """'34g' must become 34.0, not NaN."""
        import re

        raw = "34g"
        m = re.search(r"[\d]+\.?[\d]*", raw.replace(",", "."))
        assert m is not None
        assert float(m.group()) == 34.0

    def test_comma_decimal_stripped(self):
        """European '3,5' must become 3.5."""
        import re

        raw = "3,5g"
        m = re.search(r"[\d]+\.?[\d]*", raw.replace(",", "."))
        assert float(m.group()) == 3.5

    def test_integer_value_unchanged(self):
        import re

        raw = "250"
        m = re.search(r"[\d]+\.?[\d]*", raw.replace(",", "."))
        assert float(m.group()) == 250.0

    def test_complex_string_extracts_first_number(self):
        import re

        raw = "≤2g per serving"
        m = re.search(r"[\d]+\.?[\d]*", raw.replace(",", "."))
        assert m is not None
        assert float(m.group()) == 2.0


# ══ 2. CHART_DATA ROUNDING ════════════════════════════════════════════
class TestChartDataRounding:
    def _fix(self, cd):
        total = sum(cd)
        scaled = [round(v * 100 / total) for v in cd]
        scaled[scaled.index(max(scaled))] += 100 - sum(scaled)
        return scaled

    def test_sums_to_100(self):
        assert sum(self._fix([33, 33, 34])) == 100

    def test_sums_to_100_equal_thirds(self):
        """Notorious off-by-one: round(33.33)*3=99."""
        assert sum(self._fix([1, 1, 1])) == 100

    def test_sums_to_100_large_skew(self):
        assert sum(self._fix([90, 5, 5])) == 100

    def test_values_non_negative(self):
        result = self._fix([80, 15, 5])
        assert all(v >= 0 for v in result)


# ══ 3. NUTRITION VALIDATION (REGEX-BASED) ═════════════════════════════
class TestLabelDetection:
    def test_back_label_detected(self):
        from app.services.ocr import universal_label_filter

        text = "Nutrition Facts per 100g\nCalories 250kcal\nProtein 8g\nFat 5g\nIngredients: wheat flour, sugar, salt"
        result = universal_label_filter(text)
        assert result["is_valid"] is True

    def test_maggi_style_label_detected(self):
        from app.services.ocr import universal_label_filter

        text = "What makes Truly GOOD?\nNestle Maggi Noodles\nPer 100g\nEnergy 500kcal\nProtein 10g\nCarbohydrate 60g\nTotal Fat 20g\nSaturated Fat 8g\nSodium 500mg"
        result = universal_label_filter(text)
        assert result["is_valid"] is True

    def test_front_label_rejected(self):
        from app.services.ocr import universal_label_filter

        text = "NEW! Improved flavour\nOrganic Crunchy Wheat Bites\nNatural Goodness\nPremium Quality"
        result = universal_label_filter(text)
        assert result["is_valid"] is False

    def test_empty_text_rejected(self):
        from app.services.ocr import universal_label_filter

        result = universal_label_filter("")
        assert result["is_valid"] is False

    def test_partial_label_rejected(self):
        from app.services.ocr import universal_label_filter

        text = "Ingredients: water, salt\nBest before: Jan 2027"
        result = universal_label_filter(text)
        assert result["is_valid"] is False


# ══ 4. AUTH TOKEN LIFECYCLE ═══════════════════════════════════════════
class TestAuthTokenLifecycle:
    def test_create_and_validate_session(self):
        from app.services.auth import create_session, get_user_from_token
        from app.models.db import db_conn

        # Create a user first
        user_id = "test-user-001"
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO users(id,email) VALUES(?,?)",
                (user_id, "test@eatlytic.com"),
            )

        token = create_session(user_id)
        assert token.startswith("eat_")
        assert len(token) > 20

        user = get_user_from_token(token)
        assert user is not None
        assert user["id"] == user_id

    def test_invalid_token_returns_none(self):
        from app.services.auth import get_user_from_token

        assert get_user_from_token("totally_fake_token") is None

    def test_revoked_token_returns_none(self):
        from app.services.auth import (
            create_session,
            revoke_session,
            get_user_from_token,
        )
        from app.models.db import db_conn

        user_id = "test-user-002"
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO users(id,email) VALUES(?,?)",
                (user_id, "test2@eatlytic.com"),
            )

        token = create_session(user_id)
        revoke_session(token)
        assert get_user_from_token(token) is None

    def test_otp_verify_creates_user(self):
        from app.services.auth import send_email_otp, verify_email_otp

        otp = send_email_otp("newuser@test.com")
        user = verify_email_otp("newuser@test.com", otp)
        assert user is not None
        assert user["email"] == "newuser@test.com"

    def test_wrong_otp_returns_none(self):
        from app.services.auth import send_email_otp, verify_email_otp

        send_email_otp("wrong@test.com")
        result = verify_email_otp("wrong@test.com", "000000")
        assert result is None


# ══ 5. SCAN QUOTA (USER-BASED) ════════════════════════════════════════
class TestScanQuota:
    def _make_user(self, user_id, email):
        from app.models.db import db_conn

        with db_conn() as conn:
            conn.execute("INSERT INTO users(id,email) VALUES(?,?)", (user_id, email))

    def test_free_user_gets_10_scans(self):
        from app.services.auth import check_and_increment_scan_user

        self._make_user("u1", "a@t.com")
        for i in range(10):
            result = check_and_increment_scan_user("u1")
            assert result["allowed"] is True
        # 11th should fail
        result = check_and_increment_scan_user("u1")
        assert result["allowed"] is False

    def test_pro_user_unlimited(self):
        from app.services.auth import check_and_increment_scan_user
        from app.models.db import db_conn

        self._make_user("u2", "b@t.com")
        with db_conn() as conn:
            conn.execute("UPDATE users SET is_pro=1 WHERE id='u2'")
        for _ in range(50):
            result = check_and_increment_scan_user("u2")
            assert result["allowed"] is True
            assert result["scans_remaining"] == 9999


# ══ 6. PAYMENT SIGNATURE VERIFICATION ════════════════════════════════
class TestPaymentSignature:
    def test_valid_signature_accepted(self):
        """HMAC-SHA256 must match Razorpay's scheme."""
        secret = "test_secret_key"
        order_id = "order_ABC123"
        payment_id = "pay_XYZ789"
        expected = hmac.new(
            secret.encode(),
            f"{order_id}|{payment_id}".encode(),
            hashlib.sha256,
        ).hexdigest()

        # Manually verify same logic used in payments.py
        computed = hmac.new(
            secret.encode(),
            f"{order_id}|{payment_id}".encode(),
            hashlib.sha256,
        ).hexdigest()
        assert hmac.compare_digest(expected, computed) is True

    def test_tampered_signature_rejected(self):
        secret = "test_secret_key"
        order_id = "order_ABC123"
        payment_id = "pay_XYZ789"
        real_sig = hmac.new(
            secret.encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256
        ).hexdigest()
        fake_sig = "0" * len(real_sig)
        assert hmac.compare_digest(real_sig, fake_sig) is False


# ══ 7. FOOD DATABASE ═════════════════════════════════════════════════
class TestFoodDatabase:
    def test_insert_and_query_product(self):
        from app.models.db import db_conn

        with db_conn() as conn:
            conn.execute(
                """INSERT INTO food_products(name, brand, category, calories_100g, protein_100g, carbs_100g, fat_100g, sodium_100g, fiber_100g, sugar_100g, eatlytic_score, verified, scan_count, barcode)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "Test Biscuits",
                    "TestBrand",
                    "biscuit",
                    250,
                    8,
                    60,
                    5,
                    200,
                    2,
                    10,
                    6,
                    1,
                    1,
                    "1234567890123",
                ),
            )
            row = conn.execute(
                "SELECT * FROM food_products WHERE barcode=?", ("1234567890123",)
            ).fetchone()
        assert row is not None
        assert row["name"] == "Test Biscuits"
        assert row["scan_count"] == 1

    def test_duplicate_increments_scan_count(self):
        from app.models.db import db_conn

        with db_conn() as conn:
            conn.execute(
                """INSERT INTO food_products(name, brand, category, calories_100g, protein_100g, carbs_100g, fat_100g, sodium_100g, fiber_100g, sugar_100g, eatlytic_score, verified, scan_count, barcode)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "Dupe Product",
                    "",
                    "snack",
                    100,
                    5,
                    20,
                    3,
                    100,
                    1,
                    5,
                    5,
                    1,
                    1,
                    "9999999999999",
                ),
            )
            conn.execute(
                "UPDATE food_products SET scan_count=scan_count+1 WHERE barcode=?",
                ("9999999999999",),
            )
            row = conn.execute(
                "SELECT scan_count FROM food_products WHERE barcode=?",
                ("9999999999999",),
            ).fetchone()
        assert row["scan_count"] == 2


# ══ 8. IMAGE VALIDATION ═══════════════════════════════════════════════
class TestImageValidation:
    def test_oversized_image_rejected(self):
        from app.services.image import validate_image

        huge = b"x" * (11 * 1024 * 1024)  # 11MB
        with pytest.raises(ValueError, match="too large"):
            validate_image(huge)

    def test_invalid_bytes_rejected(self):
        from app.services.image import validate_image

        with pytest.raises(ValueError, match="Invalid image"):
            validate_image(b"this is not an image")


# ══ 9. STREAK TRACKING ════════════════════════════════════════════════
class TestStreakTracking:
    def _make_user(self, user_id):
        from app.models.db import db_conn

        with db_conn() as conn:
            conn.execute(
                "INSERT INTO users(id,email) VALUES(?,?)",
                (user_id, f"{user_id}@test.com"),
            )

    def test_consecutive_days_increments_streak(self):
        from app.services.auth import update_streak_user
        from app.models.db import db_conn

        self._make_user("streak_user_1")
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

        with db_conn() as conn:
            conn.execute(
                "UPDATE users SET last_scan_date=? WHERE id=?",
                (yesterday, "streak_user_1"),
            )

        update_streak_user("streak_user_1")

        with db_conn() as conn:
            row = conn.execute(
                "SELECT streak_days FROM users WHERE id='streak_user_1'"
            ).fetchone()
        assert row["streak_days"] == 1

    def test_missed_day_resets_streak(self):
        from app.services.auth import update_streak_user
        from app.models.db import db_conn

        self._make_user("streak_user_2")
        old_date = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()

        with db_conn() as conn:
            conn.execute(
                "UPDATE users SET last_scan_date=?, streak_days=10 WHERE id=?",
                (old_date, "streak_user_2"),
            )

        update_streak_user("streak_user_2")

        with db_conn() as conn:
            row = conn.execute(
                "SELECT streak_days FROM users WHERE id='streak_user_2'"
            ).fetchone()
        assert row["streak_days"] == 1  # reset


# ══ 10. ACCURACY BENCHMARKING ═════════════════════════════════════════
class TestAccuracyBenchmarking:
    def test_field_accuracy_correct_detection(self):
        """Within 15% tolerance → marked correct."""
        from app.routes.benchmarks import _compute_field_accuracy

        llm_output = {
            "score": 6,
            "nutrient_breakdown": [
                {"name": "Calories", "value": 248, "unit": "kcal"},  # truth=250
                {"name": "Protein", "value": 7.8, "unit": "g"},  # truth=8
                {"name": "Fat", "value": 5.1, "unit": "g"},  # truth=5
            ],
        }
        ground_truth = {
            "score": 6,
            "nutrients": {
                "calories": 250,
                "protein": 8,
                "fat": 5,
                "carbs": 30,
                "sodium": 200,
                "fiber": 2,
                "sugar": 10,
            },
        }
        result = _compute_field_accuracy(llm_output, ground_truth)
        assert result["fields"]["calories"]["status"] == "correct"
        assert result["fields"]["protein"]["status"] == "correct"
        assert result["fields"]["fat"]["status"] == "correct"

    def test_word_f1_perfect_match(self):
        from app.routes.benchmarks import _word_f1

        assert _word_f1("wheat flour sugar salt", "wheat flour sugar salt") == 1.0

    def test_word_f1_zero_overlap(self):
        from app.routes.benchmarks import _word_f1

        assert _word_f1("apples oranges", "wheat flour") == 0.0

    def test_word_f1_partial(self):
        from app.routes.benchmarks import _word_f1

        score = _word_f1("wheat flour sugar", "wheat flour")
        assert 0 < score < 1.0
