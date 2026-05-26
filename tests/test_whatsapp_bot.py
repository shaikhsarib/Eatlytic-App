import pytest
import httpx
from unittest.mock import patch, MagicMock
from main import app
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
@patch("app.routes.scan.assess_image_quality")
@patch("app.routes.scan.deblur_and_enhance")
@patch("app.routes.scan.process_image_for_ocr")
@patch("app.routes.scan.run_ocr")
@patch("app.routes.scan.unified_analyze_flow")
@patch("httpx.AsyncClient.get")
async def test_whatsapp_webhook_no_media(
    mock_get, mock_analyze, mock_ocr, mock_crop, mock_enhance, mock_assess
):
    """Test that incoming WhatsApp messages without a photo receive instructions."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/whatsapp",
            data={
                "From": "whatsapp:+919876543210",
                "Body": "Hello!"
            }
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "Message" in response.text
    assert "🩺 *Eatlytic Diabetic Safety Scanner*" in response.text

@pytest.mark.asyncio
@patch("app.routes.scan.assess_image_quality")
@patch("app.routes.scan.deblur_and_enhance")
@patch("app.routes.scan.process_image_for_ocr")
@patch("app.routes.scan.run_ocr")
@patch("app.routes.scan.unified_analyze_flow")
@patch("httpx.AsyncClient.get")
async def test_whatsapp_webhook_with_media_clear(
    mock_get, mock_analyze, mock_ocr, mock_crop, mock_enhance, mock_assess
):
    """Test standard WhatsApp scan flow with a clear label image attachment."""
    # 1. Mock Twilio image download
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fake_image_bytes"
    
    # We must patch get on httpx.AsyncClient so it doesn't try to actually fetch the image
    async def mock_get_content(*args, **kwargs):
        return mock_resp
    mock_get.side_effect = mock_get_content

    # 2. Mock quality checks (clear image)
    mock_assess.return_value = {
        "is_blurry": False,
        "blur_severity": "none",
        "blur_score": 90,
        "should_enhance": False
    }

    # 3. Mock cropper
    mock_crop.return_value = b"fake_cropped_bytes"

    # 4. Mock OCR results
    mock_ocr.return_value = {
        "text": "Nutrition Facts: Sugar 20g, Carbs 50g",
        "avg_confidence": 0.88,
        "word_count": 8,
        "language": "en"
    }

    # 5. Mock diabetic analysis engine
    mock_analyze.return_value = {
        "product_name": "Test Snack",
        "score": 4,
        "safety_tier": "Avoid",
        "safety_verdict": "High Glycemic Threat",
        "safety_reason": "High sugar causes insulin spikes.",
        "better_alternative": "Dry Roasted Almonds",
        "nutrient_breakdown": [
            {"name": "Sugar", "value": 20, "unit": "g", "rating": "bad"},
            {"name": "Total Carbohydrate", "value": 50, "unit": "g", "rating": "moderate"}
        ],
        "explanation": {
            "verdict": "🔴 RED (HIGH)",
            "nova_level": 4,
            "humanized_insights": [
                "🍬 Contains ~5 teaspoons of sugar per 100g."
            ]
        }
    }

    # 6. Send POST request
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/whatsapp",
            data={
                "From": "whatsapp:+919876543210",
                "Body": "",
                "MediaUrl0": "https://example.com/media/label.jpg",
                "MediaContentType0": "image/jpeg"
            }
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    
    # Verify XML content contains the Tier 1 summary & custom alternative swap
    assert "<Response>" in response.text
    assert "Test Snack" in response.text
    assert "Avoid" in response.text
    assert "Dry Roasted Almonds" in response.text
