import numpy as np
import pytest
from app.services.image import assess_image_quality

def test_assess_image_quality_serializable():
    """Verify that assess_image_quality returns standard native Python types (no raw NumPy types)."""
    # Create a 100x100 white block image in bytes (jpeg)
    import cv2
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    content = buf.tobytes()

    quality = assess_image_quality(content)
    
    # Assert all keys are returned
    required_keys = ["is_blurry", "blur_score", "blur_severity", "brightness", "is_dark", "is_washed_out", "should_enhance"]
    for k in required_keys:
        assert k in quality, f"Key {k} is missing from quality response"

    # Assert types are native Python types
    assert isinstance(quality["is_blurry"], bool), f"is_blurry must be native bool, got {type(quality['is_blurry'])}"
    assert isinstance(quality["blur_score"], float), f"blur_score must be native float, got {type(quality['blur_score'])}"
    assert isinstance(quality["brightness"], float), f"brightness must be native float, got {type(quality['brightness'])}"
    assert isinstance(quality["is_dark"], bool), f"is_dark must be native bool, got {type(quality['is_dark'])}"
    assert isinstance(quality["is_washed_out"], bool), f"is_washed_out must be native bool, got {type(quality['is_washed_out'])}"
    assert isinstance(quality["should_enhance"], bool), f"should_enhance must be native bool, got {type(quality['should_enhance'])}"
    assert isinstance(quality["blur_severity"], str), f"blur_severity must be native str, got {type(quality['blur_severity'])}"

    # Confirm JSON-serializable without errors
    import json
    try:
        json.dumps(quality)
    except TypeError as e:
        pytest.fail(f"Quality dictionary contains non-serializable elements: {e}")
