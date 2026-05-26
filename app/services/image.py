"""
app/services/image.py
Industrial-grade image processing: quality assessment, deblurring, and enhancement.
"""

import cv2
import numpy as np
import logging
import base64
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

def assess_image_quality(content: bytes) -> dict:
    """Assess blur and lighting quality of a nutrition label."""
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "is_blurry": True,
            "blur_score": 0.0,
            "blur_severity": "critical",
            "brightness": 0.0,
            "is_dark": True,
            "is_washed_out": False,
            "should_enhance": True,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplace variance for blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness check
    brightness = np.mean(gray)
    
    is_blurry = laplacian_var < 60
    severity = "none"
    if laplacian_var < 20: severity = "critical"
    elif laplacian_var < 45: severity = "high"
    elif laplacian_var < 60: severity = "medium"

    return {
        "is_blurry": bool(is_blurry),
        "blur_score": float(round(laplacian_var, 2)),
        "blur_severity": severity,
        "brightness": float(round(brightness, 2)),
        "is_dark": bool(brightness < 40),
        "is_washed_out": bool(brightness > 220),
        "should_enhance": bool(is_blurry or brightness < 40),
    }

def validate_image(content: bytes) -> bytes:
    """Ensure image is valid, not empty, and within size limits."""
    if not content:
        raise ValueError("Empty image content")
    
    # 10MB limit
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Image file too large (Max 10MB)")

    try:
        Image.open(BytesIO(content)).verify()
        return content
    except Exception:
        raise ValueError("Invalid image file")

def deblur_and_enhance(content: bytes, severity: str = "medium") -> tuple[bytes, str]:
    """Applies specialized filters based on blur severity."""
    img_np = np.frombuffer(content, np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    method_log = ["Original"]
    
    try:
        # 1. Sharpening
        if severity in ["high", "critical"]:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_cv = cv2.filter2D(img_cv, -1, kernel)
            method_log.append("Sharpen")

        # 2. CLAHE for contrast
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        method_log.append("CLAHE")

        # 3. Denoising
        if severity == "critical":
            img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
            method_log.append("Denoise")

        _, buf = cv2.imencode('.jpg', img_cv, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return buf.tobytes(), " → ".join(method_log)

    except Exception as e:
        logger.error("Enhancement failed: %s", e)
        return content, "fallback"

def image_to_b64(content: bytes) -> str:
    """Convert bytes to base64 string with prefix."""
    return "data:image/jpeg;base64," + base64.b64encode(content).decode()

def ocr_quality_score(ocr_result: dict) -> float:
    """Compute a combined quality score for OCR results."""
    return (
        ocr_result.get("word_count", 0) * 0.6
        + ocr_result.get("avg_confidence", 0) * 100 * 0.4
    )
