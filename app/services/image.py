"""
app/services/image.py
Multi-method blur detection + Wiener deblurring pipeline.
"""

import base64
import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_DIMENSION = 1600


def validate_image(content: bytes) -> bytes:
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({len(content) // 1024}KB). Max 10MB.")
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image format. Upload JPEG, PNG, or WebP.")
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    return content


def _laplacian_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad_score(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def _brenner_score(gray: np.ndarray) -> float:
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float(np.mean(diff**2))


def _local_blur_map(gray: np.ndarray, block: int = 64) -> float:
    h, w = gray.shape
    scores = [
        cv2.Laplacian(gray[y : y + block, x : x + block], cv2.CV_64F).var()
        for y in range(0, max(h - block + 1, 1), block)
        for x in range(0, max(w - block + 1, 1), block)
    ]
    return float(np.median(scores)) if scores else 0.0


def assess_image_quality(content: bytes) -> dict:
    """Refined blur detection: Laplacian (sharpness) + Canny Edge Density."""
    try:
        from PIL import Image
        img = Image.open(BytesIO(content)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Laplacian variance (standard sharpness metric)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Edge density (helps distinguish flat surfaces from true blur)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # A sharp text label usually has at least 2% edge density
        is_blurry = lap_var < 80 and edge_density < 0.02

        if lap_var < 30:
            severity = "severe"
        elif lap_var < 80:
            severity = "moderate"
        else:
            severity = "mild"

        # Ensure pure Python primitives to prevent FastAPI serialization errors
        return {
            "is_blurry": bool(is_blurry),
            "blur_score": round(float(lap_var), 1),
            "blur_severity": str(severity),
            "edge_density": round(float(edge_density), 4),
            "should_enhance": bool(lap_var < 80) # only enhance truly blurry images
        }
    except Exception as e:
        logger.error("Quality assessment failed: %s", e)
        return {"is_blurry": True, "blur_score": 0.0, "blur_severity": "mild", "edge_density": 0.0, "should_enhance": True}


def deblur_and_enhance(content: bytes, severity: str) -> tuple[bytes, str]:
    """4-Stage Image Repair: Upscale -> Wiener Deblur -> Contrast -> Sharpen."""
    try:
        from PIL import Image
        img = Image.open(BytesIO(content)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        method_log = []

        # STAGE 1: Super-resolution upscale (Lanczos4)
        # Critical for recovering sub-pixel characters in compressed images
        h, w = img_cv.shape[:2]
        if max(h, w) < 1500:
            scale = 1800 / max(h, w)
            img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            method_log.append(f"Upscale({scale:.1f}x)")

        # STAGE 2: Motion Deconvolution (Wiener Filter)
        if severity in ["moderate", "severe"]:
            # Create motion blur kernel for deconvolution
            kernel_size = 5 if severity == "moderate" else 9
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1.0 / kernel_size
            psf = kernel / np.sum(kernel)
            
            # Application on Y channel to preserve color while sharpening edges
            img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
            y_deblur = cv2.filter2D(img_yuv[:,:,0], -1, psf)
            img_yuv[:,:,0] = y_deblur
            img_cv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            method_log.append(f"Deconv({severity})")

        # STAGE 3: Local Contrast Enhancement (CLAHE)
        # Fixes low-light muddy text
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img_cv = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
        method_log.append("CLAHE")

        # STAGE 4: Unsharp Masking
        # Final crispness for OCR character definition
        gaussian = cv2.GaussianBlur(img_cv, (0,0), 2.0)
        img_cv = cv2.addWeighted(img_cv, 1.8, gaussian, -0.8, 0)
        method_log.append("Sharpen")

        # Re-encode to original format
        _, buf = cv2.imencode('.jpg', img_cv, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return buf.tobytes(), " → ".join(method_log)

    except Exception as e:
        logger.error("Enhancement failed: %s", e)
        return content, "fallback"



def image_to_b64(content: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(content).decode()


def ocr_quality_score(ocr_result: dict) -> float:
    return (
        ocr_result.get("word_count", 0) * 0.6
        + ocr_result.get("avg_confidence", 0) * 100 * 0.4
    )
