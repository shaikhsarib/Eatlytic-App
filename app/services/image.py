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
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        lap = _laplacian_score(gray)
        ten = _tenengrad_score(gray)
        bren = _brenner_score(gray)
        loc = _local_blur_map(gray)

        comp = (
            0.25 * min(lap / 300.0 * 100, 100)
            + 0.20 * min(ten / 500.0 * 100, 100)
            + 0.20 * min(bren / 200.0 * 100, 100)
            + 0.35 * min(loc / 300.0 * 100, 100)
        )

        if comp < 15:
            severity, is_blurry = "severe", True
        elif comp < 35:
            severity, is_blurry = "moderate", True
        elif comp < 50:
            severity, is_blurry = "mild", True
        else:
            severity, is_blurry = "none", False

        return {
            "blur_score": round(comp, 2),
            "laplacian_score": round(lap, 2),
            "tenengrad_score": round(ten, 2),
            "brenner_score": round(bren, 2),
            "local_median_score": round(loc, 2),
            "is_blurry": is_blurry,
            "blur_severity": severity,
            "quality": "poor" if comp < 30 else ("fair" if comp < 50 else "good"),
        }
    except Exception as exc:
        logger.error("Blur detection error: %s", exc)
        return {
            "blur_score": 0,
            "laplacian_score": 0,
            "tenengrad_score": 0,
            "brenner_score": 0,
            "local_median_score": 0,
            "is_blurry": True,
            "blur_severity": "unknown",
            "quality": "unknown",
        }


def _wiener_deconvolution(
    gray: np.ndarray, psf_size: int = 5, noise_ratio: float = 0.02
) -> np.ndarray:
    psf_size = max(3, psf_size | 1)
    psf = cv2.getGaussianKernel(psf_size, psf_size / 3.0)
    psf = psf @ psf.T
    psf /= psf.sum()
    padded = np.zeros_like(gray, dtype=np.float64)
    ph, pw = psf.shape
    padded[:ph, :pw] = psf
    padded = np.roll(np.roll(padded, -ph // 2, 0), -pw // 2, 1)
    Y = np.fft.fft2(gray.astype(np.float64) / 255.0)
    H = np.fft.fft2(padded)
    W = np.conj(H) / (np.abs(H) ** 2 + noise_ratio)
    return np.clip(np.real(np.fft.ifft2(W * Y)) * 255.0, 0, 255).astype(np.uint8)


def _unsharp_mask(
    img: np.ndarray, strength: float = 1.5, radius: int = 3
) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (radius * 2 + 1, radius * 2 + 1), 0)
    mask = cv2.subtract(img.astype(np.int16), blurred.astype(np.int16))
    return np.clip(img.astype(np.float32) + strength * mask, 0, 255).astype(np.uint8)


def _apply_clahe(img: np.ndarray, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _denoise(img: np.ndarray, h: int = 6) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(
        cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21), cv2.COLOR_BGR2RGB
    )


def deblur_and_enhance(content: bytes, severity: str = "moderate") -> tuple[bytes, str]:
    img_np = np.array(Image.open(BytesIO(content)).convert("RGB"))
    log = []

    h, w = img_np.shape[:2]
    if min(h, w) < 1200:
        s = 1200 / min(h, w)
        img_np = cv2.resize(
            img_np, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4
        )
        log.append("upscale")

    if severity in ("severe", "moderate"):
        img_np = _denoise(img_np, h=8 if severity == "severe" else 5)
        log.append("NLM")

    if severity != "mild":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        psf = 9 if severity == "severe" else 5
        kr = 0.01 if severity == "severe" else 0.025
        rest = _wiener_deconvolution(gray, psf, kr)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = rest
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        log.append(f"Wiener(psf={psf})")

    sm = {"severe": 2.2, "moderate": 1.8, "mild": 1.2}
    rm = {"severe": 4, "moderate": 3, "mild": 2}
    img_np = _unsharp_mask(img_np, sm.get(severity, 1.8), rm.get(severity, 3))
    log.append("unsharp")
    cm = {"severe": 3.0, "moderate": 2.5, "mild": 1.8}
    img_np = _apply_clahe(img_np, cm.get(severity, 2.5))
    log.append("CLAHE")
    img_np = _unsharp_mask(img_np, 1.2, 2)
    log.append("sharpen2")

    buf = BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=92)
    return buf.getvalue(), " → ".join(log)


def image_to_b64(content: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(content).decode()


def ocr_quality_score(ocr_result: dict) -> float:
    return (
        ocr_result.get("word_count", 0) * 0.6
        + ocr_result.get("avg_confidence", 0) * 100 * 0.4
    )
