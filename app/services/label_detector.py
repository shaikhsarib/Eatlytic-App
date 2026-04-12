"""
app/services/label_detector.py
Improved ROI detection — finds the nutrition table region in real product photos.
Strategy: score every candidate rectangle by text density, prefer table-like regions.
"""

import cv2
import numpy as np
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def _score_region(gray_crop: np.ndarray) -> float:
    """
    Score a candidate region for 'nutrition table likelihood'.
    Criteria: high horizontal line density + lots of small text blobs.
    """
    if gray_crop.size == 0:
        return 0.0
    h, w = gray_crop.shape
    if h < 40 or w < 60:
        return 0.0

    # Horizontal edge density (tables have many horizontal lines)
    edges = cv2.Canny(gray_crop, 50, 150)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
    line_score = np.sum(horiz_lines > 0) / (h * w + 1e-6)

    # Text blob count via connected components
    _, bw = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    small_blobs = sum(
        1 for i in range(1, n_labels)
        if 4 <= stats[i, cv2.CC_STAT_WIDTH] <= 40
        and 4 <= stats[i, cv2.CC_STAT_HEIGHT] <= 40
    )
    blob_density = small_blobs / (h * w / 100.0 + 1e-6)

    return line_score * 40 + blob_density * 60


def get_nutrition_table_roi(image_np: np.ndarray) -> np.ndarray:
    """
    Find the nutrition facts table in a real-world product photo.
    Returns the best candidate crop, or full image on failure.
    """
    try:
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # ── Pass 1: look for white/light rectangular blocks (nutrition tables
        #   are almost always printed on a white or light background panel) ──
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < (h * w * 0.005):       # ≥0.5% of image (was 2%)
                continue
            if area > (h * w * 0.99):        # reject strictly full-image blobs
                continue
            ar = cw / float(ch)
            if not (0.1 < ar < 10.0):        # wider ratio range for tall/narrow tables
                continue
            candidates.append((x, y, cw, ch))

        # ── Pass 2: if Pass 1 found nothing, try dark-background labels ──
        if not candidates:
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
            closed2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel2, iterations=2)
            cnts2, _ = cv2.findContours(closed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts2:
                x, y, cw, ch = cv2.boundingRect(c)
                area = cw * ch
                if area < (h * w * 0.005) or area > (h * w * 0.99):
                    continue
                if not (0.1 < cw / float(ch) < 10.0):
                    continue
                candidates.append((x, y, cw, ch))

        # ── Score each candidate and pick best ────────────────────────────
        best_score = -1.0
        best_roi = image_np
        for (x, y, cw, ch) in candidates:
            # Add a small margin
            pad = 8
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad);  y2 = min(h, y + ch + pad)
            crop = image_np[y1:y2, x1:x2]
            gray_crop = gray[y1:y2, x1:x2]
            score = _score_region(gray_crop)
            if score > best_score:
                best_score = score
                best_roi = crop

        # Only use crop if score is meaningful — low scores mean no clear table found
        if best_score < 0.03:
            logger.info("No confident ROI found (score=%.3f), using full image", best_score)
            return image_np
        # Sanity check: the crop should be at least 20% of the full image area
        # to avoid returning a tiny irrelevant region
        if best_roi.shape[0] * best_roi.shape[1] < h * w * 0.05:
            logger.info("ROI too small (%.1f%%), using full image", 
                       best_roi.shape[0]*best_roi.shape[1]/(h*w)*100)
            return image_np

        logger.info("ROI found (score=%.3f)", best_score)
        return best_roi

    except Exception as e:
        logger.warning("ROI detection failed: %s — using full image", e)
        return image_np


def deskew_image(image_np: np.ndarray) -> np.ndarray:
    """Auto-rotate the image to horizontal using text baseline detection."""
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return image_np
        angle = cv2.minAreaRect(coords)[-1]
        if angle > 45:
            angle = angle - 90
        # Only deskew small angles — large angles are probably real rotations
        if abs(angle) > 20:
            return image_np
        (h, w) = image_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        logger.warning("Deskewing failed: %s", e)
        return image_np


def enhance_for_ocr(image_np: np.ndarray) -> np.ndarray:
    """
    Improve contrast and sharpness for OCR on a label crop.
    Works on both light and dark backgrounds.
    """
    try:
        # CLAHE for local contrast enhancement
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.merge([l_ch, a_ch, b_ch])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Mild unsharp mask for sharpness
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2)
        sharp = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
        return sharp
    except Exception:
        return image_np


def process_image_for_ocr(content: bytes) -> bytes:
    """
    Full pipeline: ROI Detection → Deskew → Enhance → bytes.
    On any failure, returns the original image bytes.
    """
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        img_np = np.array(img)

        roi = get_nutrition_table_roi(img_np)
        deskewed = deskew_image(roi)
        enhanced = enhance_for_ocr(deskewed)

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buf.tobytes() if ok else content
    except Exception as e:
        logger.error("Image pipeline failed: %s", e)
        return content
