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
    Find the most text-dense region (Maximally Stable Extremal Regions) — 
    works for any label worldwide regardless of color/layout.
    """
    try:
        h, w = image_np.shape[:2]

        # Always upscale to minimum 1200px for OCR accuracy
        if max(h, w) < 1200:
            scale = 1200 / max(h, w)
            image_np = cv2.resize(image_np, None, fx=scale, fy=scale, 
                                interpolation=cv2.INTER_LANCZOS4)
            h, w = image_np.shape[:2]

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # ── Step 1: MSER detection (find candidate text characters) ──
        # delta=5 is sensitivity, min_area avoids noise, max_area avoids large blobs
        mser = cv2.MSER_create(5, 60, 14400)
        regions, _ = mser.detectRegions(gray)

        if not regions:
            logger.info("Universal ROI: No text regions detected, using full image.")
            return image_np

        # ── Step 2: Build text density heatmap ──
        heatmap = np.zeros(gray.shape, dtype=np.uint8)
        for region in regions:
            # Convex hull of the region helps fill the heatmap accurately
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.fillConvexPoly(heatmap, hull, 255)

        # ── Step 3: Gaussian Blur to merge nearby text clusters into blocks ──
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

        # ── Step 4: Find the densest cluster (the nutrition facts table) ──
        _, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # The nutrition table is usually the largest dense text block on the packet
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)

            # Add 20% margin to prevent cutting off labels at the edges
            pad = int(max(cw, ch) * 0.15)
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad);  y2 = min(h, y + ch + pad)

            roi = image_np[y1:y2, x1:x2]
            
            # Sanity check: Ensure ROI is at least 15% of the original image area
            if roi.shape[0] * roi.shape[1] > h * w * 0.15:
                logger.info("Universal ROI found: %dx%d (%.1f%% of image)", 
                           cw, ch, (roi.shape[0]*roi.shape[1])/(h*w)*100)
                return roi

        logger.info("Universal ROI: No valid cluster found, using full image.")
        return image_np

    except Exception as e:
        logger.error("Universal ROI detection failed: %s — using full image", e)
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
        
        # Handle 90 degree rotations specifically
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if abs(angle) > 45:
            # If it's a massive angle, it might be exactly 90 degrees.
            # Let's trust the rotation matrix to handle it.
            pass
            
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
    Optimized for shiny plastic packets using CLAHE + Shadow Removal + Denoising.
    """
    try:
        # 1. Shadow Removal (Luminance Balancing)
        # Using a large morphological closing to estimate background illumination
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (33, 33))
        background = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        l = cv2.divide(l, background, scale=255)
        
        # 2. Refined CLAHE (Glare Reduction)
        # Lower clipLimit (2.0 instead of 3.0) prevents over-amplifying glare highlights
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 3. Bilateral Filter: Denoise while preserving critical text edges
        denoised = cv2.bilateralFilter(image_enhanced, 7, 50, 50)
        
        # 4. Sharpening (Unsharp Mask)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharp = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        
        return sharp
    except Exception as e:
        logger.warning("Hardened Enhancement failed: %s", e)
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


LABEL_FORMAT_RULES = {
    "indian_fssai": {
        "required_nutrients": ["energy", "protein", "carb", "fat"],
        "trust_signals": ["fssai", "per 100g", "per 100 g"],
    },
    "us_fda": {
        "required_nutrients": ["calorie", "fat", "sodium", "carb", "protein"],
        "trust_signals": ["nutrition facts", "daily value", "serving size", "% dv"],
    },
    "eu_format": {
        "required_nutrients": ["energy", "fat", "carb", "protein", "salt"],
        "trust_signals": ["reference intake", "typical values", "per 100g", "kj"],
    },
    "unknown": {
        "required_nutrients": ["energy", "fat", "protein"],
        "trust_signals": ["nutrition", "ingredient"],
    }
}


def detect_label_format(ocr_text: str) -> str:
    """Detect the regulatory format of this label."""
    t = ocr_text.lower()
    if "fssai" in t or "know your portion" in t or "per 100 g" in t:
        return "indian_fssai"
    if "nutrition facts" in t and "daily value" in t:
        return "us_fda"
    if "reference intake" in t or "typical values" in t:
        return "eu_format"
    return "unknown"


def validate_against_format(nutrients: list, format_key: str) -> dict:
    """Check if extracted nutrients match what this label format requires."""
    rules = LABEL_FORMAT_RULES.get(format_key, LABEL_FORMAT_RULES["unknown"])
    names = [n.get("name", "").lower() for n in nutrients]

    missing_required = [
        r for r in rules["required_nutrients"]
        if not any(r in nm for nm in names)
    ]

    return {
        "format": format_key,
        "missing_required": missing_required,
        "completeness": 1.0 - (len(missing_required) / max(1, len(rules["required_nutrients"]))),
    }
