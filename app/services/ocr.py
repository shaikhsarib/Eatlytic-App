"""
app/services/ocr.py
EasyOCR wrapper with lazy loading, caching, and the Universal Label Filter.
"""

import os
import re
import sys
import logging
import hashlib
import threading
import warnings
import numpy as np

# Suppress noisy PyTorch pin_memory warning on CPU-only machines
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

logger = logging.getLogger(__name__)

from PIL import Image
from io import BytesIO
from app.models.db import get_ocr_cache, set_ocr_cache

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.getcwd(), "data")
CACHE_DIR = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".cache"))
MODEL_DIR = os.path.join(CACHE_DIR, "easyocr_models")

_LANG_READERS: dict = {}
_READERS_LOCK = threading.Lock()
INDIAN_LANG_SET = ["en", "hi", "ta"]
_EASYOCR_LANG_MAP = {
    "en": ["en"],
    "hi": INDIAN_LANG_SET,
    "ta": INDIAN_LANG_SET,
    "te": ["en", "te"],
    "bn": ["en", "bn"],
    "mr": ["en", "mr"],
    "zh": ["en", "ch_sim"],
}


def get_reader_for(lang_hint: str):
    langs = _EASYOCR_LANG_MAP.get(lang_hint, ["en"])
    key = "_".join(sorted(langs))
    if key not in _LANG_READERS:
        with _READERS_LOCK:
            if key not in _LANG_READERS:
                import easyocr as _easyocr

                logger.info("Loading EasyOCR for langs=%s", langs)
                _LANG_READERS[key] = _easyocr.Reader(
                    langs, gpu=False, model_storage_directory=MODEL_DIR
                )
    return _LANG_READERS[key]


def run_ocr(content: bytes, lang_hint: str = "en") -> dict:
    """Extract text from image bytes. Returns text, word_count, avg_confidence."""
    cache_key = f"{hashlib.md5(content).hexdigest()}_{lang_hint}"
    cached = get_ocr_cache(cache_key)
    if cached:
        return cached

    img = Image.open(BytesIO(content)).convert("RGB")
    w, h = img.size
    
    # NEW: AI Super-Resolution Fallback for small thumbnails
    # If the image is < 800px, upscale it significantly to help OCR detect tiny characters
    if max(w, h) < 800:
        scale_factor = 2000 / max(w, h)
        img = img.resize((int(w * scale_factor), int(h * scale_factor)), Image.LANCZOS)
        
        # Enhanced Computer Vision Pass (Deep-Scan V3)
        import cv2
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. Denoise: Remove compression artifacts (Salt & Pepper noise)
        denoised = cv2.medianBlur(gray, 3)
        
        # 2. Threshold: Otsu's Bitonal conversion (Perfect B&W for OCR)
        # This separates the text cleanly from colorful backgrounds
        _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Dilate/Erode (Morphology) to connect broken letters
        kernel = np.ones((1, 1), np.uint8)
        img_np = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        
    elif max(w, h) > 2200:
        # Downscale massive 4K photos to prevent memory OOM
        ratio = 2200 / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        img_np = np.array(img)
    else:
        img_np = np.array(img)

    results = get_reader_for(lang_hint).readtext(img_np, detail=1)
    boxes = results

    # Reconstruct logical lines by grouping boxes by their vertical (Y) position.
    # Each result: (box_coords, text, confidence)
    # box_coords is a list of 4 [x,y] corners. We use the average Y of the box.
    # Dynamic band size: ~1.5% of image height (scales with image resolution)
    # Minimum 8px, maximum 22px — handles both tiny crops and full-res photos
    img_h, img_w = img_np.shape[:2]
    band_px = max(8, min(22, int(img_h * 0.015)))

    line_groups: dict[int, list[tuple[float, str, float]]] = {}
    for box, text, conf in boxes:
        if not text.strip():
            continue
        avg_x = sum(pt[0] for pt in box) / 4.0
        avg_y = sum(pt[1] for pt in box) / 4.0
        avg_h = abs(box[2][1] - box[0][1])   # box height in pixels
        # Use the larger of the dynamic band or half the text box height
        # so tall characters don't bleed into the wrong row
        effective_band = max(band_px, int(avg_h * 0.6))
        band = int(round(avg_y / effective_band))
        line_groups.setdefault(band, []).append((avg_x, text, conf))

    # Sort bands top-to-bottom, then words within each band left-to-right
    sorted_lines = []
    for band in sorted(line_groups.keys()):
        words_sorted = sorted(line_groups[band], key=lambda x: x[0])

        # Build line text. Use TAB as column separator only for wide gaps
        # (actual column gap in a nutrition table is typically > 80px after resize)
        col_gap_threshold = max(60, int(img_w * 0.05))
        line_text = ""
        last_x = -1
        for x, text, _ in words_sorted:
            if last_x == -1:
                line_text = text
            else:
                gap = x - last_x
                sep = "\t" if gap > col_gap_threshold else " "
                line_text += f"{sep}{text}"
            last_x = x + (len(text) * 7)   # ~7px per char estimate

        sorted_lines.append(line_text)

    all_text = "\n".join(sorted_lines)
    flat_text = " ".join(sorted_lines)

    confidences = [max(r[2], 0.05) if r[1].strip() else 0.0 for r in boxes]
    word_count = len(flat_text.split())
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "text": all_text,
        "flat_text": flat_text,
        "word_count": word_count,
        "avg_confidence": round(avg_conf, 3),
        "is_readable": int(word_count >= 3 and avg_conf > 0.10),
    }
    set_ocr_cache(cache_key, result)
    return result


def universal_label_filter(raw_ocr_text: str) -> dict:
    """
    THE TRASH COMPACTOR (v4 — permissive, context-aware):
    Strips out Indian legal text (FSSAI, MRP) and keeps only nutrition data.

    KEY FIXES in v4:
    - Expanded nutrition_words to catch serving size, daily value, etc.
    - High-strength header check now catches ANY of: "Nutrition Facts",
      "Serving Size", "Total Fat", "Saturated Fat", "Daily Value",
      "Per 100g", "Total Carb", "Dietary Fiber".
    - This ensures stylized product mockups (peanut butter label mockups, etc.)
      are NEVER falsely rejected — if the image has nutrition keywords, it passes.
    """
    lines = raw_ocr_text.split("\n")
    clean_lines = []
    number_count = 0

    nutrition_words = (
        r"(energy|calorie|calories|protein|protien|fat|carb|sugar|sodium|fibre|fiber|"
        r"salt|per\s*100\s*g|per\s*100\s*ml|per\s*serve|per\s*serving|serving|"
        r"trans\s*fat|saturated|unsaturated|mono.unsaturated|poly.unsaturated|"
        r"cholesterol|polyunsaturated|monounsaturated|total\s*fat|total\s*carb|"
        r"dietary\s*fibre|dietary\s*fiber|added\s*sugar|vitamin|calcium|iron|"
        r"potassium|magnesium|moisture|ash|total\s*sugar|amount\s*per\s*serving|"
        r"nutrition\s*facts|nutritional\s*info|nutritional\s*value|dietary\s*info|"
        r"information\s*per|serving\s*size|daily\s*value|%\s*dv|percent\s*daily|"
        r"kj|kilojoule|kilojoules|kcal|kilo\s*calorie|oleic|starch|glycogen|"
        r"nutrient|nutrients|per\s*portion|reference\s*intake|ri|\%\s*ri)"
    )
    garbage_words = (
        r"(fssai|lic\.?|net\s*wt|net\s*qty|mrp|customer\s*care|batch\s*no|"
        r"best\s*before|mfd|date\s*of|packed\s*on|manufactured\s*by|marketed\s*by|"
        r"imported\s*by|country\s*of|origin|www\.|toll\s*free|tel\.|e-mail|"
        r"storage\s*instructions|keep\s*in\s*a|store\s*in\s*a|cool\s*and\s*dry|"
        r"place\s*away|direct\s*heat|sunlight|green\s*dot|red\s*dot|fpo|license\s*no)"
    )

    unit_pattern = (
        r"\b\d+(\.\d+)?\s*(g|mg|kcal|kj|kjoules|kilojoule|kilojoules|gm|gms|ml|iu|%)(\b|/)"
    )

    header_line_count = 0
    max_header_lines = 5

    for idx, line in enumerate(lines):
        line_l = line.lower().strip()
        if not line_l:
            continue

        has_nutrition_keyword = bool(re.search(nutrition_words, line_l))
        has_garbage = bool(re.search(garbage_words, line_l))
        has_number_unit = bool(re.search(unit_pattern, line_l))

        coherent_words = [w for w in line_l.split() if len(w) >= 2]
        substantial_words = [w for w in line_l.split() if len(w) >= 4]
        
        # RELAXED QUALITY: If we already found a nutrition keyword, lower the bar
        has_min_quality = len(coherent_words) >= 1
        if not has_nutrition_keyword:
             has_min_quality = len(coherent_words) >= 2 and len(substantial_words) >= 1

        # HEADER PRESERVATION: keep first N non-garbage lines with minimum quality
        if header_line_count < max_header_lines and not has_garbage and has_min_quality:
            header_line_count += 1
            clean_lines.append(line)
            if has_number_unit:
                number_count += 1
            if re.search(r"\b\d+(\.\d+)?\b", line_l):
                number_count += 1
            continue

        # KEYWORD PRIORITY: nutrition lines are never discarded
        if has_nutrition_keyword:
            clean_lines.append(line)
            if has_number_unit:
                number_count += 1
            # Also count bare numbers on nutrition lines (e.g. "Protein 9.2")
            if re.search(r"\b\d+(\.\d+)?\b", line_l):
                number_count += 1
            continue

        # CONTEXT PRESERVATION: keep "Ingredients" or "Information" lines
        if re.search(r"(ingredient|information|nutritional|nutrition info|serving)", line_l):
            clean_lines.append(line)
            if re.search(r"\b\d+(\.\d+)?\b", line_l):
                number_count += 1
            continue

        # Skip pure garbage lines (no nutrition keyword present)
        if has_garbage:
            continue

        # METRIC PRIORITY: any line with a number followed by a unit is KEPT
        if has_number_unit:
            clean_lines.append(line)
            number_count += 1
            continue

        # RELAXED NUMBER GATE: captures rows like "Energy (kcal) per 100g 384"
        if re.search(r"\b\d+(\.\d+)?\b", line_l) and len(line_l.split()) <= 15:
            clean_lines.append(line)
            number_count += 1
            continue

    clean_text = "\n".join(clean_lines)

    # FINAL GATE (v5): Valid if EITHER:
    #   A) At least 1 number-like line found in cleaned text, OR
    #   B) Any high-strength nutrition keyword/header found ANYWHERE in original text
    # This ensures any label with recognizable nutrition keywords is always passed to AI.
    high_strength_header = bool(re.search(
        r"(nutrition\s*facts|amount\s*per\s*serving|information\s*per|serving\s*size|"
        r"calories\s+from|daily\s+value|percent\s+daily|per\s+100\s*g|per\s+100\s*ml|"
        r"total\s+fat|total\s+carb|dietary\s+fiber|dietary\s+fibre|"
        r"saturated\s+fat|trans\s+fat|reference\s+intake|typical\s+values|"
        r"nutritional\s+information|nutritional\s+value|nutrient\s+content|"
        r"per\s+portion|per\s+serving|kj\s+\d|kcal\s+\d|\d+\s*kcal|\d+\s*kj)",
        raw_ocr_text.lower()
    ))
    is_valid = number_count >= 1 or high_strength_header

    return {"is_valid": is_valid, "clean_text": clean_text}


def strip_marketing_fluff(raw_ocr_text: str) -> str:
    """Wrapper to easily call the filter from the LLM flow."""
    result = universal_label_filter(raw_ocr_text)
    return result["clean_text"]


OCR_CONFIDENCE_THRESHOLD = 0.25


def passes_confidence_gate(ocr_result: dict) -> tuple[bool, str]:
    """Check if OCR confidence meets minimum threshold."""
    avg_conf = ocr_result.get("avg_confidence", 0.0)
    word_count = ocr_result.get("word_count", 0)
    if avg_conf < OCR_CONFIDENCE_THRESHOLD and word_count < 15:
        return False, (
            f"Image quality too low (confidence: {avg_conf:.0%}). "
            "Please take a closer photo of the label."
        )
    return True, ""
