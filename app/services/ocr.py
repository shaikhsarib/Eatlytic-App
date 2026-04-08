"""
app/services/ocr.py
EasyOCR wrapper with lazy loading, caching, and the Universal Label Filter.
"""

import os
import re
import logging
import hashlib
import threading
import numpy as np
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
    max_dim = 1600
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img_np = np.array(img)
    results = get_reader_for(lang_hint).readtext(img_np, detail=1)
    boxes = results

    # Reconstruct logical lines by grouping boxes by their vertical (Y) position.
    # Each result: (box_coords, text, confidence)
    # box_coords is a list of 4 [x,y] corners. We use the average Y of the box.
    line_groups: dict[int, list[tuple[str, float]]] = {}
    for box, text, conf in boxes:
        if not text.strip():
            continue
        avg_y = sum(pt[1] for pt in box) / 4.0
        # Bucket Y into ~10px bands to tolerate small alignment differences
        band = int(round(avg_y / 10))
        line_groups.setdefault(band, []).append((text, conf))

    # Sort bands top-to-bottom, then join words within each band
    sorted_lines = []
    for band in sorted(line_groups.keys()):
        words_in_line = " ".join(t for t, _ in line_groups[band])
        sorted_lines.append(words_in_line)

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
    THE TRASH COMPACTOR (v3 — context-aware):
    Strips out Indian legal text (FSSAI, MRP) and keeps only nutrition data.
    KEYWORD PRIORITY: a line containing a nutrition keyword is NEVER skipped,
    even if it also contains a garbage word (e.g. "Calories — MRP inclusive").
    HEADER PRESERVATION: first 3-5 non-garbage lines are kept (product name lives here).
    CONTEXT PRESERVATION: "Ingredients" / "Information" lines are kept for AI context.
    """
    lines = raw_ocr_text.split("\n")
    clean_lines = []
    number_count = 0

    nutrition_words = r"(energy|calories|protein|protien|fat|carb|sugar|sodium|fibre|fiber|salt|per\s*100\s*g|per\s*serve|serving|trans\s*fat|saturated|cholesterol|polyunsaturated|monounsaturated|total\s*fat|total\s*carb|dietary\s*fibre|dietary\s*fiber|added\s*sugar|vitamin|calcium|iron|potassium|moisture|ash|total\s*sugar)"
    garbage_words = r"(fssai|lic\.?|net\s*wt|net\s*qty|mrp|customer\s*care|batch\s*no|best\s*before|mfd|date\s*of|packed\s*on|manufactured\s*by|marketed\s*by|imported\s*by|country\s*of|origin|www\.|toll\s*free|phone|tel\.|e-mail|email|ingredient\s*you\s*know|storage\s*instructions|keep\s*in\s*a|store\s*in\s*a|cool\s*and\s*dry|place\s*away|direct\s*heat|sunlight|consumption|vegetarian|non.?vegetarian|green\s*dot|red\s*dot|fpo|license\s*no)"

    unit_pattern = (
        r"\b\d+(\.\d+)?\s*(g|mg|kcal|kj|kjoules|kilojoule|kilojoules|gm|gms|ml|iu|%)\b"
    )

    # HEADER PRESERVATION: keep first N non-garbage lines unconditionally
    # but still reject lines that are pure garbage (FSSAI, MRP, etc.)
    # Also require minimum text quality (at least 2 coherent words of 2+ chars)
    header_line_count = 0
    max_header_lines = 5

    for idx, line in enumerate(lines):
        line_l = line.lower().strip()
        if not line_l:
            continue

        has_nutrition_keyword = bool(re.search(nutrition_words, line_l))
        has_garbage = bool(re.search(garbage_words, line_l))
        has_number_unit = bool(re.search(unit_pattern, line_l))

        # Quality check: at least 2 coherent words (2+ chars each) AND
        # at least one substantial word (4+ chars) to filter out garbled text
        coherent_words = [w for w in line_l.split() if len(w) >= 2]
        substantial_words = [w for w in line_l.split() if len(w) >= 4]
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
        # so the AI understands what section it's reading
        if re.search(r"(ingredient|information|nutritional|nutrition info)", line_l):
            clean_lines.append(line)
            continue

        # Skip pure garbage lines (no nutrition keyword present)
        if has_garbage:
            continue

        # METRIC PRIORITY: any line with a number followed by a unit is KEPT
        # regardless of context (e.g. "384 kcal", "9.2g", "500mg")
        if has_number_unit:
            clean_lines.append(line)
            number_count += 1
            continue

        # RELAXED NUMBER GATE: allow more words in lines containing numbers
        # to capture long rows like "Energy (kcal) per 100g 384 288"
        if re.search(r"\b\d+(\.\d+)?\b", line_l) and len(line_l.split()) <= 10:
            clean_lines.append(line)
            number_count += 1
            continue

    clean_text = "\n".join(clean_lines)

    # If we find AT LEAST 1 metric number, it's a valid label.
    is_valid = number_count >= 1

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
