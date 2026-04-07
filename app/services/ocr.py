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

    confidences = [max(r[2], 0.05) if r[1].strip() else 0.0 for r in boxes]
    words = [r[1] for r in boxes]
    text = " ".join(w for w in words if w.strip())
    word_count = len(text.split())
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "text": text,
        "word_count": word_count,
        "avg_confidence": round(avg_conf, 3),
        "is_readable": int(word_count >= 3 and avg_conf > 0.10),
    }
    set_ocr_cache(cache_key, result)
    return result


def universal_label_filter(raw_ocr_text: str) -> dict:
    """
    THE TRASH COMPACTOR:
    Strips out Indian legal text (FSSAI, MRP) and keeps only nutrition data.
    Threshold is 1 number so single-ingredient items (Tata Salt) don't fail.
    """
    lines = raw_ocr_text.split('\n')
    clean_lines = []
    number_count = 0
    
    nutrition_words = r'(energy|protein|fat|carb|sugar|sodium|fibre|fiber|salt|per 100g|per serve)'
    garbage_words = r'(fssai|lic\.?|net wt|net qty|mrp|customer care|batch|best before|mfd|ingredient you know)'

    for line in lines:
        line_l = line.lower().strip()
        if not line_l: continue

        if re.search(garbage_words, line_l): 
            continue

        if re.search(nutrition_words, line_l):
            clean_lines.append(line)
            
        if re.search(r'\b\d+(\.\d+)?\s*(g|mg|kcal|kj)\b', line_l):
            number_count += 1

    clean_text = "\n".join(clean_lines)
    
    # If we find AT LEAST 1 metric number (e.g., "39,100mg"), it's a valid label.
    is_valid = number_count >= 1

    return {"is_valid": is_valid, "clean_text": clean_text}


def strip_marketing_fluff(raw_ocr_text: str) -> str:
    """Wrapper to easily call the filter from the LLM flow."""
    result = universal_label_filter(raw_ocr_text)
    return result["clean_text"]


OCR_CONFIDENCE_THRESHOLD = 0.30

def passes_confidence_gate(ocr_result: dict) -> tuple[bool, str]:
    """Check if OCR confidence meets minimum threshold."""
    avg_conf = ocr_result.get("avg_confidence", 0.0)
    word_count = ocr_result.get("word_count", 0)
    if avg_conf < OCR_CONFIDENCE_THRESHOLD and word_count < 20:
        return False, (
            f"⚠️ Image quality too low (confidence: {avg_conf:.0%}). "
            "Please take a closer photo of the label."
        )
    return True, ""
