"""
app/services/ocr.py
EasyOCR wrapper with lazy loading, caching, and label-presence detection.
"""

import re
import logging
import hashlib
import threading
import numpy as np
from PIL import Image
from io import BytesIO
from app.models.db import get_ocr_cache, set_ocr_cache

logger = logging.getLogger(__name__)
DATA_DIR = __import__("os").path.join(__import__("os").getcwd(), "data")
CACHE_DIR = __import__("os").environ.get("HF_HOME", "/app/.cache")
MODEL_DIR = __import__("os").path.join(CACHE_DIR, "easyocr_models")

_LANG_READERS: dict = {}
_READERS_LOCK = threading.Lock()
_EASYOCR_LANG_MAP = {
    "en": ["en"],
    "hi": ["en", "hi"],
    "zh": ["en", "ch_sim"],
    "ta": ["en", "ta"],
    "te": ["en", "te"],
    "bn": ["en", "bn"],
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
    img.thumbnail((1200, 1200))
    img_np = np.array(img)
    results = get_reader_for(lang_hint).readtext(img_np, detail=1)
    words = [r[1] for r in results]
    confidences = [r[2] for r in results]
    text = " ".join(words)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "text": text,
        "word_count": len(words),
        "avg_confidence": round(avg_conf, 3),
        "is_readable": len(words) >= 3 and avg_conf > 0.15,
    }
    set_ocr_cache(cache_key, result)
    return result


LABEL_KEYWORDS = [
    "ingredients",
    "nutrition",
    "nutritional",
    "calories",
    "calorie",
    "protein",
    "fat",
    "carbohydrate",
    "carbs",
    "sodium",
    "sugar",
    "sugars",
    "fiber",
    "fibre",
    "serving",
    "cholesterol",
    "saturated",
    "trans",
    "vitamin",
    "calcium",
    "iron",
    "per 100g",
    "per 100 g",
    "daily value",
    "daily values",
    "amount per",
    "total fat",
    "contains",
    "may contain",
    "preservative",
    "flavour",
    "flavor",
    "emulsifier",
    "mg",
    "mcg",
    "kcal",
    "kj",
    "% dv",
    "%dv",
    "g per",
    "per serving",
    "fssai",
    "best before",
    "mfg",
    "mrp",
    "net wt",
    "manufactured",
    "packed",
]
FRONT_PACK_SIGNALS = [
    "new",
    "improved",
    "original",
    "classic",
    "natural",
    "organic",
    "premium",
    "delicious",
    "flavoured",
    "variety",
    "crunchy",
    "crispy",
    "fresh",
    "tasty",
    "yummy",
    "light",
    "baked",
    "roasted",
]
NUTRITION_TABLE_ANCHORS = [
    "per 100g",
    "per 100 g",
    "per serving",
    "serving size",
    "amount per",
    "daily value",
    "daily values",
    "% dv",
    "%dv",
    "calories",
    "calorie",
    "kcal",
    "kj",
    "energy",
    "nutrition facts",
    "nutritional information",
    "total fat",
    "saturated fat",
    "trans fat",
    "total carbohydrate",
    "dietary fiber",
    "ingredients:",
    "fssai",
    "best before",
    "mfg",
    "mrp",
    "net wt",
]


def detect_label_presence(ocr_text: str) -> dict:
    if not ocr_text:
        return {
            "has_label": False,
            "confidence": "high",
            "label_hits": [],
            "front_hits": [],
            "suggestion": "no_text",
        }
    tl = ocr_text.lower()
    label_hits = [kw for kw in LABEL_KEYWORDS if kw in tl]
    front_hits = [kw for kw in FRONT_PACK_SIGNALS if kw in tl]
    anchor_hits = [kw for kw in NUTRITION_TABLE_ANCHORS if kw in tl]
    ls, fs = len(label_hits), len(front_hits)
    has_table = len(anchor_hits) >= 2

    if has_table and ls >= 3:
        return {
            "has_label": True,
            "confidence": "high" if ls >= 6 else "medium",
            "label_hits": label_hits[:5],
            "front_hits": front_hits[:3],
            "suggestion": None,
        }
    if has_table and ls >= 1 and fs <= 2:
        return {
            "has_label": True,
            "confidence": "low",
            "label_hits": label_hits,
            "front_hits": front_hits,
            "suggestion": None,
        }
    if fs > ls or not has_table:
        sug = "wrong_side" if fs > 0 else "no_label"
        return {
            "has_label": False,
            "confidence": "high",
            "label_hits": label_hits,
            "front_hits": front_hits[:3],
            "suggestion": sug,
        }
    return {
        "has_label": True,
        "confidence": "low",
        "label_hits": label_hits,
        "front_hits": front_hits,
        "suggestion": "partial",
    }


OCR_CONFIDENCE_THRESHOLD = 0.70


def passes_confidence_gate(ocr_result: dict) -> tuple[bool, str]:
    """Check if OCR confidence meets minimum threshold. Returns (pass, message)."""
    avg_conf = ocr_result.get("avg_confidence", 0.0)
    if avg_conf < OCR_CONFIDENCE_THRESHOLD:
        return False, (
            f"\u26a0\ufe0f Image too blurry (confidence: {avg_conf:.0%}). "
            "Please retake the photo in better lighting."
        )
    return True, ""
