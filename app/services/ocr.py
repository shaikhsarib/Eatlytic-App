"""
app/services/ocr.py
EasyOCR wrapper with lazy loading, caching, and label-presence detection.
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
# Robust cache path: use local project dir on Windows, /app/.cache in Docker
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
    # Optimize: 1600px is the sweet spot for dense tables
    w, h = img.size
    max_dim = 1600
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img_np = np.array(img)
    results = get_reader_for(lang_hint).readtext(img_np, detail=1)
    boxes = results  # each entry: (bbox, text, confidence)

    # EasyOCR quirk: some boxes return 0.0 even when text is clearly read.
    # Clamp only boxes with non-empty text to 0.05 (not 0.1) so avg_confidence
    # is not poisoned by a single bad box, but still reflects poor quality.
    confidences = [max(r[2], 0.05) if r[1].strip() else 0.0 for r in boxes]
    words = [r[1] for r in boxes]
    text = " ".join(w for w in words if w.strip())
    word_count = len(text.split())  # count actual words from the joined text
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "text": text,
        "word_count": word_count,
        "avg_confidence": round(avg_conf, 3),
        "is_readable": word_count >= 3 and avg_conf > 0.10,
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
    "potassium",
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
    "colour",
    "color",
    "emulsifier",
    "stabilizer",
    "antioxidant",
    "wheat",
    "milk",
    "soy",
    "salt",
    "water",
    "oil",
    "starch",
    "extract",
    "atta",
    "maida",
    "lecithin",
    "maltodextrin",
    "rising",
    "mg",
    "mcg",
    "kcal",
    "kj",
    "% dv",
    "%dv",
    "g per",
    "per serving",
    "fssai",
    "veg",
    "non-veg",
    "best before",
    "mfg",
    "mrp",
    "net wt",
    "manufactured",
    "packed",
    "distributed",
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
    "flavor",
    "crisps",
    "chips",
    "potato",
    "style",
    "authentic",
    "traditional",
    "indulge",
    "real",
    "nature",
    "best",
    "quality",
    "value",
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
    "protein",
    "carbohydrate",
    "fat",
    "sugars",
    "nutrition facts",
    "nutritional information",
    "nutrition information",
    "total fat",
    "saturated fat",
    "trans fat",
    "total carbohydrate",
    "dietary fiber",
    "total sugars",
    "ingredients:",
    "ingredients list",
    "fssai",
    "best before",
    "mfg",
    "mrp",
    "net wt",
    "nutrichoice",
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


# Gate rejects if confidence is too low.
# Tuning: 30% is much safer for noisy labels, especially with word-count bypass
OCR_CONFIDENCE_THRESHOLD = 0.30


def passes_confidence_gate(ocr_result: dict) -> tuple[bool, str]:
    """Check if OCR confidence meets minimum threshold. Returns (pass, message)."""
    avg_conf = ocr_result.get("avg_confidence", 0.0)
    word_count = ocr_result.get("word_count", 0)
    if avg_conf < OCR_CONFIDENCE_THRESHOLD and word_count < 20:
        return False, (
            f"⚠️ Image quality too low (confidence: {avg_conf:.0%}). "
            "Please take a closer photo of the label."
        )
    return True, ""
