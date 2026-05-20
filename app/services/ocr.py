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

DATA_DIR = os.path.join(os.getcwd(), "data")
CACHE_DIR = os.environ.get("HF_HOME", os.path.join(os.getcwd(), ".cache"))
MODEL_DIR = os.path.join(CACHE_DIR, "easyocr_models")

_LANG_READERS: dict = {}
_READERS_LOCK = threading.Lock()
INDIAN_LANG_SET = ["en", "hi", "ta"]
_EASYOCR_LANG_MAP = {
    "auto": ["en"], # will be detected
    "en": ["en"], "hi": ["en","hi"], "ta": ["en","ta"],
    "zh": ["ch_sim","en"], "ja": ["ja","en"], "ko": ["ko","en"],
    "ar": ["ar","en"], "th": ["th","en"], "ru": ["ru","en"],
    "de": ["de","en"], "fr": ["fr","en"], "es": ["es","en"],
    "pt": ["pt","en"], "it": ["it","en"], "bn": ["bn","en"],
    "te": ["te","en"], "mr": ["mr","en"], "gu": ["gu", "en"], "pa": ["pa", "en"]
}


def detect_language_from_text(text: str) -> str:
    """Detect script/language from already-extracted text. Zero extra OCR cost."""
    if re.search(r'[\u4e00-\u9fff]', text): return "zh"   # CJK
    if re.search(r'[\u3040-\u30ff]', text): return "ja"   # Hiragana/Katakana
    if re.search(r'[\uac00-\ud7af]', text): return "ko"   # Hangul
    if re.search(r'[\u0600-\u06ff]', text): return "ar"   # Arabic
    if re.search(r'[\u0900-\u097f]', text): return "hi"   # Devanagari (Hindi)
    if re.search(r'[\u0b80-\u0bff]', text): return "ta"   # Tamil
    if re.search(r'[\u0c00-\u0c7f]', text): return "te"   # Telugu
    if re.search(r'[\u0a80-\u0aff]', text): return "gu"   # Gujarati
    if re.search(r'[\u0a00-\u0a7f]', text): return "pa"   # Punjabi (Gurmukhi)
    if re.search(r'[\u0980-\u09ff]', text): return "bn"   # Bengali
    if re.search(r'[\u0400-\u04ff]', text): return "ru"   # Cyrillic
    if re.search(r'[\u0e00-\u0e7f]', text): return "th"   # Thai
    return "en"


def detect_language_from_image(content: bytes) -> str:
    """BUG FIX: Run a SINGLE cheap first-pass OCR with English-only reader, 
    then detect script from the extracted text — no second full OCR pass."""
    try:
        img: Image.Image = Image.open(BytesIO(content)).convert("RGB")
        w, h = img.size
        crop: Image.Image = img.crop((w//4, h//4, 3*w//4, 3*h//4))
        reader = get_reader_for("en")
        results: list[str] = reader.readtext(np.array(crop), detail=0)[:10]
        text: str = " ".join(results)
        return detect_language_from_text(text)
    except Exception as e:
        logger.debug("Script detection fallback to 'en': %s", e)
        return "en"

def get_reader_for(lang_hint: str):
    langs: list[str] = _EASYOCR_LANG_MAP.get(lang_hint, ["en"])
    key: str = "_".join(sorted(langs))
    if key not in _LANG_READERS:
        with _READERS_LOCK:
            if key not in _LANG_READERS:
                import easyocr as _easyocr

                logger.info("Loading EasyOCR for langs=%s", langs)
                _LANG_READERS[key] = _easyocr.Reader(
                    langs, gpu=False, model_storage_directory=MODEL_DIR
                )
    return _LANG_READERS[key]


def run_ocr(content: bytes, lang_hint: str = "auto") -> dict:
    """Universal OCR with auto language detection and dual-pass enhancement."""
    if lang_hint == "auto":
        lang_hint = detect_language_from_image(content)

    cache_key: str = f"{hashlib.md5(content).hexdigest()}_{lang_hint}"
    cached: dict = get_ocr_cache(cache_key)
    if cached:
        return cached

    img: Image.Image = Image.open(BytesIO(content)).convert("RGB")
    w, h = img.size
    
    # Universal upscale (capped at 3x to prevent memory OOM)
    if max(w, h) < 1000:
        scale_factor: float = min(3.0, 1200 / max(w, h))
        img = img.resize((int(w * scale_factor), int(h * scale_factor)), Image.LANCZOS)
    elif max(w, h) > 2200:
        ratio: float = 2200 / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    img_np: np.ndarray = np.array(img)
    reader = get_reader_for(lang_hint)
    
    # Pass 1: Standard Enhancement (CLAHE)
    import cv2
    img_cv: np.ndarray = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray: np.ndarray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced: np.ndarray = clahe.apply(gray)
    results: list = reader.readtext(enhanced, detail=1)
    
    # ── Multi-Pass Retry Strategy for Low Confidence (v5) ──
    def _get_avg_conf(res: list) -> float:
        if not res: return 0.0
        return sum(r[2] for r in res) / len(res)

    initial_conf: float = _get_avg_conf(results)
    
    if initial_conf < 0.4 or len(results) < 8:
        logger.info("Low OCR confidence (%.2f) — starting multi-pass retry...", initial_conf)
        
        passes: list[tuple[str, np.ndarray]] = [
            ("denoise", cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)),
            ("sharpen", cv2.filter2D(enhanced, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))),
            ("binary", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
        ]
        
        best_results: list = results
        best_conf: float = initial_conf
        
        for name, processed in passes:
            try:
                temp_results: list = reader.readtext(processed, detail=1)
                temp_conf: float = _get_avg_conf(temp_results)
                if temp_conf > best_conf and len(temp_results) >= len(best_results):
                    best_conf = temp_conf
                    best_results = temp_results
                    logger.info("Pass '%s' improved confidence to %.2f", name, best_conf)
            except Exception as e:
                logger.warning("OCR Pass %s failed: %s", name, e)
        
        results = best_results

    boxes: list = results

    # Reconstruct logical lines by grouping boxes by their vertical (Y) position.
    img_h, img_w = img_np.shape[:2]
    band_px: int = max(8, min(22, int(img_h * 0.015)))

    line_groups: dict[int, list[tuple[float, str, float]]] = {}
    for box, text, conf in boxes:
        if not text.strip():
            continue
        avg_x: float = sum(pt[0] for pt in box) / 4.0
        avg_y: float = sum(pt[1] for pt in box) / 4.0
        avg_h: float = abs(box[2][1] - box[0][1])   # box height in pixels
        effective_band: int = max(band_px, int(avg_h * 0.6))
        band: int = int(round(avg_y / effective_band))
        line_groups.setdefault(band, []).append((avg_x, text, conf))

    sorted_lines: list[str] = []
    for band in sorted(line_groups.keys()):
        words_sorted: list[tuple[float, str, float]] = sorted(line_groups[band], key=lambda x: x[0])
        col_gap_threshold: int = max(60, int(img_w * 0.05))
        line_text: str = ""
        last_x: float = -1
        for x, text, _ in words_sorted:
            if last_x == -1:
                line_text = text
            else:
                gap: float = x - last_x
                sep: str = "\t" if gap > col_gap_threshold else " "
                line_text += f"{sep}{text}"
            last_x = x + (len(text) * 7)

        sorted_lines.append(line_text)

    all_text: str = "\n".join(sorted_lines)
    flat_text: str = " ".join(sorted_lines)

    confidences: list[float] = [float(max(r[2], 0.05)) if r[1].strip() else 0.0 for r in boxes]
    word_count: int = len(flat_text.split())
    avg_conf: float = sum(confidences) / len(confidences) if confidences else 0.0

    result: dict = {
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
    """
    lines: list[str] = raw_ocr_text.split("\n")
    clean_lines: list[str] = []
    number_count: int = 0

    nutrition_words: str = (
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
    garbage_words: str = (
        r"(fssai|lic\.?|net\s*wt|net\s*qty|mrp|customer\s*care|batch\s*no|"
        r"best\s*before|mfd|date\s*of|packed\s*on|manufactured\s*by|marketed\s*by|"
        r"imported\s*by|country\s*of|origin|www\.|toll\s*free|tel\.|e-mail|"
        r"storage\s*instructions|keep\s*in\s*a|store\s*in\s*a|cool\s*and\s*dry|"
        r"place\s*away|direct\s*heat|sunlight|green\s*dot|red\s*dot|fpo|license\s*no)"
    )

    unit_pattern: str = (
        r"\b\d+([\.,]\d+)?\s*(g|mg|kcal|kj|cal|gm|gms|ml|iu|mcg|µg|%|千卡|克|毫克|エネルギー|蛋白质|脂肪)(\b|/)"
    )

    header_line_count: int = 0
    max_header_lines: int = 5

    for idx, line in enumerate(lines):
        line_l: str = line.lower().strip()
        if not line_l:
            continue

        has_nutrition_keyword: bool = bool(re.search(nutrition_words, line_l))
        has_garbage: bool = bool(re.search(garbage_words, line_l))
        has_number_unit: bool = bool(re.search(unit_pattern, line_l))

        coherent_words: list[str] = [w for w in line_l.split() if len(w) >= 2]
        substantial_words: list[str] = [w for w in line_l.split() if len(w) >= 4]
        
        has_min_quality: bool = len(coherent_words) >= 1
        if not has_nutrition_keyword:
             has_min_quality = len(coherent_words) >= 2 and len(substantial_words) >= 1

        if header_line_count < max_header_lines and not has_garbage and has_min_quality:
            header_line_count += 1
            clean_lines.append(line)
            if has_number_unit or re.search(r"\b\d+(\.\d+)?\b", line_l):
                number_count += 1
            continue

        if has_nutrition_keyword:
            clean_lines.append(line)
            if has_number_unit or re.search(r"\b\d+(\.\d+)?\b", line_l):
                number_count += 1
            continue

        if re.search(r"(ingredient|information|nutritional|nutrition info|serving)", line_l):
            clean_lines.append(line)
            if re.search(r"\b\d+(\.\d+)?\b", line_l):
                number_count += 1
            continue

        if has_garbage:
            continue

        if has_number_unit:
            clean_lines.append(line)
            number_count += 1
            continue

        if re.search(r"\b\d+(\.\d+)?\b", line_l) and len(line_l.split()) <= 15:
            clean_lines.append(line)
            number_count += 1
            continue

    clean_text: str = "\n".join(clean_lines)

    high_strength_header: bool = bool(re.search(
        r"(nutrition\s*facts|amount\s*per\s*serving|information\s*per|serving\s*size|"
        r"calories\s+from|daily\s+value|percent\s+daily|per\s+100\s*g|per\s+100\s*ml|"
        r"total\s+fat|total\s+carb|dietary\s+fiber|dietary\s+fibre|"
        r"saturated\s+fat|trans\s+fat|reference\s+intake|typical\s+values|"
        r"nutritional\s+information|nutritional\s+value|nutrient\s+content|"
        r"per\s+portion|per\s+serving|kj\s+\d|kcal\s+\d|\d+\s*kcal|\d+\s*kj|energy\s*\d|protein\s*\d)",
        raw_ocr_text.lower()
    ))
    is_valid: bool = (number_count >= 4) and high_strength_header

    return {"is_valid": is_valid, "clean_text": clean_text}


def strip_marketing_fluff(raw_ocr_text: str) -> str:
    """Wrapper to easily call the filter from the LLM flow."""
    result: dict = universal_label_filter(raw_ocr_text)
    return result["clean_text"]


OCR_CONFIDENCE_THRESHOLD: float = 0.35


def passes_confidence_gate(ocr_result: dict) -> tuple[bool, str]:
    """Check if OCR confidence meets minimum threshold."""
    avg_conf: float = float(ocr_result.get("avg_confidence", 0.0))
    word_count: int = int(ocr_result.get("word_count", 0))
    if avg_conf < OCR_CONFIDENCE_THRESHOLD and word_count < 15:
        return False, (
            f"Image quality too low (confidence: {avg_conf:.0%}). "
            "Please take a closer photo of the label."
        )
    return True, ""
