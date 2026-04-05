import os
import io
import json
import logging
import hashlib
import base64
import secrets
import datetime
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from duckduckgo_search import DDGS
from groq import Groq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Eatlytic: Startup Scale")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- PERSISTENT STORAGE ---
DATA_DIR = os.path.join(os.getcwd(), "data")
CACHE_DIR = os.environ.get("HF_HOME", "/app/.cache")
MODEL_DIR = os.path.join(CACHE_DIR, "easyocr_models")

for d in [MODEL_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# --- CACHE SETUP ---
OCR_CACHE_FILE = os.path.join(DATA_DIR, "ocr_cache.json")
AI_CACHE_FILE = os.path.join(DATA_DIR, "ai_cache.json")


def load_cache(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache, file_path):
    try:
        with open(file_path, "w") as f:
            json.dump(cache, f)
    except IOError:
        pass


ocr_cache = load_cache(OCR_CACHE_FILE)
ai_cache = load_cache(AI_CACHE_FILE)

# --- SCAN LIMITS & API KEYS (Task 11 + 13) ---
SCAN_LIMIT_FILE = os.path.join(DATA_DIR, "scan_limits.json")
API_KEYS_FILE = os.path.join(DATA_DIR, "api_keys.json")
FREE_SCAN_LIMIT = 10


def load_scan_limits():
    if os.path.exists(SCAN_LIMIT_FILE):
        try:
            with open(SCAN_LIMIT_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_scan_limits(data):
    try:
        with open(SCAN_LIMIT_FILE, "w") as f:
            json.dump(data, f)
    except IOError:
        pass


def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_api_keys(data):
    try:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(data, f)
    except IOError:
        pass


scan_limits = load_scan_limits()
api_keys_db = load_api_keys()

# --- API KEY AUTH (Task 13) ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    global api_keys_db
    api_keys_db = load_api_keys()
    if not api_key:
        return None
    key_data = api_keys_db.get(api_key)
    if key_data and isinstance(key_data, dict):
        return dict(key_data)
    return None


def generate_api_key(client_name: str, plan: str = "business") -> str:
    global api_keys_db
    key = "eak_" + secrets.token_urlsafe(32)
    api_keys_db[key] = {
        "name": client_name,
        "plan": plan,
        "scans_this_month": 0,
        "month": "",
        "active": True,
    }
    save_api_keys(api_keys_db)
    return key


# --- DEVICE FINGERPRINT + SCAN GATE (Task 11) ---
def get_device_key(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")
    return hashlib.md5(f"{ip}:{ua}".encode()).hexdigest()[:16]


def check_and_increment_scan(device_key: str) -> dict:
    global scan_limits
    scan_limits = load_scan_limits()
    month_key = datetime.date.today().isoformat()[:7]
    if device_key not in scan_limits:
        scan_limits[device_key] = {}
    u = scan_limits[device_key]
    if u.get("month") != month_key:
        u["month"] = month_key
        u["count"] = 0
        u["pro"] = u.get("pro", False)
    if u.get("pro"):
        u["count"] += 1
        save_scan_limits(scan_limits)
        return {
            "allowed": True,
            "scans_used": u["count"],
            "scans_remaining": 9999,
            "is_pro": True,
        }
    if u["count"] >= FREE_SCAN_LIMIT:
        return {
            "allowed": False,
            "scans_used": u["count"],
            "scans_remaining": 0,
            "is_pro": False,
        }
    u["count"] += 1
    save_scan_limits(scan_limits)
    return {
        "allowed": True,
        "scans_used": u["count"],
        "scans_remaining": FREE_SCAN_LIMIT - u["count"],
        "is_pro": False,
    }


# --- CLIENTS ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY missing! App will fail.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

reader = easyocr.Reader(["en", "ch_sim"], gpu=False, model_storage_directory=MODEL_DIR)

# --- MULTI-LANGUAGE READER CACHE (Task 17) ---
# Readers are expensive to load; cache by language group
_LANG_READERS: dict = {}
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
        logger.info(f"Loading EasyOCR reader for langs: {langs}")
        _LANG_READERS[key] = easyocr.Reader(
            langs, gpu=False, model_storage_directory=MODEL_DIR
        )
    return _LANG_READERS[key]


#  SECTION 1: MULTI-METHOD BLUR DETECTION
# ══════════════════════════════════════════════════════════════════════


def _laplacian_score(gray: np.ndarray) -> float:
    """
    Laplacian variance — high sensitivity to edges.
    Scores below ~100 typically indicate blur.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad_score(gray: np.ndarray) -> float:
    """
    Tenengrad — sum of squared Sobel gradient magnitudes.
    Very robust for detecting out-of-focus / motion blur.
    Normalised to image pixel count for size-independence.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = gx**2 + gy**2
    return float(np.mean(mag))


def _brenner_score(gray: np.ndarray) -> float:
    """
    Brenner gradient — fast and sensitive to fine text edges.
    Computed as mean squared difference between pixels 2 apart.
    """
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float(np.mean(diff**2))


def _local_blur_map(gray: np.ndarray, block: int = 64) -> float:
    h, w = gray.shape
    scores = []
    for y in range(0, max(h - block + 1, 1), block):
        for x in range(0, max(w - block + 1, 1), block):
            patch = gray[y : y + block, x : x + block]
            scores.append(cv2.Laplacian(patch, cv2.CV_64F).var())
    return float(np.median(scores)) if scores else 0.0


def assess_image_quality(content: bytes) -> dict:
    """
    Multi-method blur detection combining:
    • Laplacian variance (global)
    • Tenengrad (gradient energy)
    • Brenner gradient (text sensitivity)
    • Local block-median (spatial robustness)
    Returns a rich quality dict with per-method scores.
    """
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        lap = _laplacian_score(gray)
        ten = _tenengrad_score(gray)
        bren = _brenner_score(gray)
        local = _local_blur_map(gray)

        # Normalise scores to 0-100 for consistent comparison
        # Thresholds tuned against a food-label test set
        lap_norm = min(lap / 300.0 * 100, 100)
        ten_norm = min(ten / 500.0 * 100, 100)
        bren_norm = min(bren / 200.0 * 100, 100)
        local_norm = min(local / 300.0 * 100, 100)

        # Weighted composite: local_median carries the most weight
        composite = (
            0.25 * lap_norm + 0.20 * ten_norm + 0.20 * bren_norm + 0.35 * local_norm
        )

        # Blur severity bands
        if composite < 15:
            severity = "severe"
            is_blurry = True
        elif composite < 35:
            severity = "moderate"
            is_blurry = True
        elif composite < 55:
            severity = "mild"
            is_blurry = True  # still attempt enhancement
        else:
            severity = "none"
            is_blurry = False

        quality = "poor" if composite < 35 else ("fair" if composite < 55 else "good")

        return {
            "blur_score": round(composite, 2),
            "laplacian_score": round(lap, 2),
            "tenengrad_score": round(ten, 2),
            "brenner_score": round(bren, 2),
            "local_median_score": round(local, 2),
            "is_blurry": is_blurry,
            "blur_severity": severity,
            "quality": quality,
        }
    except Exception as e:
        logger.error(f"Blur detection error: {e}")
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


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2: DEBLURRING & IMAGE ENHANCEMENT PIPELINE
# ══════════════════════════════════════════════════════════════════════


def _wiener_deconvolution(
    gray: np.ndarray, psf_size: int = 5, noise_ratio: float = 0.02
) -> np.ndarray:
    """
    Blind Wiener deconvolution using an estimated Gaussian PSF.
    Works in the frequency domain:
        restored = (H* / (|H|^2 + K)) * Y
    where H = FFT of the PSF, Y = FFT of the blurred image, K = noise ratio.
    Effective for Gaussian and mild motion blur.
    """
    # Clamp PSF size to valid odd numbers
    psf_size = max(3, psf_size | 1)

    # Build Gaussian PSF
    psf = cv2.getGaussianKernel(psf_size, psf_size / 3.0)
    psf = psf @ psf.T
    psf /= psf.sum()

    h, w = gray.shape
    psf_padded = np.zeros_like(gray, dtype=np.float64)
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf

    # Roll to centre the PSF
    psf_padded = np.roll(psf_padded, -ph // 2, axis=0)
    psf_padded = np.roll(psf_padded, -pw // 2, axis=1)

    # Frequency-domain Wiener filter
    Y = np.fft.fft2(gray.astype(np.float64) / 255.0)
    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)
    W = H_conj / (np.abs(H) ** 2 + noise_ratio)
    restored = np.real(np.fft.ifft2(W * Y))

    # Normalise to uint8
    restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    return restored


def _unsharp_mask(
    img_np: np.ndarray, strength: float = 1.5, radius: int = 3
) -> np.ndarray:
    """
    Unsharp masking:  sharpened = original + strength * (original − blurred)
    Works on colour images; more robust and artefact-free than Wiener for
    already near-sharp images.
    """
    blurred = cv2.GaussianBlur(img_np, (radius * 2 + 1, radius * 2 + 1), 0)
    mask = cv2.subtract(img_np.astype(np.int16), blurred.astype(np.int16))
    sharp = np.clip(img_np.astype(np.float32) + strength * mask, 0, 255)
    return sharp.astype(np.uint8)


def _apply_clahe(img_np: np.ndarray, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast-Limited Adaptive Histogram Equalisation) applied to the
    L-channel of LAB colour space.  Preserves hue / saturation while
    dramatically improving local contrast in dim or washed-out images.
    """
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _denoise(img_np: np.ndarray, h: int = 6) -> np.ndarray:
    """
    Non-local means denoising. Removes sensor/JPEG noise that unsharp masking
    would otherwise amplify, giving cleaner text edges post-sharpening.
    """
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    bgr_denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
    return cv2.cvtColor(bgr_denoised, cv2.COLOR_BGR2RGB)


def deblur_and_enhance(content: bytes, severity: str = "moderate") -> tuple[bytes, str]:
    """
    Full deblurring & enhancement pipeline.  Returns (enhanced_bytes, method_log).
    Pipeline stages (applied in order):
    1. Upscale small images to improve OCR accuracy.
    2. Denoise (mild NLM pass).
    3. Wiener deconvolution on grey channel   — removes Gaussian/defocus blur.
    4. Colour unsharp masking                 — sharpens edges/text.
    5. CLAHE                                  — restores contrast in dark areas.
    6. Final light sharpening pass.
    Strength is tuned to blur severity:
        severe   → aggressive PSF + strong unsharp
        moderate → standard settings
        mild     → gentle enhancement only
    """
    img = Image.open(BytesIO(content)).convert("RGB")
    img_np = np.array(img)
    methods_used = []

    # ── Stage 1: Upscale if too small ──────────────────────────────────
    h, w = img_np.shape[:2]
    target_short = 1200
    short_side = min(h, w)
    if short_side < target_short:
        scale = target_short / short_side
        new_h = int(h * scale)
        new_w = int(w * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        methods_used.append(f"upscale({new_w}×{new_h})")

    # ── Stage 2: Denoise ──────────────────────────────────────────────
    if severity in ("severe", "moderate"):
        h_param = 8 if severity == "severe" else 5
        img_np = _denoise(img_np, h=h_param)
        methods_used.append(f"NLM-denoise(h={h_param})")

    # ── Stage 3: Wiener deconvolution (grey channel) ───────────────────
    if severity != "mild":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        psf_size = 9 if severity == "severe" else 5
        noise_ratio = 0.01 if severity == "severe" else 0.025
        restored = _wiener_deconvolution(gray, psf_size, noise_ratio)
        # Blend restored grey back: convert to LAB, replace L
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = restored
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        methods_used.append(f"Wiener(psf={psf_size},K={noise_ratio})")

    # ── Stage 4: Unsharp masking ───────────────────────────────────────
    strength_map = {"severe": 2.2, "moderate": 1.8, "mild": 1.2}
    radius_map = {"severe": 4, "moderate": 3, "mild": 2}
    strength = strength_map.get(severity, 1.8)
    radius = radius_map.get(severity, 3)
    img_np = _unsharp_mask(img_np, strength=strength, radius=radius)
    methods_used.append(f"unsharp(s={strength},r={radius})")

    # ── Stage 5: CLAHE contrast enhancement ───────────────────────────
    clip_map = {"severe": 3.0, "moderate": 2.5, "mild": 1.8}
    clip = clip_map.get(severity, 2.5)
    img_np = _apply_clahe(img_np, clip=clip)
    methods_used.append(f"CLAHE(clip={clip})")

    # ── Stage 6: Mild final sharpening pass ───────────────────────────
    sharpen_kernel = np.array(
        [[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]], dtype=np.float32
    )
    img_np = cv2.filter2D(img_np, -1, sharpen_kernel)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    methods_used.append("sharpen-kernel")

    # ── Encode to bytes ───────────────────────────────────────────────
    pil_out = Image.fromarray(img_np)
    buf = BytesIO()
    pil_out.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), " → ".join(methods_used)


def image_to_b64(content: bytes) -> str:
    """Convert raw image bytes to a base-64 data-URL for front-end display."""
    return "data:image/jpeg;base64," + base64.b64encode(content).decode()


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3: OCR QUALITY COMPARISON HELPER
# ══════════════════════════════════════════════════════════════════════


def _ocr_quality_score(ocr_result: dict) -> float:
    """
    Score an OCR result for quality comparison.
    Higher is better.  Used to choose original vs deblurred image.
    """
    return (
        ocr_result.get("word_count", 0) * 0.6
        + ocr_result.get("avg_confidence", 0) * 100 * 0.4
    )


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4: LABEL CONTENT DETECTION (unchanged logic, kept intact)
# ══════════════════════════════════════════════════════════════════════

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
]

# BUG FIX: words like 'wheat','milk','salt','oil' are in LABEL_KEYWORDS but
# also appear on the FRONT of a pack. These NUTRITION TABLE ANCHORS are specific
# to the back label — at least 2 must be present to confirm a nutrition panel.
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

    text_lower = ocr_text.lower()
    label_hits = [kw for kw in LABEL_KEYWORDS if kw in text_lower]
    front_hits = [kw for kw in FRONT_PACK_SIGNALS if kw in text_lower]
    # Count how many nutrition-table-specific anchors are present
    anchor_hits = [kw for kw in NUTRITION_TABLE_ANCHORS if kw in text_lower]

    label_score = len(label_hits)
    front_score = len(front_hits)
    anchor_score = len(anchor_hits)

    # BUG FIX: require at least 2 nutrition-table anchors to confirm a back label.
    # Without this, front-of-pack images that mention "wheat / milk / salt / oil"
    # reached label_score >= 3 and were incorrectly analysed.
    has_nutrition_table = anchor_score >= 2

    if has_nutrition_table and label_score >= 3:
        return {
            "has_label": True,
            "confidence": "high" if label_score >= 6 else "medium",
            "label_hits": label_hits[:5],
            "front_hits": front_hits[:3],
            "suggestion": None,
        }
    elif has_nutrition_table and label_score >= 1 and front_score <= 2:
        return {
            "has_label": True,
            "confidence": "low",
            "label_hits": label_hits,
            "front_hits": front_hits,
            "suggestion": None,
        }
    elif front_score > label_score or not has_nutrition_table:
        suggestion = "wrong_side" if front_score > 0 else "no_label"
        return {
            "has_label": False,
            "confidence": "high",
            "label_hits": label_hits,
            "front_hits": front_hits[:3],
            "suggestion": suggestion,
        }
    else:
        return {
            "has_label": True,
            "confidence": "low",
            "label_hits": label_hits,
            "front_hits": front_hits,
            "suggestion": "partial",
        }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5: OCR
# ══════════════════════════════════════════════════════════════════════


def get_server_ocr(content: bytes, lang_hint: str = "en") -> dict:
    img_hash = hashlib.md5(content).hexdigest()
    cache_key = f"{img_hash}_{lang_hint}"
    if cache_key in ocr_cache:
        return ocr_cache[cache_key]

    img = Image.open(BytesIO(content)).convert("RGB")
    w, h = img.size
    max_dim = 1600
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    img_np = np.array(img)

    def _run_ocr(lang):
        active_reader = get_reader_for(lang)
        results = active_reader.readtext(img_np, detail=1)
        words = [r[1] for r in results]
        confidences = [r[2] for r in results]
        text = " ".join(words)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        word_count = len(words)
        return {
            "text": text,
            "word_count": word_count,
            "avg_confidence": round(avg_conf, 3),
            "is_readable": word_count >= 3 and avg_conf > 0.15,
        }

    result = _run_ocr(lang_hint)

    # INDIAN SCRIPT FALLBACK
    if lang_hint == "en" and result["avg_confidence"] < 0.60:
        logger.info("English confidence < 60%, triggering Indian Script Fallback (hi, ta)...")
        res_hi = _run_ocr("hi")
        res_ta = _run_ocr("ta")
        
        best_fallback = sorted([res_hi, res_ta], key=lambda x: x["avg_confidence"], reverse=True)[0]
        if best_fallback["avg_confidence"] > result["avg_confidence"]:
            result = best_fallback
            logger.info("Indian script fallback selected over English.")

    ocr_cache[cache_key] = result
    save_cache(ocr_cache, OCR_CACHE_FILE)
    return result


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6: WEB SEARCH & UTILITIES
# ══════════════════════════════════════════════════════════════════════


def get_live_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [
                f"{r['title']}: {r['body']}" for r in ddgs.text(query, max_results=3)
            ]
        return "\n".join(results)
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return "No web data available."


LANGUAGE_MAP = {
    "en": "English",
    "zh": "Simplified Chinese (简体中文)",
    "es": "Spanish (Español)",
    "ar": "Arabic (العربية)",
    "fr": "French (Français)",
    "hi": "Hindi (हिन्दी)",
    "pt": "Portuguese (Português)",
    "de": "German (Deutsch)",
}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 7: ROUTES
# ══════════════════════════════════════════════════════════════════════


@app.get("/")
async def home():
    return FileResponse("index.html")


@app.post("/check-image")
@limiter.limit("30/minute")
async def check_image(request: Request, image: UploadFile = File(...)):
    """
    Pre-flight image quality check.
    Returns multi-method blur scores + severity.
    """
    content = await image.read()
    return assess_image_quality(content)


@app.post("/enhance-preview")
@limiter.limit("20/minute")
async def enhance_preview(request: Request, image: UploadFile = File(...)):
    """
    Deblur an image and return the result as a base-64 JPEG.
    Useful for showing the user what the enhanced image looks like
    before running a full analysis.
    """
    content = await image.read()
    quality = assess_image_quality(content)

    if not quality["is_blurry"]:
        return JSONResponse(
            {
                "deblurred": False,
                "message": "Image is already clear — no enhancement needed.",
                "quality": quality,
            }
        )

    enhanced_bytes, method_log = deblur_and_enhance(content, quality["blur_severity"])
    b64 = image_to_b64(enhanced_bytes)

    return JSONResponse(
        {
            "deblurred": True,
            "image_b64": b64,
            "method_log": method_log,
            "blur_severity": quality["blur_severity"],
            "quality_before": quality,
        }
    )


@app.post("/ocr")
@limiter.limit("20/minute")
async def perform_ocr(
    request: Request,
    image: UploadFile = File(...),
    language: str = Form("en"),
):
    """Perform OCR and return text + readability assessment."""
    content = await image.read()
    result = get_server_ocr(content, language)
    return result


@app.post("/analyze")
@limiter.limit("15/minute")
async def analyze_product(
    request: Request,
    persona: str = Form(...),
    age_group: str = Form("adult"),
    product_category: str = Form("general"),
    language: str = Form("en"),
    extracted_text: str = Form(None),
    image: UploadFile = File(...),
):
    """
    Full nutrition-label analysis pipeline with automatic blur correction.
    Processing steps:
    1. Multi-method blur detection.
    2. If blurry → deblur/enhance, then run OCR on BOTH versions and
       keep whichever yields better text quality.
    3. Label presence detection.
    4. AI analysis via Groq LLM.
    5. Returns analysis JSON with blur_info metadata.
    """
    if not client:
        return {"error": "Server Error: Missing GROQ_API_KEY in Settings"}

    # ── Scan gate (Task 11) ───────────────────────────────────────────
    device_key = get_device_key(request)
    scan_check = check_and_increment_scan(device_key)
    if not scan_check["allowed"]:
        return JSONResponse(
            status_code=402,
            content={
                "error": "scan_limit_reached",
                "message": f"You've used all {FREE_SCAN_LIMIT} free scans this month.",
                "upgrade_url": "/pro",
                "scans_used": scan_check["scans_used"],
            },
        )

    try:
        content = await image.read()

        # ── Step 1: Blur Detection ────────────────────────────────────
        quality = assess_image_quality(content)
        blur_info = {
            "detected": quality["is_blurry"],
            "severity": quality["blur_severity"],
            "score": quality["blur_score"],
            "deblurred": False,
            "method_log": None,
            "image_b64": None,
            "ocr_source": "original",
        }

        working_content = content  # may be swapped for deblurred version

        # ── Step 2: Conditional Deblurring ───────────────────────────
        if quality["is_blurry"]:
            logger.info(
                f"Blur detected — severity={quality['blur_severity']}, "
                f"composite_score={quality['blur_score']}"
            )
            try:
                enhanced_bytes, method_log = deblur_and_enhance(
                    content, quality["blur_severity"]
                )

                # Run OCR on both and compare quality
                ocr_orig = get_server_ocr(content, language)
                ocr_enhanced = get_server_ocr(enhanced_bytes, language)

                orig_score = _ocr_quality_score(ocr_orig)
                enhanced_score = _ocr_quality_score(ocr_enhanced)

                logger.info(
                    f"OCR quality — original: {orig_score:.1f}, "
                    f"enhanced: {enhanced_score:.1f}"
                )

                if enhanced_score >= orig_score * 0.85:
                    # Enhanced is at least 85% as good → prefer it
                    working_content = enhanced_bytes
                    blur_info["deblurred"] = True
                    blur_info["method_log"] = method_log
                    blur_info["image_b64"] = image_to_b64(enhanced_bytes)
                    blur_info["ocr_source"] = "deblurred"
                    extracted_text = None  # force re-OCR from enhanced image
                    logger.info("Using deblurred image for analysis.")
                else:
                    logger.info("Original OCR was better; keeping original.")

            except Exception as e:
                logger.warning(f"Deblurring failed, using original: {e}")

        # ── Step 3: OCR ───────────────────────────────────────────────
        if not extracted_text:
            ocr_result = get_server_ocr(working_content, language)
            extracted_text = ocr_result["text"]
            ocr_word_count = ocr_result["word_count"]

            # ── Step 3a: Hard gate — reject if OCR confidence too low ─
            if ocr_result["avg_confidence"] < 0.70:
                return {
                    "error": "blurry_image",
                    "message": f"⚠️ Image too blurry (confidence: {ocr_result['avg_confidence']:.0%}). Please retake the photo in better lighting.",
                }
        else:
            ocr_word_count = len(extracted_text.split())

        # ── Step 3b: Hard-block if truly no text ──────────────────────
        if not extracted_text or ocr_word_count == 0:
            return {
                "error": "no_text",
                "message": "No text found on this image. Make sure the label side is facing the camera.",
                "tip": "flip_product",
            }

        # ── Step 3b: Label presence check ─────────────────────────────
        label_check = detect_label_presence(extracted_text)
        if not label_check["has_label"]:
            if label_check["suggestion"] == "wrong_side":
                return {
                    "error": "no_label",
                    "message": "This looks like the front of the product. Please flip it over and scan the back label.",
                    "tip": "wrong_side",
                    "front_words_found": label_check.get("front_hits", []),
                }
            else:
                return {
                    "error": "no_label",
                    "message": "Could not find nutrition or ingredient information. Please upload a clear photo of the back label.",
                    "tip": "flip_product",
                }

        label_confidence = label_check.get("confidence", "medium")

        # ── Step 4: Cache lookup ──────────────────────────────────────
        # Cache key — v2 prefix invalidates any old cached results that had
        # the score=7-anchor bug (they would forever return score≈6).
        cache_key = f"v2:{language}:{persona}:{age_group}:{extracted_text[:80]}"
        if cache_key in ai_cache:
            cached = dict(ai_cache[cache_key])
            cached["blur_info"] = blur_info  # always inject fresh blur_info
            return cached

        # ── Step 5: Web search ─────────────────────────────────────────
        web_context = get_live_search(
            f"health analysis ingredients {extracted_text[:120]}"
        )

        # ── Step 6: Prompt construction ────────────────────────────────
        lang_name = LANGUAGE_MAP.get(language, "English")
        output_lang_instr = (
            f"CRITICAL: Respond ENTIRELY in {lang_name}. "
            f"Every single field value must be in {lang_name}."
        )
        confidence_note = (
            "Note: label text may be partially visible. Do your best with available information and set confidence=low in response."
            if label_confidence == "low"
            else ""
        )

        # Blur context for the AI — helps it interpret partially illegible text
        blur_context = ""
        if blur_info["detected"]:
            if blur_info["deblurred"]:
                blur_context = (
                    f"Note: The image was detected as {blur_info['severity']}ly blurry and "
                    f"has been enhanced using advanced deblurring. "
                    f"The OCR text was extracted from the enhanced image. "
                    f"Some characters might still be uncertain — prioritise nutrients "
                    f"and ingredients you can identify with high confidence."
                )
            else:
                blur_context = (
                    f"Note: The image has some blur (severity: {blur_info['severity']}). "
                    f"OCR was run on the original image. Where text is ambiguous, "
                    f"use your domain knowledge to infer likely values."
                )

        persona_rules = "GENERAL ADULT: Apply standard FSSAI limits."
        p_lower = persona.lower()
        if "diabetic" in p_lower:
            persona_rules = "DIABETIC RULES: Multiply sugar penalty by 3x. Flag any Maltodextrin or Dextrose exactly like sugar."
        elif "child" in p_lower or "baby" in p_lower or "parent" in p_lower:
            persona_rules = "CHILD RULES: Multiply sodium penalty by 2x. Flag all artificial colors."
        elif "pregnant" in p_lower:
            persona_rules = "PREGNANCY RULES: Give bonus for folic acid and protein. Flag any raw/unpasteurized ingredients."
        elif "senior" in p_lower:
            persona_rules = "SENIOR RULES: Flag low fiber content strongly. Flag high sodium."
        elif "gym" in p_lower or "athlete" in p_lower or "fitness" in p_lower:
            persona_rules = "ATHLETE RULES: High sugar allowed if pre/post workout. High protein gives score bonus."

        prompt = f"""
[INST] You are an expert nutritional scientist and health auditor. Analyze the product label below.
{output_lang_instr}
Target Persona: {persona}
Age Group: {age_group}
Product Category: {product_category}
{persona_rules}
{confidence_note}
{blur_context}
Label Text: "{extracted_text}"
Web Context: "{web_context}"

Return ONLY valid JSON — no markdown, no preamble — with this exact structure:
{{
    "product_name": "Short product name from the label",
    "product_category": "Detected category (e.g. Snack, Dairy, Beverage)",
    "score": <INTEGER 1-10 based on SCORING RUBRIC below — modified by persona rules>,
    "verdict": "Two-word verdict",
    "fake_claim_detected": <true if text claims 'No Added Sugar'/'Sugar-Free' BUT ingredients have Maltodextrin, Dextrose, Fructose, Corn Syrup, Date Syrup>,
    "ingredients_raw": "Comma, separated, list, of, ingredients, extracted",
    "chart_data": [<Safe%>, <Moderate%>, <Risky%>],
    "summary": "Professional 2-sentence summary in {lang_name}.",
    "eli5_explanation": "Explain using simple words and emojis for a child in {lang_name}.",
    "molecular_insight": "Explain the biochemical/chemical impact on the body in {lang_name}.",
    "paragraph_benefits": "One full paragraph about the product's main benefits in {lang_name}.",
    "paragraph_uniqueness": "If this product has unique characteristics, describe them. Otherwise suggest 2 better alternatives. Write in {lang_name}.",
    "is_unique": <BOOLEAN true if it has unique characteristics, false otherwise>,
    "nutrient_breakdown": [
        {{"name": "Protein", "value": <ACTUAL g from label>, "unit": "g", "rating": "good", "impact": "Brief impact note in {lang_name}"}},
        {{"name": "Sugar", "value": <ACTUAL g from label>, "unit": "g", "rating": "moderate", "impact": "Brief impact note in {lang_name}"}},
        {{"name": "Fat", "value": <ACTUAL g from label>, "unit": "g", "rating": "good", "impact": "Brief impact note in {lang_name}"}},
        {{"name": "Sodium", "value": <ACTUAL mg from label>, "unit": "mg", "rating": "caution", "impact": "Brief impact note in {lang_name}"}},
        {{"name": "Fiber", "value": <ACTUAL g from label>, "unit": "g", "rating": "good", "impact": "Brief impact note in {lang_name}"}}
    ],
    "pros": ["Benefit 1 in {lang_name}", "Benefit 2", "Benefit 3"],
    "cons": ["Risk 1 in {lang_name}", "Risk 2"],
    "age_warnings": [
        {{"group": "Children", "emoji": "👶", "status": "warning", "message": "Warning or approval in {lang_name}"}},
        {{"group": "Adults", "emoji": "🧑", "status": "good", "message": "Info in {lang_name}"}},
        {{"group": "Seniors", "emoji": "👴", "status": "caution", "message": "Advice in {lang_name}"}},
        {{"group": "Pregnant", "emoji": "🤰", "status": "caution", "message": "Safety info in {lang_name}"}}
    ],
    "better_alternative": "A specific healthier alternative product in {lang_name}."
}}
STRICT SCORING RUBRIC — score MUST reflect actual label nutrition, not examples:
  9-10 : Whole food / minimal processing, no added sugar, low sodium, high fiber/protein
  7-8  : Moderately processed, low sugar (<5g/100g), reasonable sodium, decent nutrients
  5-6  : Processed, moderate sugar (5-15g/100g) OR moderate sodium (400-700mg/100g)
  3-4  : High sugar (>15g/100g) OR high sodium (>700mg/100g) OR poor nutrient profile
  1-2  : Ultra-processed, very high sugar/sodium/saturated fat, minimal nutritional value

RULES:
- score MUST match the actual nutrient values found — modified by persona rules
- chart_data must be [Safe%, Moderate%, Risky%] summing to exactly 100
- nutrient "rating" must be one of: "good", "moderate", "caution", "bad"
- age_warnings "status" must be one of: "good", "caution", "warning"
- All text values MUST be in {lang_name}
- Extract ACTUAL values from the label text, do NOT use placeholder numbers
[/INST]
        """

        # ── Step 7: Groq LLM call ─────────────────────────────────────
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"Primary model failed, using fallback: {e}")
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

        result = json.loads(completion.choices[0].message.content)

        # ── Step 7.5: Lie Detector & NOVA Classifier ──────────────────
        if result.get("fake_claim_detected"):
            result["verdict"] = "🚨 FAKE CLAIM: Brand claims 'No Sugar' but contains Sugar alternatives"
            result["score"] = min(result.get("score", 5), 2)
            if "Hidden Sugar" not in result.get("cons", []):
                result.setdefault("cons", []).append("Hidden Sugar (Maltodextrin/Dextrose)")

        raw_ing = str(result.get("ingredients_raw", "")).lower()
        if raw_ing:
            nova_triggers = [
                "e471", "e442", "glycerol", "aspartame", "sucralose", 
                "hydrogenated", "maltodextrin", "emulsifier", "artificial"
            ]
            hits = [t for t in nova_triggers if t in raw_ing]
            if len(hits) >= 2:
                result["nova_group"] = 4
                result.setdefault("cons", []).append("⚠️ NOVA 4: Ultra-processed food")
                result["score"] = min(result.get("score", 5), 3)

        # ── Step 8: Validate chart_data ───────────────────────────────
        if "chart_data" in result:
            cd = result["chart_data"]
            if len(cd) == 3:
                total = sum(cd)
                if total != 100 and total > 0:
                    scaled = [round(v * 100 / total) for v in cd]
                    scaled[scaled.index(max(scaled))] += 100 - sum(scaled)
                    result["chart_data"] = scaled

        # ── Step 8b: Atwater Math Check ───────────────────────────────
        def _get_nutrient(key):
            for n in result.get("nutrient_breakdown", []):
                if key in n.get("name", "").lower():
                    v = n.get("value", 0)
                    return float(v) if isinstance(v, (int, float)) else 0
            return 0

        stated_cal = (
            _get_nutrient("calorie") or _get_nutrient("energy") or _get_nutrient("kcal")
        )
        protein = _get_nutrient("protein")
        carbs = _get_nutrient("carbohydrate") or _get_nutrient("carbs")
        fat = _get_nutrient("fat")

        if stated_cal > 0 and (protein > 0 or carbs > 0 or fat > 0):
            calculated_cal = (protein * 4) + (carbs * 4) + (fat * 9)
            margin = stated_cal * 0.15
            if abs(calculated_cal - stated_cal) > margin:
                result["atwater_warning"] = {
                    "error": "atwater_mismatch",
                    "message": f"Nutrient math mismatch: stated {stated_cal} kcal vs calculated {calculated_cal:.0f} kcal from macros.",
                    "stated_calories": stated_cal,
                    "calculated_calories": round(calculated_cal, 1),
                }
                result["is_low_confidence"] = True

        # ── Step 9: Attach blur metadata ─────────────────────────────
        result["blur_info"] = blur_info

        # ── Step 9b: Attach scan metadata (Task 11) ──────────────────
        result["scan_meta"] = {
            "scans_remaining": scan_check["scans_remaining"],
            "is_pro": scan_check["is_pro"],
            "scans_used": scan_check["scans_used"],
        }

        # ── Step 10: Cache & return ───────────────────────────────────
        ai_cache[cache_key] = result
        save_cache(ai_cache, AI_CACHE_FILE)
        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": f"Scan failed: {str(e)[:100]}... Please try again."}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 8: NEW ENDPOINTS (Tasks 6, 11, 13, 14, 18)
# ══════════════════════════════════════════════════════════════════════


# ── Health check (for Dockerfile HEALTHCHECK) ─────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0"}


# ── Pro activation via Razorpay (Task 11) ─────────────────────────────
@app.post("/activate-pro")
async def activate_pro(request: Request, payment_id: str = Form(...)):
    """Called after Razorpay payment confirmation. Marks device as Pro."""
    global scan_limits
    scan_limits = load_scan_limits()
    device_key = get_device_key(request)
    if device_key not in scan_limits:
        scan_limits[device_key] = {}
    scan_limits[device_key]["pro"] = True
    scan_limits[device_key]["month"] = datetime.date.today().isoformat()[:7]
    scan_limits[device_key]["count"] = scan_limits[device_key].get("count", 0)
    save_scan_limits(scan_limits)
    logger.info(f"Pro activated for device {device_key}, payment_id={payment_id}")
    return {
        "status": "activated",
        "message": "Pro activated! Unlimited scans unlocked.",
    }


# ── Scan status check (for frontend banner) ───────────────────────────
@app.get("/scan-status")
async def scan_status(request: Request):
    """Returns remaining scans and pro status for the current device."""
    global scan_limits
    scan_limits = load_scan_limits()
    device_key = get_device_key(request)
    month_key = datetime.date.today().isoformat()[:7]
    u = scan_limits.get(device_key, {})
    if u.get("month") != month_key:
        return {
            "scans_used": 0,
            "scans_remaining": FREE_SCAN_LIMIT,
            "is_pro": False,
            "limit": FREE_SCAN_LIMIT,
        }
    used = u.get("count", 0)
    return {
        "scans_used": used,
        "scans_remaining": 9999 if u.get("pro") else max(0, FREE_SCAN_LIMIT - used),
        "is_pro": u.get("pro", False),
        "limit": FREE_SCAN_LIMIT,
    }


# ── Shareable PNG card (Task 6) ───────────────────────────────────────
@app.post("/generate-share-card")
@limiter.limit("20/minute")
async def generate_share_card(
    request: Request,
    product_name: str = Form(...),
    score: int = Form(...),
    verdict: str = Form(...),
    top_warning: str = Form(""),
    top_pro: str = Form(""),
):
    """Generate a 1080×1080 shareable PNG card for Instagram/WhatsApp."""
    W, H = 1080, 1080
    BG = (15, 17, 23)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    score_rgb = (
        (34, 197, 94) if score >= 7 else (245, 158, 11) if score >= 4 else (239, 68, 68)
    )

    # Score ring
    ring_box = [340, 160, 740, 560]
    draw.ellipse(ring_box, outline=score_rgb, width=18)
    draw.text((540, 360), str(score), fill=score_rgb, anchor="mm", font=font)
    draw.text((540, 420), "/10", fill=(100, 116, 139), anchor="mm", font=font)

    # Product name (truncate)
    pname = product_name[:38] + ("…" if len(product_name) > 38 else "")
    draw.text((540, 610), pname, fill=(255, 255, 255), anchor="mm", font=font)
    draw.text((540, 660), verdict[:50], fill=(148, 163, 184), anchor="mm", font=font)

    # Pro banner
    if top_pro:
        draw.rectangle([60, 700, 1020, 760], fill=(15, 60, 40))
        draw.text(
            (540, 730), f"✓ {top_pro[:65]}", fill=(74, 222, 128), anchor="mm", font=font
        )

    # Warning banner
    if top_warning:
        draw.rectangle([60, 775, 1020, 840], fill=(124, 29, 29))
        draw.text(
            (540, 807),
            f"⚠ {top_warning[:65]}",
            fill=(252, 165, 165),
            anchor="mm",
            font=font,
        )

    # Branding
    draw.text(
        (540, 1010),
        "eatlytic.com  •  scan any food label, no barcode needed",
        fill=(71, 85, 105),
        anchor="mm",
        font=font,
    )

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=eatlytic-scan.png"},
    )


# ── B2B API endpoint with API key auth (Task 13) ──────────────────────
@app.post("/api/v1/analyze")
@limiter.limit("60/minute")
async def api_analyze(
    request: Request,
    image: UploadFile = File(...),
    language: str = Form("en"),
    persona: str = Form("general adult"),
    age_group: str = Form("adult"),
    api_key_data: dict = Security(verify_api_key),
):
    """B2B API endpoint — requires X-API-Key header."""
    if not api_key_data:
        raise HTTPException(
            status_code=401, detail="Invalid API key. Get one at eatlytic.com/api"
        )
    if not api_key_data.get("active"):
        raise HTTPException(status_code=403, detail="API key suspended.")

    global api_keys_db
    api_keys_db = load_api_keys()
    month_key = datetime.date.today().isoformat()[:7]
    if api_key_data.get("month") != month_key:
        api_key_data["month"] = month_key
        api_key_data["scans_this_month"] = 0

    LIMITS = {"business": 1000, "enterprise": 99999}
    limit = LIMITS.get(api_key_data["plan"], 1000)
    if api_key_data["scans_this_month"] >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly limit ({limit} scans) reached. Upgrade at eatlytic.com/api",
        )

    content = await image.read()
    quality = assess_image_quality(content)
    working = content
    blur_info = {
        "detected": quality["is_blurry"],
        "severity": quality["blur_severity"],
        "score": quality["blur_score"],
    }
    if quality["is_blurry"]:
        try:
            enhanced, mlog = deblur_and_enhance(content, quality["blur_severity"])
            o_score = _ocr_quality_score(get_server_ocr(content, language))
            e_score = _ocr_quality_score(get_server_ocr(enhanced, language))
            if e_score >= o_score * 0.85:
                working = enhanced
                blur_info["deblurred"] = True
                blur_info["method_log"] = mlog
        except Exception as e:
            logger.warning(f"B2B deblur failed: {e}")

    ocr = get_server_ocr(working, language)
    text = ocr["text"]
    lc = detect_label_presence(text)
    if not lc["has_label"]:
        return {"error": "no_label", "message": "No nutrition label detected in image."}

    cache_key = f"b2b:{language}:{persona}:{text[:80]}"
    if cache_key in ai_cache:
        cached = dict(ai_cache[cache_key])
        cached["blur_info"] = blur_info
        api_key_data["scans_this_month"] += 1
        save_api_keys(api_keys_db)
        return cached

    web_ctx = get_live_search(f"health analysis ingredients {text[:120]}")
    lang_name = LANGUAGE_MAP.get(language, "English")
    prompt = (
        f'[INST] Analyze: "{text}". Web: "{web_ctx}". Persona: {persona}. '
        f"Respond in {lang_name} as valid JSON with: product_name, score(1-10), "
        f"verdict, summary, nutrient_breakdown, pros, cons, age_warnings, better_alternative. [/INST]"
    )

    try:
        comp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        result = json.loads(comp.choices[0].message.content)
        result["blur_info"] = blur_info
        api_key_data["scans_this_month"] += 1
        save_api_keys(api_keys_db)
        result["api_usage"] = {
            "scans_this_month": api_key_data["scans_this_month"],
            "limit": limit,
            "client": api_key_data["name"],
        }
        ai_cache[cache_key] = result
        save_cache(ai_cache, AI_CACHE_FILE)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:100]}")


# ── Admin: create API key (protect with env var in production) ────────
@app.post("/admin/create-api-key")
async def create_api_key_endpoint(
    admin_token: str = Form(...),
    client_name: str = Form(...),
    plan: str = Form("business"),
):
    expected = os.environ.get("ADMIN_TOKEN", "changeme")
    if admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token.")
    key = generate_api_key(client_name, plan)
    return {"api_key": key, "client": client_name, "plan": plan}


# ── PDF export (Task 18) ──────────────────────────────────────────────
@app.post("/export-pdf")
@limiter.limit("10/minute")
async def export_pdf(request: Request, analysis_json: str = Form(...)):
    """Generate a PDF report from analysis JSON. Requires reportlab."""
    try:
        data = json.loads(analysis_json)
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.units import cm
    except ImportError:
        return JSONResponse(
            {"error": "reportlab not installed. Add 'reportlab' to requirements.txt."},
            status_code=501,
        )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    stys = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Eatlytic Food Label Analysis", stys["Title"]))
    story.append(
        Paragraph(f"Product: {data.get('product_name', 'Unknown')}", stys["Heading2"])
    )
    story.append(Spacer(1, 0.4 * cm))

    score = data.get("score", 0)
    sc = "22c55e" if score >= 7 else "f59e0b" if score >= 4 else "ef4444"
    story.append(
        Paragraph(
            f"<font color='#{sc}'>Health Score: {score}/10 — {data.get('verdict', '')}</font>",
            stys["Heading1"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    if data.get("summary"):
        story.append(Paragraph("Summary", stys["Heading2"]))
        story.append(Paragraph(data["summary"], stys["Normal"]))
        story.append(Spacer(1, 0.4 * cm))

    nutrients = data.get("nutrient_breakdown", [])
    if nutrients:
        story.append(Paragraph("Nutrient Breakdown", stys["Heading2"]))
        tbl_data = [["Nutrient", "Amount", "Rating"]]
        for n in nutrients:
            tbl_data.append(
                [
                    n.get("name", ""),
                    f"{n.get('value', '')} {n.get('unit', '')}",
                    n.get("rating", "").upper(),
                ]
            )
        tbl = Table(tbl_data, colWidths=[6 * cm, 4 * cm, 4 * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1D9E75")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [rl_colors.HexColor("#f8faf8"), rl_colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#d0d8d4")),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    if data.get("pros"):
        story.append(Paragraph("Benefits", stys["Heading2"]))
        for p in data["pros"]:
            story.append(Paragraph(f"✓ {p}", stys["Normal"]))
    if data.get("cons"):
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Concerns", stys["Heading2"]))
        for c in data["cons"]:
            story.append(Paragraph(f"✗ {c}", stys["Normal"]))

    if data.get("age_warnings"):
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Age-Group Warnings", stys["Heading2"]))
        for w in data["age_warnings"]:
            story.append(
                Paragraph(
                    f"{w.get('emoji', '')} {w.get('group', '')} — {w.get('message', '')}",
                    stys["Normal"],
                )
            )

    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "Generated by Eatlytic — eatlytic.com  |  AI food label analysis",
            ParagraphStyle(
                "footer", parent=stys["Normal"], fontSize=8, textColor=rl_colors.grey
            ),
        )
    )

    doc.build(story)
    buf.seek(0)
    safe_name = (
        data.get("product_name", "scan").replace(" ", "-").replace("/", "-")[:40]
    )
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=eatlytic-{safe_name}.pdf"
        },
    )


# ── WhatsApp webhook (Task 7) — requires twilio in requirements.txt ───
@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """Twilio WhatsApp sandbox webhook."""
    try:
        from twilio.twiml.messaging_response import MessagingResponse
    except ImportError:
        return Response(
            content="<Response><Message>twilio not installed.</Message></Response>",
            media_type="application/xml",
        )

    form = await request.form()
    media_url = form.get("MediaUrl0")
    resp = MessagingResponse()
    msg = resp.message()

    if media_url:
        try:
            import httpx

            TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
            TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
            async with httpx.AsyncClient() as hc:
                img_bytes = (
                    await hc.get(media_url, auth=(TWILIO_SID, TWILIO_TOKEN))
                ).content

            quality = assess_image_quality(img_bytes)
            if quality["is_blurry"]:
                img_bytes, _ = deblur_and_enhance(img_bytes, quality["blur_severity"])

            ocr_r = get_server_ocr(img_bytes, "en")
            lc = detect_label_presence(ocr_r["text"])

            if not lc["has_label"]:
                msg.body(
                    "❌ Couldn't find a nutrition label. "
                    "Please send the *back* of the packaging."
                )
            elif not client:
                msg.body("⚠️ AI service unavailable. Full analysis at *eatlytic.com*")
            else:
                web_ctx = get_live_search(f"health ingredients {ocr_r['text'][:80]}")
                prompt = f"""
[INST] CRITICAL: Respond ONLY in valid JSON. You are an expert.
Target Persona: General Adult
GENERAL ADULT: Apply standard FSSAI limits.
Label Text: "{ocr_r['text']}"
Web Context: "{web_ctx}"
Return exactly this JSON:
{{
    "score": <1-10>,
    "verdict": "Two-word verdict",
    "fake_claim_detected": <true if text claims 'No Added Sugar'/'Sugar-Free' BUT ingredients have Maltodextrin, Dextrose, Fructose, Corn Syrup, Date Syrup>,
    "ingredients_raw": "Comma, separated, list, of, ingredients, extracted",
    "summary": "Professional 2-sentence summary",
    "cons": ["Risk 1", "Risk 2"]
}}
[/INST]"""
                comp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=600,
                    response_format={"type": "json_object"}
                )
                
                res = json.loads(comp.choices[0].message.content)
                
                # Apply Lie Detector + NOVA
                if res.get("fake_claim_detected"):
                    res["verdict"] = "🚨 FAKE CLAIM"
                    res["score"] = min(res.get("score", 5), 2)
                    if "Hidden Sugar" not in res.get("cons", []):
                        res.setdefault("cons", []).append("Hidden Sugar (Maltodextrin/Dextrose/Fructose)")

                raw_ing = str(res.get("ingredients_raw", "")).lower()
                if raw_ing:
                    hits = [t for t in ["e471", "e442", "glycerol", "aspartame", "sucralose", "hydrogenated", "maltodextrin", "emulsifier", "artificial"] if t in raw_ing]
                    if len(hits) >= 2:
                        res["score"] = min(res.get("score", 5), 3)
                        res.setdefault("cons", []).append("⚠️ NOVA 4: Ultra-processed food")

                def generate_whatsapp_response(r: dict, ocr: dict) -> str:
                    if ocr.get("avg_confidence", 1) < 0.75:
                        return "❌ Cannot Score\n\nThe label is too blurry or incomplete to provide a definitive score. Please send a clearer picture of the ingredients list."
                    s = r.get("score", "?")
                    v = r.get("verdict", "Analyzed")
                    summ = r.get("summary", "")
                    fl = r.get("cons", [])
                    flags_t = "\n".join([f"- {f}" for f in fl]) if fl else "- None"
                    return f"Eatlytic Score: {s}/10\n\n[{v}]\n{summ}\n\n⚠️ Risk Flags:\n{flags_t}"

                final_text = generate_whatsapp_response(res, ocr_r)
                msg.body(final_text)
        except Exception as e:
            logger.error(f"WhatsApp error: {e}")
            msg.body("⚠️ Something went wrong. Try again or visit *eatlytic.com*")
    else:
        msg.body(
            "👋 Welcome to *Eatlytic*!\n\n"
            "Send me a photo of any food label (back of pack) "
            "and I'll analyse it instantly.\n\n"
            "Works even on blurry photos 📸\nFree — no barcode needed."
        )

    return Response(content=str(resp), media_type="application/xml")


# ── OCR accuracy test helper (Task 1) ────────────────────────────────
@app.post("/test-accuracy")
@limiter.limit("5/minute")
async def test_accuracy(
    request: Request,
    image: UploadFile = File(...),
    ground_truth: str = Form(""),
):
    """Compare OCR output to ground truth. Returns F1 + blur scores."""
    content = await image.read()
    quality = assess_image_quality(content)

    # Run without blur fix
    ocr_orig = get_server_ocr(content, "en")

    # Run with blur fix if blurry
    ocr_enhanced = None
    if quality["is_blurry"]:
        try:
            enhanced_bytes, mlog = deblur_and_enhance(content, quality["blur_severity"])
            ocr_enhanced = get_server_ocr(enhanced_bytes, "en")
        except Exception:
            pass

    def f1(pred: str, truth: str) -> float:
        if not truth:
            return 0.0
        p_w = set(pred.lower().split())
        t_w = set(truth.lower().split())
        tp = len(p_w & t_w)
        prec = tp / len(p_w) if p_w else 0
        rec = tp / len(t_w) if t_w else 0
        return round(2 * prec * rec / (prec + rec), 3) if (prec + rec) else 0.0

    result = {
        "blur_score": quality["blur_score"],
        "blur_severity": quality["blur_severity"],
        "is_blurry": quality["is_blurry"],
        "original_ocr": {
            "word_count": ocr_orig["word_count"],
            "avg_confidence": ocr_orig["avg_confidence"],
            "f1_vs_truth": f1(ocr_orig["text"], ground_truth),
        },
    }
    if ocr_enhanced:
        result["enhanced_ocr"] = {
            "word_count": ocr_enhanced["word_count"],
            "avg_confidence": ocr_enhanced["avg_confidence"],
            "f1_vs_truth": f1(ocr_enhanced["text"], ground_truth),
            "f1_delta": round(
                f1(ocr_enhanced["text"], ground_truth)
                - f1(ocr_orig["text"], ground_truth),
                3,
            ),
        }
    return result
