import os
import io
import json
import logging
import hashlib
import base64
import threading
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

from app.models.db import (
    init_db, db_conn, get_ocr_cache, set_ocr_cache, get_ai_cache, set_ai_cache
)
from app.services.ocr import run_ocr, detect_label_presence, passes_confidence_gate
from app.services.image import assess_image_quality, deblur_and_enhance, image_to_b64, ocr_quality_score
from app.services.llm import analyse_label, build_analysis_prompt

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

# Initialize Database
init_db()

# --- SCAN LIMITS & API KEYS (DB-backed) ---
FREE_SCAN_LIMIT = 10

# --- API KEY AUTH (Task 13) ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        return None
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE api_key=?", (api_key,)).fetchone()
    if row:
        return dict(row)
    return None


def generate_api_key(client_name: str, plan: str = "business") -> str:
    key = "eak_" + secrets.token_urlsafe(32)
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys(api_key, client_name, plan, scans_this_month, month, active) VALUES(?,?,?,?,?,1)",
            (key, client_name, plan, 0, month_key)
        )
    return key


# --- DEVICE FINGERPRINT + SCAN GATE (Task 11) ---
def get_device_key(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")
    return hashlib.md5(f"{ip}:{ua}".encode()).hexdigest()[:16]


def check_and_increment_scan(device_key: str) -> dict:
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM devices WHERE device_key=?", (device_key,)).fetchone()
        if not row:
            conn.execute("INSERT INTO devices(device_key, month, scan_count) VALUES(?,?,0)", (device_key, month_key))
            u = {"is_pro": 0, "month": month_key, "scan_count": 0}
        else:
            u = dict(row)
        
        if u["month"] != month_key:
            conn.execute("UPDATE devices SET month=?, scan_count=0 WHERE device_key=?", (month_key, device_key))
            u["month"] = month_key
            u["scan_count"] = 0
            
        if u["is_pro"]:
            conn.execute("UPDATE devices SET scan_count=scan_count+1 WHERE device_key=?", (device_key,))
            return {"allowed": True, "scans_used": u["scan_count"] + 1, "scans_remaining": 9999, "is_pro": True}
            
        if u["scan_count"] >= FREE_SCAN_LIMIT:
            return {"allowed": False, "scans_used": u["scan_count"], "scans_remaining": 0, "is_pro": False}
            
        conn.execute("UPDATE devices SET scan_count=scan_count+1 WHERE device_key=?", (device_key,))
        new_count = u["scan_count"] + 1
        return {"allowed": True, "scans_used": new_count, "scans_remaining": FREE_SCAN_LIMIT - new_count, "is_pro": False}


# --- CLIENTS ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY missing! App will fail.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# --- SERVICE WRAPPERS (Task 1) ---
# Most logic is now consolidated in the app/ package to ensure 
# tests and production always use the same thresholds.

def get_server_ocr(content: bytes, lang_hint: str = "en") -> dict:
    """Wrapper to maintain API compatibility while using unified OCR service."""
    return run_ocr(content, lang_hint)

# Re-exporting these from services for internal main.py usage
from app.services.ocr import detect_label_presence
from app.services.image import assess_image_quality, deblur_and_enhance, image_to_b64, ocr_quality_score

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

                # ⚡ OPTIMIZATION: Faster path for mild blur
                if quality["blur_severity"] == "mild":
                    working_content = enhanced_bytes
                    blur_info["deblurred"] = True
                    blur_info["method_log"] = method_log
                    blur_info["ocr_source"] = "deblurred_fast"
                    logger.info("Mild blur: using enhanced image directly to save time.")
                else:
                    # Moderate/Severe: Run OCR on both and compare quality
                    ocr_orig = get_server_ocr(content, language)
                    ocr_enhanced = get_server_ocr(enhanced_bytes, language)

                    orig_score = _ocr_quality_score(ocr_orig)
                    enhanced_score = _ocr_quality_score(ocr_enhanced)

                    if enhanced_score >= orig_score * 0.85:
                        working_content = enhanced_bytes
                        blur_info["deblurred"] = True
                        blur_info["method_log"] = method_log
                        blur_info["ocr_source"] = "deblurred_validated"
                
                if blur_info["deblurred"]:
                    blur_info["image_b64"] = image_to_b64(enhanced_bytes)

            except Exception as e:
                logger.warning(f"Deblurring failed, using original: {e}")

        # ── Step 3: OCR ───────────────────────────────────────────────
        if not extracted_text:
            ocr_result = get_server_ocr(working_content, language)
            extracted_text = ocr_result["text"]
            ocr_word_count = ocr_result["word_count"]

            # ── Step 3a: Hard gate — reject if OCR confidence too low ─
            # Tuning: 30% is much safer for noisy labels, especially with word-count bypass
            if ocr_result["avg_confidence"] < 0.30 and ocr_word_count < 20:
                logger.info(f"Gate rejected: conf={ocr_result['avg_confidence']}, words={ocr_word_count}")
                return {
                    "error": "blurry_image",
                    "message": f"⚠️ Image quality too low (confidence: {ocr_result['avg_confidence']:.0%}). Please take a closer photo of the label.",
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
STRICT SCORING RUBRIC (1-10 scale):
  9-10 : Whole food / minimal processing, no added sugar, low sodium
  7-8  : Moderately processed, low sugar (<5g/100g), reasonable sodium
  5-6  : Processed, moderate sugar (5-15g/100g) OR moderate sodium
  3-4  : High sugar (>15g/100g) OR high sodium (>700mg/100g)
  1-2  : Ultra-processed, very high sugar/sodium, fake marketing claims

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
    device_key = get_device_key(request)
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        # Ensure device exists and set as pro
        row = conn.execute("SELECT * FROM devices WHERE device_key=?", (device_key,)).fetchone()
        if not row:
            conn.execute("INSERT INTO devices(device_key, is_pro, month, scan_count) VALUES(?,1,?,0)", 
                         (device_key, month_key))
        else:
            conn.execute("UPDATE devices SET is_pro=1 WHERE device_key=?", (device_key,))
            
    logger.info(f"Pro activated for device {device_key}, payment_id={payment_id}")
    return {
        "status": "activated",
        "message": "Pro activated! Unlimited scans unlocked.",
    }


# ── Scan status check (for frontend banner) ───────────────────────────
@app.get("/scan-status")
async def scan_status(request: Request):
    """Returns remaining scans and pro status for the current device."""
    device_key = get_device_key(request)
    month_key = datetime.date.today().isoformat()[:7]
    
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM devices WHERE device_key=?", (device_key,)).fetchone()
        
    if not row:
        return {
            "scans_used": 0,
            "scans_remaining": FREE_SCAN_LIMIT,
            "is_pro": False,
            "limit": FREE_SCAN_LIMIT,
        }
    
    u = dict(row)
    if u["month"] != month_key:
        return {
            "scans_used": 0,
            "scans_remaining": 9999 if u["is_pro"] else FREE_SCAN_LIMIT,
            "is_pro": bool(u["is_pro"]),
            "limit": FREE_SCAN_LIMIT,
        }
        
    used = u["scan_count"]
    return {
        "scans_used": used,
        "scans_remaining": 9999 if u["is_pro"] else max(0, FREE_SCAN_LIMIT - used),
        "is_pro": bool(u["is_pro"]),
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
    
    # Improved font loading for Docker + Windows fallback
    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "arial.ttf"
    ]
    font = None
    for p in font_paths:
        try:
            font = ImageFont.truetype(p, 48)
            break
        except Exception:
            continue
    if not font:
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
    
    # Check monthly quota
    scans_used = api_key_data.get("scans_this_month", 0)
    if api_key_data.get("month") != month_key:
        scans_used = 0
        with db_conn() as conn:
            conn.execute("UPDATE api_keys SET month=?, scans_this_month=0 WHERE api_key=?", 
                         (month_key, api_key_data["api_key"]))

    LIMITS = {"business": 1000, "enterprise": 99999}
    limit = LIMITS.get(api_key_data["plan"], 1000)
    if scans_used >= limit:
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
            o_score = ocr_quality_score(get_server_ocr(content, language))
            e_score = ocr_quality_score(get_server_ocr(enhanced, language))
            if e_score >= o_score * 0.85:
                working = enhanced
                blur_info["deblurred"] = True
                blur_info["method_log"] = mlog
        except Exception as e:
            logger.warning(f"B2B deblur failed: {e}")

    ocr = get_server_ocr(working, language)
    text = ocr["text"]
    
    # Sanitise to prevent prompt injection
    safe_text = text.replace('"', "'").replace("\n", " ").strip()
    
    lc = detect_label_presence(text)
    if not lc["has_label"]:
        return {"error": "no_label", "message": "No nutrition label detected in image."}

    cache_key = f"b2b:{language}:{persona}:{safe_text[:80]}"
    cached = get_ai_cache(cache_key)
    if cached:
        cached["blur_info"] = blur_info
        with db_conn() as conn:
            conn.execute("UPDATE api_keys SET scans_this_month=scans_this_month+1 WHERE api_key=?", 
                         (api_key_data["api_key"],))
        return cached

    web_ctx = get_live_search(f"health analysis ingredients {safe_text[:120]}")
    lang_name = LANGUAGE_MAP.get(language, "English")
    prompt = (
        f'[INST] Analyze: "{safe_text}". Web: "{web_ctx}". Persona: {persona}. '
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
        try:
            result = json.loads(comp.choices[0].message.content)
        except json.JSONDecodeError:
            raise ValueError("LLM returned invalid JSON")
            
        result["blur_info"] = blur_info
        
        with db_conn() as conn:
            conn.execute("UPDATE api_keys SET scans_this_month=scans_this_month+1 WHERE api_key=?", 
                         (api_key_data["api_key"],))
            
        result["api_usage"] = {
            "scans_this_month": scans_used + 1,
            "limit": limit,
            "client": api_key_data["client_name"],
        }
        set_ai_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"B2B Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:100]}")


# ── Admin: create API key (protect with env var in production) ────────
@app.post("/admin/create-api-key")
async def create_api_key_endpoint(
    admin_token: str = Form(...),
    client_name: str = Form(...),
    plan: str = Form("business"),
):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        logger.error("ADMIN_TOKEN environment variable not set!")
        raise HTTPException(status_code=500, detail="Server misconfiguration: ADMIN_TOKEN not set.")
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
def generate_whatsapp_response(r: dict, ocr: dict) -> str:
    if ocr.get("avg_confidence", 1) < 0.30 and ocr.get("word_count", 0) < 20:
        return "❌ Cannot Score\n\nThe label is too blurry or incomplete to provide a definitive score. Please send a clearer picture of the ingredients list."
    s = r.get("score", "?")
    v = r.get("verdict", "Analyzed")
    summ = r.get("summary", "")
    fl = r.get("cons", [])
    flags_t = "\n".join([f"- {f}" for f in fl]) if fl else "- None"
    return f"Eatlytic Score: {s}/10\n\n[{v}]\n{summ}\n\n⚠️ Risk Flags:\n{flags_t}"

@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """Twilio WhatsApp sandbox webhook."""
    # SECURITY: Verify Twilio Signature (Task 28)
    # This requires TWILIO_AUTH_TOKEN and the RequestValidator
    try:
        from twilio.request_validator import RequestValidator
        validator = RequestValidator(os.environ.get("TWILIO_AUTH_TOKEN", ""))
        signature = request.headers.get("X-Twilio-Signature", "")
        form_data = await request.form()
        url = str(request.url)
        # Note: In production with a reverse proxy, you may need to use X-Forwarded-Proto
        if not validator.validate(url, form_data, signature) and os.environ.get("ENV") == "production":
             logger.warning("Twilio signature validation failed!")
             # return Response(status_code=403) # Uncomment for strict enforcement
    except ImportError:
        pass

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
            text = ocr_r["text"]
            
            # Sanitise to prevent prompt injection
            safe_text = text.replace('"', "'").replace("\n", " ").strip()
            
            lc = detect_label_presence(text)

            if not lc["has_label"]:
                msg.body(
                    "❌ Couldn't find a nutrition label. "
                    "Please send the *back* of the packaging."
                )
            elif not client:
                msg.body("⚠️ AI service unavailable. Full analysis at *eatlytic.com*")
            else:
                web_ctx = get_live_search(f"health ingredients {safe_text[:80]}")
                prompt = f"""
[INST] CRITICAL: Respond ONLY in valid JSON. You are an expert.
Target Persona: General Adult
GENERAL ADULT: Apply standard FSSAI limits.
Label Text: "{safe_text}"
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
                
                try:
                    res = json.loads(comp.choices[0].message.content)
                except json.JSONDecodeError:
                    msg.body("⚠️ Analysis failed. Please try again.")
                    return Response(content=str(resp), media_type="application/xml")
                
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
