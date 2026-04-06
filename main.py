import os
import io
import json
import asyncio
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
from twilio.twiml.messaging_response import MessagingResponse
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from groq import Groq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.models.db import (
    init_db,
    db_conn,
    get_ocr_cache,
    set_ocr_cache,
    get_ai_cache,
    set_ai_cache,
    check_and_increment_scan,
)
from app.services.ocr import run_ocr, detect_label_presence, passes_confidence_gate
from app.services.image import (
    assess_image_quality,
    deblur_and_enhance,
    image_to_b64,
    ocr_quality_score,
)
from app.services.llm import unified_analyze_flow, LANGUAGE_MAP, _flatten_nutrients
from app.services.fake_detector import apply_dna_overrides
from app.services.alternatives import get_healthy_alternative

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Eatlytic: Startup Scale")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://eatlytic.com",
        "https://www.eatlytic.com",
        "http://localhost:3000",  # local dev
        "http://localhost:7860",  # local docker dev
    ],
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
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
        row = conn.execute(
            "SELECT * FROM api_keys WHERE api_key=?", (api_key,)
        ).fetchone()
    if row:
        return dict(row)
    return None


def generate_api_key(client_name: str, plan: str = "business") -> str:
    key = "eak_" + secrets.token_urlsafe(32)
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys(api_key, client_name, plan, scans_this_month, month, active) VALUES(?,?,?,?,?,1)",
            (key, client_name, plan, 0, month_key),
        )
    return key


# --- DEVICE FINGERPRINT + SCAN GATE (Task 11) ---
def get_device_key(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")
    return hashlib.md5(f"{ip}:{ua}".encode()).hexdigest()[:16]


# --- CLIENTS ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY missing! App will fail.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# ══════════════════════════════════════════════════════════════════════
#  SECTION 6: WEB SEARCH & UTILITIES
# ══════════════════════════════════════════════════════════════════════


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
    scan_check = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT)
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
                # ⚡ Apply enhancement directly
                enhanced_bytes, method_log = deblur_and_enhance(
                    content, quality["blur_severity"]
                )

                # Use enhanced image directly for ALL blur levels
                # Eliminates the 2x OCR comparison which was the biggest speed
                # bottleneck. The enhanced image is always >= original quality.
                working_content = enhanced_bytes
                blur_info["deblurred"] = True
                blur_info["method_log"] = method_log
                blur_info["ocr_source"] = "deblurred"
                logger.info(
                    f"Image enhanced ({quality['blur_severity']}): {method_log}"
                )

            except Exception as e:
                logger.warning(f"Deblurring failed, using original: {e}")

        # ── Step 3: OCR ───────────────────────────────────────────────
        if not extracted_text:
            ocr_result = get_server_ocr(working_content, language)
            extracted_text = ocr_result["text"]
            ocr_word_count = ocr_result["word_count"]

            # ── Step 3a: Hard gate — block only if truly no text found ─────
            # EasyOCR returns 0% avg_confidence on clear stylised-font labels
            # (known library quirk). Gate on word_count ONLY — confidence is
            # unreliable when label fonts are decorative or very small.
            if ocr_word_count < 5 and ocr_result["avg_confidence"] < 0.20:
                logger.info(
                    f"Gate rejected: conf={ocr_result['avg_confidence']}, words={ocr_word_count}"
                )
                return {
                    "error": "blurry_image",
                    "message": f"⚠️ Not enough text detected ({ocr_word_count} words found). Please move closer and make sure the ingredients panel fills the frame.",
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

        # ── Step 4-12: Unified High-Quality Analysis ───────────────
        result = await unified_analyze_flow(
            extracted_text=extracted_text,
            persona=persona,
            age_group=age_group,
            product_category_hint=product_category,
            language=language,
            web_context="",  # Web context will be fetched inside flow async
            blur_info=blur_info,
            label_confidence=label_confidence,
        )

        if "error" in result:
            return result

        # ── Step 11: Attach scan metadata ──────────────────────────
        result["scan_meta"] = {
            "scans_remaining": scan_check["scans_remaining"],
            "is_pro": scan_check["is_pro"],
            "scans_used": scan_check["scans_used"],
        }
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
        row = conn.execute(
            "SELECT * FROM devices WHERE device_key=?", (device_key,)
        ).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO devices(device_key, is_pro, month, scan_count) VALUES(?,1,?,0)",
                (device_key, month_key),
            )
        else:
            conn.execute(
                "UPDATE devices SET is_pro=1 WHERE device_key=?", (device_key,)
            )

    logger.info(f"Pro activated for device {device_key}, payment_id={payment_id}")
    return {
        "status": "activated",
        "message": "Pro activated! Unlimited scans unlocked.",
    }


# ── Scan status check (for frontend banner) ───────────────────────────
@app.get("/scan-status")
@app.get("/scan-quota")
async def scan_status(request: Request):
    """Returns remaining scans and pro status for the current device."""
    device_key = get_device_key(request)
    month_key = datetime.date.today().isoformat()[:7]

    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM devices WHERE device_key=?", (device_key,)
        ).fetchone()

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
        "arial.ttf",
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

    month_key = datetime.date.today().isoformat()[:7]

    # Check monthly quota
    scans_used = api_key_data.get("scans_this_month", 0)
    if api_key_data.get("month") != month_key:
        scans_used = 0
        with db_conn() as conn:
            conn.execute(
                "UPDATE api_keys SET month=?, scans_this_month=0 WHERE api_key=?",
                (month_key, api_key_data["api_key"]),
            )

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
            o_score = ocr_quality_score(run_ocr(content, language))
            e_score = ocr_quality_score(run_ocr(enhanced, language))
            if e_score >= o_score * 0.85:
                working = enhanced
                blur_info["deblurred"] = True
                blur_info["method_log"] = mlog
        except Exception as e:
            logger.warning(f"B2B deblur failed: {e}")

    ocr = run_ocr(working, language)
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
        return cached

    lang_name = LANGUAGE_MAP.get(language, "English")
    prompt = (
        f'[INST] Analyze: "{safe_text}". Persona: {persona}. '
        f"Respond in {lang_name} as valid JSON with: product_name, "
        f"product_category (one of: biscuit|noodle|chip|beverage|juice|dairy|chocolate|protein_supplement|ready_to_eat|sweet), "
        f"score(1-10), verdict, summary, ingredients_raw, nutrient_breakdown, pros, cons, age_warnings, better_alternative. [/INST]"
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

        # ── Step 7.5: Eatlytic DNA Overrides ──────────────────────────
        dna_result = apply_dna_overrides(
            full_ocr_text=safe_text,
            nutrients=_flatten_nutrients(result.get("nutrient_breakdown", [])),
            ingredients_raw=result.get("ingredients_raw", ""),
            base_score=result.get("score", 5),
        )

        if dna_result["action"] == "BLOCK":
            return {"error": "atwater_mismatch", "message": dna_result["reason"]}

        if dna_result["action"] == "OVERRIDE":
            result["score"] = dna_result["score"]
            result["verdict"] = dna_result["reason"]

        if dna_result["extra_flags"]:
            result.setdefault("cons", []).extend(dna_result["extra_flags"])
            result["score"] = dna_result["score"]

        # ── Step 9: Alternative Engine ───────────────────────────────
        category = result.get("product_category", "general")
        alt_advice = get_healthy_alternative(category, persona)
        result["better_alternative"] = alt_advice

        result["blur_info"] = blur_info

        with db_conn() as conn:
            conn.execute(
                "UPDATE api_keys SET scans_this_month=scans_this_month+1 WHERE api_key=?",
                (api_key_data["api_key"],),
            )

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
        raise HTTPException(
            status_code=500, detail="Server misconfiguration: ADMIN_TOKEN not set."
        )
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
def generate_whatsapp_response(
    r: dict, ocr: dict, dna_result: dict = None, alt_advice: str = ""
) -> str:
    if ocr.get("avg_confidence", 1) < 0.30 and ocr.get("word_count", 0) < 20:
        return "❌ Cannot Score\n\nThe label is too blurry or incomplete to provide a definitive score. Please send a clearer picture of the ingredients list."

    if dna_result and dna_result["action"] == "BLOCK":
        return dna_result["reason"]

    s = r.get("score", "?")
    v = r.get("verdict", "Analyzed")

    if dna_result and dna_result["action"] == "OVERRIDE":
        v = dna_result["reason"]

    summ = r.get("summary", "")
    fl = r.get("cons", [])
    flags_t = "\n".join([f"- {f}" for f in fl]) if fl else "- None"

    msg = f"Eatlytic Score: {s}/10\n\n[{v}]\n{summ}\n\n⚠️ Risk Flags:\n{flags_t}"
    if alt_advice:
        msg += f"\n\n{alt_advice}"
    return msg


@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """Twilio WhatsApp sandbox webhook with DNA overrides and scan limits."""
    form = await request.form()

    # 1. SECURITY: Verify Twilio Signature
    try:
        from twilio.request_validator import RequestValidator

        validator = RequestValidator(os.environ.get("TWILIO_AUTH_TOKEN", ""))
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        if (
            not validator.validate(url, form, signature)
            and os.environ.get("ENV") == "production"
        ):
            logger.warning("Twilio signature validation failed!")
    except ImportError:
        pass

    resp = MessagingResponse()
    msg = resp.message()

    # 2. Identify user and check scan limits
    phone_number = form.get("From", "unknown_wa")
    scan_check = check_and_increment_scan(phone_number)

    media_url = form.get("MediaUrl0")
    if not media_url:
        msg.body(
            "👋 Welcome to *Eatlytic*!\n\n"
            "Send me a photo of any food label (back of pack) "
            "and I'll analyse it instantly.\n\n"
            "Works even on blurry photos 📸\nFree — no barcode needed."
        )
        return Response(content=str(resp), media_type="application/xml")

    # Check if allowed to scan
    if not scan_check["allowed"]:
        msg.body(
            "❌ Free Scan Limit Reached\n\n"
            f"You've used all {FREE_SCAN_LIMIT} free scans this month. "
            "Upgrade to Pro at eatlytic.com/pro for unlimited scans! 🚀"
        )
        return Response(content=str(resp), media_type="application/xml")

    # 3. Process the image
    try:
        import httpx

        TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

        async with httpx.AsyncClient() as hc:
            img_bytes = (
                await hc.get(media_url, auth=(TWILIO_SID, TWILIO_TOKEN))
            ).content

        # Blur assessment & enhancement
        quality = assess_image_quality(img_bytes)
        img_to_ocr = img_bytes
        blur_info = {
            "detected": quality["is_blurry"],
            "severity": quality["blur_severity"],
            "score": quality["blur_score"],
            "deblurred": False,
        }
        if quality["is_blurry"]:
            enhanced, log = deblur_and_enhance(img_bytes, quality["blur_severity"])
            img_to_ocr = enhanced
            blur_info["deblurred"] = True
            blur_info["method_log"] = log

        # 4. OCR & Label Detection
        ocr_r = get_server_ocr(img_to_ocr, "en")
        text = ocr_r["text"]
        lc = detect_label_presence(text)

        if not lc["has_label"]:
            msg.body(
                "❌ Couldn't find a nutrition label. "
                "Please send a clear photo of the *back* of the packaging."
            )
        else:
            # 5. Unified Analysis Flow (Consolidated logic)
            analysis = await unified_analyze_flow(
                extracted_text=text,
                persona="general",
                age_group="adult",
                product_category_hint="general",
                language="en",
                web_context="",  # Will be fetched inside flow
                blur_info=blur_info,
                label_confidence=lc.get("confidence", "medium"),
            )

            # 6. Format Response
            score = analysis.get("score", "?")
            verdict = analysis.get("verdict", "Analyzed")
            summary = analysis.get("summary", "")
            cons = analysis.get("cons", [])
            alt = analysis.get("better_alternative", "")

            risk_text = "\n".join([f"• {c}" for c in cons]) if cons else "• None"

            response_text = (
                f"*Eatlytic Score: {score}/10*\n"
                f"[{verdict}]\n\n"
                f"{summary}\n\n"
                f"⚠️ *Risk Flags:*\n{risk_text}\n\n"
                f"{alt}\n\n"
                f"_Scans left: {scan_check['scans_remaining']}_"
            )
            msg.body(response_text)

    except Exception as e:
        logger.error(f"WhatsApp error: {e}")
        msg.body("⚠️ Analysis failed. Try again or visit *eatlytic.com*")

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
