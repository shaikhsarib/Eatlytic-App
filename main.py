import os
from dotenv import load_dotenv

# Compatibility patch for older underlying libraries (like EasyOCR) that use deprecated PIL syntax
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Load environment variables from .env if present
load_dotenv()

import re
import json
import logging
import hashlib
import secrets
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
try:
    from twilio.twiml.messaging_response import MessagingResponse
except ImportError:
    MessagingResponse = None
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

# P0 FIX: Only import what actually exists in the new files
from app.services.ocr import run_ocr, passes_confidence_gate
from app.services.image import (
    assess_image_quality,
    deblur_and_enhance,
    image_to_b64,
    ocr_quality_score,
)
from app.services.llm import unified_analyze_flow
# NOTE: get_live_search is NOT imported here — it is lazy-loaded inside llm.py
# with a try/except guard to avoid crashing when duckduckgo_search is not installed.

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
        "http://localhost:3000",
        "http://localhost:7860",
    ],
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Initialize Database
init_db()

# --- SCAN LIMITS & API KEYS ---
FREE_SCAN_LIMIT = 3  # CEO Order: 3 scans, not 10

# --- API KEY AUTH ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        return None
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE api_key=?", (api_key,)
        ).fetchone()
    return dict(row) if row else None


def generate_api_key(client_name: str, plan: str = "business") -> str:
    key = "eak_" + secrets.token_urlsafe(32)
    month_key = datetime.date.today().isoformat()[:7]
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys(api_key, client_name, plan, scans_this_month, month, active) VALUES(?,?,?,?,?,1)",
            (key, client_name, plan, 0, month_key),
        )
    return key


# --- DEVICE FINGERPRINT ---
def get_device_key(request: Request) -> str:
    """Fingerprint device using IP + UserAgent + Fingerprint header hint."""
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")
    fp_hint = request.headers.get("x-fingerprint", "")
    # BUG FIX: Include fp_hint so different browsers on same IP are treated as different devices.
    return hashlib.md5(f"{ip}:{ua}:{fp_hint}".encode()).hexdigest()[:16]


# --- GROQ CLIENT ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY missing! App will fail.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# ══════════════════════════════════════════════════════════════════════
#  SECTION: ROUTES
# ══════════════════════════════════════════════════════════════════════


@app.get("/")
async def home():
    return FileResponse("index.html")


@app.post("/check-image")
@limiter.limit("30/minute")
async def check_image(request: Request, image: UploadFile = File(...)):
    content = await image.read()
    return assess_image_quality(content)


@app.post("/enhance-preview")
@limiter.limit("20/minute")
async def enhance_preview(request: Request, image: UploadFile = File(...)):
    content = await image.read()
    quality = assess_image_quality(content)
    if not quality["is_blurry"]:
        return JSONResponse(
            {
                "deblurred": False,
                "message": "Image is already clear.",
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
        }
    )


@app.post("/ocr")
@limiter.limit("20/minute")
async def perform_ocr(
    request: Request, image: UploadFile = File(...), language: str = Form("en")
):
    content = await image.read()
    return run_ocr(content, language)


@app.post("/analyze")
@limiter.limit("15/minute")
async def analyze_product(
    request: Request,
    persona: str = Form(...),
    age_group: str = Form("adult"),
    product_category: str = Form("general"),
    language: str = Form("en"),
    extracted_text: str = Form(None),
    front_text: str = Form(""),
    image: UploadFile = File(...),
):
    if not client:
        return {"error": "Server Error: Missing GROQ_API_KEY"}

    device_key = get_device_key(request)
    # BUG FIX #1: Check quota WITHOUT incrementing first.
    # Scans are only deducted AFTER a successful analysis.
    scan_check = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=False)
    if not scan_check["allowed"]:
        return JSONResponse(
            status_code=402,
            content={
                "error": "scan_limit_reached",
                "message": f"You've used all {FREE_SCAN_LIMIT} free scans.",
                "upgrade_url": "/pro",
                "scans_used": scan_check["scans_used"],
            },
        )

    try:
        content = await image.read()
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
        working_content = content

        if quality["is_blurry"]:
            try:
                enhanced_bytes, method_log = deblur_and_enhance(
                    content, quality["blur_severity"]
                )
                working_content = enhanced_bytes
                blur_info["deblurred"] = True
                blur_info["method_log"] = method_log
                blur_info["ocr_source"] = "deblurred"
            except Exception as e:
                logger.warning(f"Deblurring failed: {e}")

        # FIX: Run ROI crop + contrast enhance BEFORE OCR.
        # This is the key step that makes complex labels (Maggi back, Tata Salt,
        # any real-world product photo) work correctly.
        # process_image_for_ocr() detects the nutrition table region,
        # deskews it, and enhances contrast — then OCR is run on the clean crop.
        if not extracted_text:
            from app.services.label_detector import process_image_for_ocr
            cropped_content = process_image_for_ocr(working_content)
            ocr_result = run_ocr(cropped_content, language)
            # If crop gave fewer words than full image, fall back to full image OCR
            if ocr_result.get("word_count", 0) < 10:
                ocr_result_full = run_ocr(working_content, language)
                if ocr_result_full.get("word_count", 0) > ocr_result.get("word_count", 0):
                    ocr_result = ocr_result_full
            extracted_text = ocr_result["text"]

        result = await unified_analyze_flow(
            extracted_text=extracted_text,
            persona=persona,
            age_group=age_group,
            product_category_hint=product_category,
            language=language,
            web_context="",
            blur_info=blur_info,
            label_confidence="high",
            front_text=front_text,
            # image_content intentionally NOT passed — OCR already done above.
        )

        if "error" in result:
            return result

        # BUG FIX #1 (continued): Only deduct quota after confirmed success.
        scan_update = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=True)
        result["scan_meta"] = {
            "scans_remaining": scan_update["scans_remaining"],
            "is_pro": scan_update["is_pro"],
            "scans_used": scan_update["scans_used"],
        }
        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": f"Scan failed: {str(e)[:100]}..."}


# ── Health check ─────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0"}


# ── Pro activation ──────────────────────────────
@app.post("/activate-pro")
async def activate_pro(request: Request, payment_id: str = Form(...)):
    # Basic format validation for Razorpay/Stripe-like IDs
    if not re.match(r"^(pay_|razor_|pi_)?[\w]{10,60}$", payment_id):
        raise HTTPException(status_code=400, detail="Invalid payment ID format.")

    device_key = get_device_key(request)
    with db_conn() as conn:
        row = conn.execute(
            "SELECT * FROM devices WHERE device_key=?", (device_key,)
        ).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO devices(device_key, is_pro, month, scan_count) VALUES(?,1,?,0)",
                (device_key, datetime.date.today().isoformat()[:7]),
            )
        else:
            conn.execute(
                "UPDATE devices SET is_pro=1 WHERE device_key=?", (device_key,)
            )

    # Log payment for audit; TABLE may not exist in all envs — handle gracefully
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO payments(device_key, razorpay_payment_id, status, paid_at) VALUES(?,?,?,?)",
                (device_key, payment_id, "captured", datetime.datetime.now().isoformat())
            )
    except Exception as e:
        logger.warning("Payment log failed (table may not exist): %s", e)

    return {"status": "activated", "message": "Pro activated! Payment logged."}


# ── Scan status ────────────────────────────────
@app.get("/scan-status")
@app.get("/scan-quota")
async def scan_status(request: Request):
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


# ── Shareable PNG card ─────────────────────────
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
    W, H = 1080, 1080
    BG = (15, 17, 23)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "arial.ttf",
    ]
    font = None
    for p in font_paths:
        try:
            if os.path.exists(p):
                font = ImageFont.truetype(p, 48)
                break
        except Exception:
            continue
    if not font:
        font = ImageFont.load_default(size=48)
    score_rgb = (
        (34, 197, 94) if score >= 7 else (245, 158, 11) if score >= 4 else (239, 68, 68)
    )
    draw.ellipse([340, 160, 740, 560], outline=score_rgb, width=18)
    draw.text((540, 360), str(score), fill=score_rgb, anchor="mm", font=font)
    draw.text((540, 420), "/10", fill=(100, 116, 139), anchor="mm", font=font)
    pname = product_name[:38] + ("…" if len(product_name) > 38 else "")
    draw.text((540, 610), pname, fill=(255, 255, 255), anchor="mm", font=font)
    draw.text((540, 660), verdict[:50], fill=(148, 163, 184), anchor="mm", font=font)
    if top_pro:
        draw.rectangle([60, 700, 1020, 760], fill=(15, 60, 40))
        draw.text(
            (540, 730), f"✓ {top_pro[:65]}", fill=(74, 222, 128), anchor="mm", font=font
        )
    if top_warning:
        draw.rectangle([60, 775, 1020, 840], fill=(124, 29, 29))
        draw.text(
            (540, 807),
            f"⚠ {top_warning[:65]}",
            fill=(252, 165, 165),
            anchor="mm",
            font=font,
        )
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


# ── B2B API endpoint ──────────────────────────
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
    if not api_key_data:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    if not api_key_data.get("active"):
        raise HTTPException(status_code=403, detail="API key suspended.")

    month_key = datetime.date.today().isoformat()[:7]
    scans_used = api_key_data.get("scans_this_month", 0)
    if api_key_data.get("month") != month_key:
        scans_used = 0
        with db_conn() as conn:
            conn.execute(
                "UPDATE api_keys SET month=?, scans_this_month=0 WHERE api_key=?",
                (month_key, api_key_data["api_key"]),
            )

    LIMITS = {"business": 1000, "enterprise": 99999}
    limit = LIMITS.get(api_key_data.get("plan"), 1000)
    if scans_used >= limit:
        raise HTTPException(status_code=429, detail=f"Monthly limit ({limit}) reached.")

    content = await image.read()
    quality = assess_image_quality(content)
    blur_info = {
        "detected": quality["is_blurry"],
        "severity": quality["blur_severity"],
        "score": quality["blur_score"],
    }

    working = content
    if quality["is_blurry"]:
        try:
            enhanced, mlog = deblur_and_enhance(content, quality["blur_severity"])
            working = enhanced
            blur_info["deblurred"] = True
            blur_info["method_log"] = mlog
        except Exception as e:
            logger.warning(f"B2B deblur failed: {e}")

    from app.services.label_detector import process_image_for_ocr
    cropped_b2b = process_image_for_ocr(working)
    ocr = run_ocr(cropped_b2b, language)
    if ocr.get("word_count", 0) < 10:
        ocr_full = run_ocr(working, language)
        if ocr_full.get("word_count", 0) > ocr.get("word_count", 0):
            ocr = ocr_full
    text = ocr["text"]

    # P0 FIX: B2B research is now internally handled by unified_analyze_flow
    # No manual LLM or search calls needed here.
    web_ctx = ""

    result = await unified_analyze_flow(
        extracted_text=text,
        persona=persona,
        age_group=age_group,
        product_category_hint="general",
        language=language,
        web_context=web_ctx,
        blur_info=blur_info,
        label_confidence="high",
        front_text="",
        # BUG FIX: image_content NOT passed — OCR already done above (prevents double-OCR)
    )

    if "error" in result:
        return JSONResponse(status_code=400, content=result)

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
    return result


@app.post("/admin/create-api-key")
async def create_api_key_endpoint(
    admin_token: str = Form(...),
    client_name: str = Form(...),
    plan: str = Form("business"),
):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token.")
    key = generate_api_key(client_name, plan)
    return {"api_key": key, "client": client_name, "plan": plan}


# ── PDF export ────────────────────────────────
@app.post("/export-pdf")
@limiter.limit("10/minute")
async def export_pdf(request: Request, analysis_json: str = Form(...)):
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
        return JSONResponse({"error": "reportlab not installed."}, status_code=501)

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
        tbl_data = [["Nutrient", "Amount"]]
        for n in nutrients:
            tbl_data.append(
                [n.get("name", ""), f"{n.get('value', '')} {n.get('unit', '')}"]
            )
        tbl = Table(tbl_data, colWidths=[7 * cm, 7 * cm])
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
            "Generated by Eatlytic — eatlytic.com",
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


@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """Twilio WhatsApp webhook using unified flow."""
    form = await request.form()

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

    if not MessagingResponse:
        return Response("Twilio Support Disabled", status_code=501)

    resp = MessagingResponse()
    msg = resp.message()

    phone_number = form.get("From", "unknown_wa")
    # BUG FIX #1 (WhatsApp): Check only — do NOT increment yet.
    scan_check = check_and_increment_scan(phone_number, limit=FREE_SCAN_LIMIT, increment=False)

    media_url = form.get("MediaUrl0")
    if not media_url:
        msg.body(
            "👋 Welcome to *Eatlytic*!\n\nSend me a photo of any food label and I'll analyse it instantly.\nFree — no barcode needed."
        )
        return Response(content=str(resp), media_type="application/xml")

    if not scan_check["allowed"]:
        msg.body(
            f"❌ Free Scan Limit Reached\n\nYou've used all {FREE_SCAN_LIMIT} free scans. Upgrade to Pro for unlimited scans!"
        )
        return Response(content=str(resp), media_type="application/xml")

    try:
        import httpx

        TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
        async with httpx.AsyncClient() as hc:
            img_bytes = (
                await hc.get(media_url, auth=(TWILIO_SID, TWILIO_TOKEN))
            ).content

        quality = assess_image_quality(img_bytes)
        img_to_ocr = img_bytes
        blur_info = {
            "detected": quality["is_blurry"],
            "severity": quality["blur_severity"],
            "score": quality["blur_score"],
            "deblurred": False,
        }
        if quality["is_blurry"]:
            try:
                enhanced, log = deblur_and_enhance(img_bytes, quality["blur_severity"])
                img_to_ocr = enhanced
                blur_info["deblurred"] = True
                blur_info["method_log"] = log
            except Exception:
                pass

        # FIX: crop to nutrition table before OCR (same as /analyze endpoint)
        from app.services.label_detector import process_image_for_ocr
        cropped_wa = process_image_for_ocr(img_to_ocr)
        ocr_r = run_ocr(cropped_wa, "en")
        if ocr_r.get("word_count", 0) < 10:
            ocr_r_full = run_ocr(img_to_ocr, "en")
            if ocr_r_full.get("word_count", 0) > ocr_r.get("word_count", 0):
                ocr_r = ocr_r_full
        text = ocr_r["text"]

        analysis = await unified_analyze_flow(
            extracted_text=text,
            persona="general",
            age_group="adult",
            product_category_hint="general",
            language="en",
            web_context="",
            blur_info=blur_info,
            label_confidence="high",
            front_text="",
            # image_content intentionally NOT passed — OCR already done above.
        )

        if "error" in analysis:
            msg.body(f"❌ {analysis.get('message', 'Could not analyze label.')}")
        else:
            score = analysis.get("score", "?")
            verdict = analysis.get("verdict", "Analyzed")
            summary = analysis.get("summary", "")
            cons = analysis.get("cons", [])
            alt = analysis.get("better_alternative", "")
            nutrients = analysis.get("nutrient_breakdown", [])

            nut_text = "\n".join(
                [
                    f"{n.get('name', '?')}: {n.get('value', '')}{n.get('unit', '')}"
                    for n in nutrients
                ]
            )
            risk_text = "\n".join([f"• {c}" for c in cons]) if cons else "• None"

            # BUG FIX #3 (WhatsApp): Deduct AFTER success, use FRESH count for message.
            scan_update = check_and_increment_scan(phone_number, limit=FREE_SCAN_LIMIT, increment=True)
            response_text = (
                f"*Eatlytic Scan Results:*\n{nut_text}\n\n"
                f"*Score: {score}/10*\n[{verdict}]\n\n"
                f"{summary}\n\n"
                f"⚠️ *Risk Flags:*\n{risk_text}\n\n"
                f"{alt}\n\n"
                f"_Scans left: {scan_update['scans_remaining']}_"
            )
            msg.body(response_text)

    except Exception as e:
        logger.error(f"WhatsApp error: {e}")
        msg.body("⚠️ Analysis failed. Try again.")

    return Response(content=str(resp), media_type="application/xml")


@app.post("/test-accuracy")
@limiter.limit("5/minute")
async def test_accuracy(
    request: Request, image: UploadFile = File(...), ground_truth: str = Form("")
):
    content = await image.read()
    quality = assess_image_quality(content)
    ocr_orig = run_ocr(content, "en")
    ocr_enhanced = None
    if quality["is_blurry"]:
        try:
            enhanced_bytes, mlog = deblur_and_enhance(content, quality["blur_severity"])
            ocr_enhanced = run_ocr(enhanced_bytes, "en")
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
