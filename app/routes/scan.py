from fastapi import APIRouter, Request, Response, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import os
import datetime
from app.services.image import assess_image_quality, deblur_and_enhance, image_to_b64
from app.services.ocr import run_ocr
from app.services.llm import unified_analyze_flow
from app.services.hash_service import get_image_fingerprint
from app.services.label_detector import process_image_for_ocr
from app.models.db import (
    check_and_increment_scan,
    save_scan,
    get_image_fingerprint_match,
    set_image_fingerprint,
)
from app.utils import get_device_key, sanitize_text

logger = logging.getLogger(__name__)
router = APIRouter(tags=["scanning"])

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
FREE_SCAN_LIMIT = int(os.environ.get("FREE_SCAN_LIMIT", "10"))

@router.post("/check-image")
async def check_image(request: Request, image: UploadFile = File(...)):
    if image.size and image.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (Max 10MB)")
    content = await image.read()
    return assess_image_quality(content)

@router.post("/enhance-preview")
async def enhance_preview(request: Request, image: UploadFile = File(...)):
    if image.size and image.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (Max 10MB)")
    content = await image.read()
    quality = assess_image_quality(content)

    def _make_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_make_serializable(x) for x in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    quality = _make_serializable(quality)
    if not quality.get("is_blurry"):
        return JSONResponse({"deblurred": False, "message": "Image is already clear.", "quality": quality})
    
    enhanced_bytes, method_log = deblur_and_enhance(content, quality["blur_severity"])
    b64 = image_to_b64(enhanced_bytes)
    return JSONResponse({
        "deblurred": True,
        "image_b64": b64,
        "method_log": method_log,
        "blur_severity": quality["blur_severity"]
    })

@router.post("/analyze")
async def analyze_product(
    request: Request,
    response: Response,
    persona: str = Form(...),
    age_group: str = Form("adult"),
    product_category: str = Form("general"),
    language: str = Form("en"),
    extracted_text: str = Form(None),
    front_text: str = Form(""),
    image: UploadFile = File(...),
):
    device_key = get_device_key(request, response)
    
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
    if not token:
        token = request.cookies.get("session_token")
    from app.services.user_auth import get_user_from_token
    user = get_user_from_token(token) if token else None
    user_id = user["id"] if user else None
    
    # Sanitize inputs
    persona = sanitize_text(persona, 100)
    product_category = sanitize_text(product_category, 100)
    front_text = sanitize_text(front_text, 1000)
    extracted_text = sanitize_text(extracted_text, 10000) if extracted_text else None

    # Check quota
    if user_id:
        from app.services.user_auth import check_and_increment_scan_user
        scan_check = check_and_increment_scan_user(user_id, increment=False)
    else:
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

    if image.size and image.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (Max 10MB)")

    try:
        content = await image.read()
        quality = assess_image_quality(content)
        blur_info = {
            "detected": quality["is_blurry"],
            "severity": quality["blur_severity"],
            "score": quality["blur_score"],
            "deblurred": False,
            "ocr_source": "original",
        }
        working_content = content

        if quality["should_enhance"]:
            try:
                enhanced_bytes, method_log = deblur_and_enhance(content, quality["blur_severity"])
                working_content = enhanced_bytes
                blur_info["deblurred"] = True
                blur_info["ocr_source"] = "enhanced"
            except Exception as e:
                logger.warning(f"Enhancement failed: {e}")

        # pHash Cache
        image_hash = get_image_fingerprint(working_content)
        if image_hash:
            cached_result = get_image_fingerprint_match(image_hash)
            if cached_result and "error" not in cached_result:
                if user_id:
                    from app.services.user_auth import check_and_increment_scan_user
                    scan_update = check_and_increment_scan_user(user_id, increment=True)
                else:
                    scan_update = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=True)
                cached_result["scan_meta"] = {
                    "scans_remaining": scan_update["scans_remaining"],
                    "is_pro": scan_update["is_pro"],
                    "scans_used": scan_update["scans_used"],
                    "cached": True,
                }
                return cached_result

        # OCR & Analysis
        if not extracted_text:
            cropped_content = process_image_for_ocr(working_content)
            ocr_result = run_ocr(cropped_content, language)
            
            # Enforce the OCR confidence gate to block extremely blurry/unreadable images
            from app.services.ocr import passes_confidence_gate
            passed, err_msg = passes_confidence_gate(ocr_result)
            if not passed:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "blurry_image",
                        "message": err_msg
                    }
                )
            
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
            image_content=working_content,
        )

        if "error" in result:
            return result

        # Deduct quota
        if user_id:
            from app.services.user_auth import check_and_increment_scan_user
            scan_update = check_and_increment_scan_user(user_id, increment=True)
        else:
            scan_update = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=True)
        result["scan_meta"] = {
            "scans_remaining": scan_update["scans_remaining"],
            "is_pro": scan_update["is_pro"],
            "scans_used": scan_update["scans_used"],
        }
        
        if image_hash:
            set_image_fingerprint(image_hash, result)

        # Save scan
        scan_id = save_scan(device_key, result, user_id=user_id)
        result["scan_id"] = scan_id

        # Update user streak if logged in
        if user_id:
            from app.services.user_auth import update_streak_user
            try:
                update_streak_user(user_id)
            except Exception as streak_err:
                logger.warning(f"Failed to update streak for {user_id}: {streak_err}")

        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": f"Scan failed: {str(e)[:100]}..."}


@router.post("/parse-voice-meal")
async def parse_voice_meal(
    request: Request,
    response: Response,
    text: str = Form(...),
    persona: str = Form(...),
    language: str = Form("en"),
):
    device_key = get_device_key(request, response)
    
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
    if not token:
        token = request.cookies.get("session_token")
    from app.services.user_auth import get_user_from_token
    user = get_user_from_token(token) if token else None
    user_id = user["id"] if user else None
    
    # Sanitize inputs
    persona = sanitize_text(persona, 100)
    text = sanitize_text(text, 10000)

    # Check quota
    if user_id:
        from app.services.user_auth import check_and_increment_scan_user
        scan_check = check_and_increment_scan_user(user_id, increment=False)
    else:
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
        result = await unified_analyze_flow(
            extracted_text=text,
            persona=persona,
            age_group="adult",
            product_category_hint="general",
            language=language,
            web_context="",
            blur_info={
                "detected": False,
                "severity": "none",
                "score": 100,
                "deblurred": False,
                "ocr_source": "voice",
            },
            label_confidence="high",
            front_text="",
            image_content=None,
        )

        if "error" in result:
            return result

        # Deduct quota
        if user_id:
            from app.services.user_auth import check_and_increment_scan_user
            scan_update = check_and_increment_scan_user(user_id, increment=True)
        else:
            scan_update = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=True)
        result["scan_meta"] = {
            "scans_remaining": scan_update["scans_remaining"],
            "is_pro": scan_update["is_pro"],
            "scans_used": scan_update["scans_used"],
        }

        # Save scan
        scan_id = save_scan(device_key, result, user_id=user_id)
        result["scan_id"] = scan_id

        # Update user streak if logged in
        if user_id:
            from app.services.user_auth import update_streak_user
            try:
                update_streak_user(user_id)
            except Exception as streak_err:
                logger.warning(f"Failed to update streak for {user_id}: {streak_err}")

        return result

    except Exception as e:
        logger.error(f"Voice Analysis error: {e}")
        return {"error": f"Voice Analysis failed: {str(e)[:100]}..."}






@router.post("/activate-pro")
async def activate_pro(request: Request, response: Response, payment_id: str = Form(...)):
    """Demo Pro activation endpoint (real flow uses /payments/verify)."""
    device_key = get_device_key(request, response)
    try:
        with __import__("app.models.db", fromlist=["db_conn"]).db_conn() as conn:
            conn.execute(
                "UPDATE devices SET is_pro=1 WHERE device_key=?", (device_key,)
            )
        return {"status": "activated", "message": "Pro plan activated!", "is_pro": True}
    except Exception as e:
        logger.error(f"activate_pro error: {e}")
        return JSONResponse(status_code=500, content={"error": "activation_failed"})


@router.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(None),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
):
    """
    Twilio WhatsApp Webhook: Receives photos from WhatsApp,
    runs OCR and Diabetic Care analysis, and replies via TwiML XML.
    """
    import httpx
    logger.info(f"Incoming WhatsApp message from: {From}")
    
    # 1. Base TwiML template helper
    def build_xml_response(msg_text: str) -> Response:
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{msg_text}</Message>
</Response>"""
        return Response(content=xml_content, media_type="application/xml")

    # 2. If no photo is sent, reply with instructions
    if not MediaUrl0:
        instructions = (
            "🩺 *Eatlytic Diabetic Safety Scanner* 🩸\n\n"
            "Please send a *clear photo* of any food product's nutrition or ingredient label to instantly audit its blood sugar safety and discover diabetic-safe swaps!"
        )
        return build_xml_response(instructions)

    # 3. Quota check or device enrollment by WhatsApp number
    # Standardize WhatsApp number as a device_key
    device_key = f"whatsapp_{From.replace('whatsapp:', '').strip()}"
    
    # Check quota
    from app.models.db import check_and_increment_scan, save_scan
    scan_check = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=False)
    if not scan_check["allowed"]:
        quota_err = (
            "🚨 *Free Scan Limit Reached*\n\n"
            f"You have used all your {FREE_SCAN_LIMIT} free scans.\n"
            "Please visit https://eatlytic.com/pro to upgrade to Pro for unlimited food scans!"
        )
        return build_xml_response(quota_err)

    # 4. Download media attachment
    try:
        async with httpx.AsyncClient() as client:
            account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
            auth = (account_sid, auth_token) if account_sid and auth_token else None
            img_resp = await client.get(MediaUrl0, auth=auth, follow_redirects=True)
            if img_resp.status_code != 200:
                return build_xml_response("❌ Failed to download your image from Twilio. Please try again.")
            content = img_resp.content
    except Exception as fetch_err:
        logger.error(f"Failed to fetch Twilio attachment: {fetch_err}")
        return build_xml_response("❌ Connection error while downloading the label photo.")

    # 5. Run Quality Assessment and Deblur
    try:
        quality = assess_image_quality(content)
        blur_info = {
            "detected": quality["is_blurry"],
            "severity": quality["blur_severity"],
            "score": quality["blur_score"],
            "deblurred": False,
            "ocr_source": "original",
        }
        working_content = content
        if quality["should_enhance"]:
            try:
                enhanced_bytes, method_log = deblur_and_enhance(content, quality["blur_severity"])
                working_content = enhanced_bytes
                blur_info["deblurred"] = True
                blur_info["ocr_source"] = "enhanced"
            except Exception as e:
                logger.warning(f"Enhancement failed: {e}")

        # 6. OCR Extraction
        cropped_content = process_image_for_ocr(working_content)
        ocr_result = run_ocr(cropped_content, "en")
        
        # Enforce confidence gate
        from app.services.ocr import passes_confidence_gate
        passed, err_msg = passes_confidence_gate(ocr_result)
        if not passed:
            return build_xml_response(f"⚠️ *Image Unreadable*\n\n{err_msg}")

        extracted_text = ocr_result["text"]

        # 7. Unified Analysis with Diabetic Care persona
        result = await unified_analyze_flow(
            extracted_text=extracted_text,
            persona="Diabetic Care",
            age_group="adult",
            product_category_hint="general",
            language="en",
            web_context="",
            blur_info=blur_info,
            label_confidence="high",
            front_text="",
            image_content=working_content,
        )

        if "error" in result:
            return build_xml_response(f"❌ Analysis failed: {result['error']}")

        # Deduct scan quota
        check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=True)

        # 8. Save scan to persistent history
        save_scan(device_key, result, user_id=None)

        # 9. Format response
        from app.services.formatter import format_whatsapp_tier1, format_whatsapp_tier2
        tier1 = format_whatsapp_tier1(result)
        tier2 = format_whatsapp_tier2(result)
        alternative = result.get("better_alternative", "")

        # Synthesize beautiful, rich WhatsApp reply
        reply = (
            f"{tier1}\n\n"
            f"💡 *Healthy Diabetic Swap:*\n{alternative}\n\n"
            f"🔍 *Nutrient & Ingredient Details:*\n{tier2}"
        )
        return build_xml_response(reply)

    except Exception as e:
        logger.error(f"WhatsApp webhook scan crash: {e}")
        return build_xml_response("❌ An unexpected error occurred while analyzing the food label.")
