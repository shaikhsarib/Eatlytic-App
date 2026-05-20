from fastapi import APIRouter, Request, Response, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import os
import datetime
from app.services.image import assess_image_quality, deblur_and_enhance, image_to_b64
from app.services.ocr import run_ocr
from app.services.llm import unified_analyze_flow
from app.services.hash_service import get_image_fingerprint
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
        "blur_severity": float(quality["blur_severity"])
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
    
    # Sanitize inputs
    persona = sanitize_text(persona, 100)
    product_category = sanitize_text(product_category, 100)
    front_text = sanitize_text(front_text, 1000)
    extracted_text = sanitize_text(extracted_text, 10000) if extracted_text else None

    # Check quota
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
            from app.services.label_detector import process_image_for_ocr
            cropped_content = process_image_for_ocr(working_content)
            ocr_result = run_ocr(cropped_content, language)
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
        scan_update = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=True)
        result["scan_meta"] = {
            "scans_remaining": scan_update["scans_remaining"],
            "is_pro": scan_update["is_pro"],
            "scans_used": scan_update["scans_used"],
        }
        
        if image_hash:
            set_image_fingerprint(image_hash, result)

        # Save scan
        scan_id = save_scan(device_key, result)
        result["scan_id"] = scan_id

        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": f"Scan failed: {str(e)[:100]}..."}


@router.post("/parse-voice-meal")
async def parse_voice_meal(
    request: Request,
    response: Response,
    text: str = Form(...),
    persona: str = Form("General Adult"),
    language: str = Form("en"),
):
    """Parse a free-text voice meal description into a nutrition estimate."""
    device_key = get_device_key(request, response)

    scan_check = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=False)
    if not scan_check["allowed"]:
        return JSONResponse(
            status_code=402,
            content={
                "error": "scan_limit_reached",
                "message": f"You've used all {FREE_SCAN_LIMIT} free scans.",
                "upgrade_url": "/pro",
            },
        )

    try:
        from app.services.llm.client import call_llm, parse_llm_response
        prompt = (
            f"The user said they ate: \"{sanitize_text(text, 300)}\"\n"
            f"Persona: {sanitize_text(persona, 80)}\n\n"
            "Estimate the nutrition facts for this meal as a JSON object with the same schema "
            "as a standard food label analysis. Include: product_name, score (1-10), verdict, "
            "summary, nutrient_breakdown (array of {name, value, unit, impact, rating}), "
            "pros (array), cons (array), ingredients_spotlight (array), "
            "safety_tier (Safe/Limit/Avoid), safety_verdict, safety_reason.\n"
            "Return ONLY valid JSON, no markdown."
        )
        import asyncio
        raw = await asyncio.to_thread(call_llm, prompt, 1500)
        result = parse_llm_response(raw)
        if not result or "error" in result:
            raise ValueError("LLM returned no usable data")

        result.setdefault("is_voice_log", True)
        result.setdefault("product_name", text[:60])

        scan_update = check_and_increment_scan(device_key, limit=FREE_SCAN_LIMIT, increment=True)
        result["scan_meta"] = {
            "scans_remaining": scan_update["scans_remaining"],
            "is_pro": scan_update["is_pro"],
        }
        scan_id = save_scan(device_key, result)
        result["scan_id"] = scan_id
        return result

    except Exception as e:
        logger.error(f"parse_voice_meal error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "voice_parse_failed", "message": "Could not parse voice meal."},
        )


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
