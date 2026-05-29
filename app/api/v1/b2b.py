from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException, Security
from fastapi.responses import JSONResponse
import logging
import datetime
_UTC = datetime.timezone.utc
from app.services.image import assess_image_quality
from app.ai.ocr.client import run_ocr
from app.ai.llm import unified_analyze_flow
from app.database.connection import db_conn
from app.services.b2b_auth import get_b2b_client

router = APIRouter(prefix="/api/v1", tags=["b2b"])

@router.post("/analyze")
async def api_analyze(
    request: Request,
    image: UploadFile = File(...),
    language: str = Form("en"),
    persona: str = Form("general adult"),
    age_group: str = Form("adult"),
    api_key_data: dict = Security(get_b2b_client),
):
    if not api_key_data.get("active"):
        raise HTTPException(status_code=403, detail="API key suspended.")

    month_key = datetime.date.today().isoformat()[:7]
    scans_used = api_key_data.get("scans_this_month", 0)
    if api_key_data.get("month") != month_key:
        scans_used = 0
        with db_conn() as conn:
            conn.execute("UPDATE api_keys SET month=?, scans_this_month=0 WHERE api_key=?", (month_key, api_key_data["api_key"]))

    LIMITS = {"business": 1000, "enterprise": 99999}
    limit = LIMITS.get(api_key_data.get("plan"), 1000)
    if scans_used >= limit:
        raise HTTPException(status_code=429, detail=f"Monthly limit ({limit}) reached.")

    content = await image.read()
    quality = assess_image_quality(content)
    
    from app.services.label_detector import process_image_for_ocr
    cropped = process_image_for_ocr(content)
    ocr = run_ocr(cropped, language)
    
    # Enforce the OCR confidence gate to block extremely blurry/unreadable images
    from app.ai.ocr.client import passes_confidence_gate
    passed, err_msg = passes_confidence_gate(ocr)
    if not passed:
        return JSONResponse(
            status_code=400,
            content={
                "error": "blurry_image",
                "message": err_msg
            }
        )
    
    result = await unified_analyze_flow(
        extracted_text=ocr["text"],
        persona=persona,
        age_group=age_group,
        product_category_hint="general",
        language=language,
        web_context="",
        blur_info={"detected": quality["is_blurry"], "severity": quality["blur_severity"]},
        label_confidence="high",
        front_text="",
        image_content=content if ocr.get("word_count", 0) < 20 else None
    )

    if "error" in result:
        return JSONResponse(status_code=400, content=result)

    result["audit"] = {
        "atwater_compliance": result.get("score", 0) > 3,
        "nova_classification": result.get("analysis_json", {}).get("nova_group", "Unknown"),
        "timestamp": datetime.datetime.now(_UTC).replace(tzinfo=None).isoformat(),
        "client_ref": api_key_data["client_name"]
    }

    # Increment usage counter upon successful scan completion
    from app.database.connection import increment_api_scan
    increment_api_scan(api_key_data["api_key"])
    return result


@router.get("/usage")
async def api_usage(
    api_key_data: dict = Security(get_b2b_client),
):
    """
    Returns monthly usage telemetry for the authenticated B2B client.
    Powers their developer portal and telemetry dashboards.
    """
    LIMITS = {"business": 1000, "enterprise": 99999}
    limit = LIMITS.get(api_key_data.get("plan"), 1000)
    scans_used = api_key_data.get("scans_this_month", 0)
    
    return {
        "client_name": api_key_data["client_name"],
        "plan": api_key_data["plan"],
        "scans_used_this_month": scans_used,
        "scans_limit": limit,
        "scans_remaining": max(0, limit - scans_used),
        "month": api_key_data.get("month", datetime.date.today().isoformat()[:7]),
        "active": bool(api_key_data.get("active", 1)),
    }
