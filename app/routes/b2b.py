from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException, Security
from fastapi.responses import JSONResponse
import logging
import datetime
from app.services.image import assess_image_quality
from app.services.ocr import run_ocr
from app.services.llm import unified_analyze_flow
from app.models.db import verify_api_key, db_conn

router = APIRouter(prefix="/api/v1", tags=["b2b"])

@router.post("/analyze")
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
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "client_ref": api_key_data["client_name"]
    }

    with db_conn() as conn:
        conn.execute("UPDATE api_keys SET scans_this_month=scans_this_month+1 WHERE api_key=?", (api_key_data["api_key"],))

    return result
