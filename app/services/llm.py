"""
app/services/llm.py
The Stupid Intern (LLM) & The Boss (DNA Rules).
"""

import os
import json
import logging
import asyncio
import hashlib
from app.models.db import get_ai_cache, set_ai_cache
from app.services.fake_detector import apply_dna_overrides
from app.services.alternatives import get_healthy_alternative
from app.services.label_detector import process_image_for_ocr
from app.services.nutrition_parser import smart_parse
from app.services.explanation_engine import get_explanation_report
from app.services.formatter import get_whatsapp_tiered_content

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_groq_client = None
if GROQ_API_KEY:
    from groq import Groq

    _groq_client = Groq(api_key=GROQ_API_KEY)

MEDICAL_DISCLAIMER = (
    "⚕️ For informational purposes only — not medical advice. "
    "Consult a qualified nutritionist or physician before making dietary decisions."
)

LANGUAGE_MAP = {
    "en": "English",
    "zh": "Simplified Chinese",
    "es": "Spanish",
    "ar": "Arabic",
    "fr": "French",
    "hi": "Hindi (हिन्दी)",
    "pt": "Portuguese",
    "de": "German",
}


def call_llm(prompt: str, max_tokens: int = 2500) -> str:
    """Provider-agnostic LLM call."""
    if not _groq_client:
        logger.error("GROQ_API_KEY is not set in llm.py environment")
        raise RuntimeError("AI Configuration Error: Please check GROQ_API_KEY")

    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    last_err = None
    for model in models:
        try:
            comp = _groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return comp.choices[0].message.content
        except Exception as exc:
            logger.warning("Groq model %s failed: %s", model, exc)
            last_err = exc

    raise RuntimeError(f"All LLM models failed. Last error: {str(last_err)[:100]}")


STRICT_EXTRACTION_PROMPT = """
You are a strict data extraction bot for Indian food labels. 
Extract the data into EXACTLY this JSON format. 

CRITICAL RULES:
1. PRODUCT NAME: Extract the product name from the first few lines.
2. HIERARCHY (NEW): Distinguish between PRIMARY macros and SECONDARY sub-components.
   - Primary: 'protein', 'carbohydrate', 'fat'.
   - Secondary: 'sugar' (sub of carb), 'fiber' (sub of carb), 'saturated_fat' (sub of fat), 'sodium_mg'.
3. SINGLE INGREDIENTS: For pure products, 0g for everything except the main ingredient is normal.
4. ZERO CREATIVITY: Output EXACTLY the numbers printed. 
5. MISSING DATA: If a macro is not mentioned, output 0.
6. TABLE LOGIC: Only extract the "Per 100g" column. IGNORE the "Per Serve" or "% RDA" columns.
7. CALORIE PHYSICS: Ensure (Protein*4 + Carbs*4 + Fat*9) ≈ Calories within 25%.

{
  "product_name": "string",
  "product_category": "string",
  "calories": float,
  "protein": float,
  "carbs": float,
  "fat": float,
  "sugar": float,
  "sodium_mg": float,
  "fiber": float,
  "ingredients_raw": "string"
}
"""


def parse_llm_response(llm_output_string: str) -> dict:
    """Strip markdown code blocks if the LLM ignores instructions."""
    clean_string = llm_output_string.strip()
    if clean_string.startswith("```"):
        lines = clean_string.split("\n")
        if len(lines) >= 3:
            clean_string = "\n".join(lines[1:-1])
        else:
            clean_string = (
                clean_string.replace("```json", "").replace("```", "").strip()
            )
    return json.loads(clean_string)


async def unified_analyze_flow(
    extracted_text: str,
    persona: str,
    age_group: str,
    product_category_hint: str,
    language: str,
    web_context: str,
    blur_info: dict,
    label_confidence: str,
    front_text: str = "",
    image_content: bytes = None, # NEW
) -> dict:
    """The Master Pipeline."""

    # 1. Pipeline Start: Label Detection & Enhanced OCR (if image provided)
    if image_content:
        from app.services.ocr import run_ocr
        logger.info("Running Intelligent Image Pipeline (Label Detector)...")
        cropped_image = process_image_for_ocr(image_content)
        ocr_result = run_ocr(cropped_image, language)
        extracted_text = ocr_result["text"]

    # 2. Cache Check
    cache_key = hashlib.md5(
        f"{extracted_text[:100]}:{persona}:{language}".encode()
    ).hexdigest()
    cached = get_ai_cache(cache_key)
    if cached:
        cached["scan_meta"] = {"cached": True, "scans_remaining": 0, "is_pro": False, "scans_used": 0}
        return cached

    # 3. Label Filter
    from app.services.ocr import universal_label_filter
    filter_result = universal_label_filter(extracted_text)
    if not filter_result["is_valid"]:
        return {"error": "no_label", "message": "No nutrition table found."}
    clean_text = filter_result["clean_text"]

    # 4. LLM Extraction — always use LLM in the unified flow for accuracy.
    #    smart_parse (hybrid regex/LLM) is kept for lightweight callers only;
    #    here we need guaranteed LLM output so the retry path works correctly.
    async def _run_llm_extraction(text: str, extra_context: str = "") -> dict:
        prompt = f"{STRICT_EXTRACTION_PROMPT}{extra_context}\n\n[LABEL TEXT]:\n{text}"
        raw_json_str = await asyncio.to_thread(call_llm, prompt, 1000)
        return parse_llm_response(raw_json_str)

    try:
        result = await _run_llm_extraction(clean_text)
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return {"error": "server_busy", "message": "⚠️ Analysis failed. Please try again."}

    def _flatten(r: dict) -> dict:
        return {
            "calories": float(r.get("calories", 0) or 0),
            "protein":  float(r.get("protein",  0) or 0),
            "carbs":    float(r.get("carbs",     0) or 0),
            "fat":      float(r.get("fat",       0) or 0),
            "sugar":    float(r.get("sugar",     0) or 0),
            "sodium":   float(r.get("sodium_mg", 0) or 0),
            "fiber":    float(r.get("fiber",     0) or 0),
        }

    # 5. Data Normalization
    flattened_nutrients = _flatten(result)

    # 6. DNA Overrides (Atwater Physics + Lie Detector + NOVA 4)
    dna_res = apply_dna_overrides(
        full_ocr_text=extracted_text,
        nutrients=flattened_nutrients,
        ingredients_raw=result.get("ingredients_raw", ""),
        base_score=5,
        category=result.get("product_category", "unknown"),
        front_text=front_text,
    )

    # 6b. Hallucination self-correction — retry once on math mismatch ─────────
    # "BLOCK" is only set by the Atwater math check (never by lie detector which uses "OVERRIDE").
    if dna_res["action"] == "BLOCK":
        logger.warning(
            "Math mismatch detected (reason=%s). Retrying with correction prompt.",
            dna_res["reason"],
        )
        correction_ctx = (
            f"\n\nPREVIOUS EXTRACTION ERROR: {dna_res['reason']}\n"
            "Please double-check the fat value. Re-extract all macros carefully "
            "and ensure Protein*4 + Carbs*4 + Fat*9 ≈ Calories."
        )
        try:
            result = await _run_llm_extraction(clean_text, correction_ctx)
            flattened_nutrients = _flatten(result)
            dna_res = apply_dna_overrides(
                full_ocr_text=extracted_text,
                nutrients=flattened_nutrients,
                ingredients_raw=result.get("ingredients_raw", ""),
                base_score=5,
                category=result.get("product_category", "unknown"),
                front_text=front_text,
            )
            logger.info("Retry DNA result: action=%s", dna_res["action"])
        except Exception as retry_exc:
            logger.error("Retry LLM call failed: %s", retry_exc)

    # 7. Explanation Engine (RDA/NOVA/Humanized)
    explanation = get_explanation_report(flattened_nutrients, result.get("ingredients_raw", ""))

    # 8. Build Output
    final_output = {
        "product_name": result.get("product_name", "Unknown Product"),
        "product_category": result.get("product_category", "unknown"),
        "score": dna_res["score"],
        "verdict": dna_res["reason"] or "Analyzed",
        "ingredients_raw": result.get("ingredients_raw", ""),
        "nutrient_breakdown": [
            {"name": "Calories", "value": flattened_nutrients["calories"], "unit": "kcal"},
            {"name": "Protein", "value": flattened_nutrients["protein"], "unit": "g"},
            {"name": "Carbs", "value": flattened_nutrients["carbs"], "unit": "g"},
            {"name": "Fat", "value": flattened_nutrients["fat"], "unit": "g"},
            {"name": "Sugar", "value": flattened_nutrients["sugar"], "unit": "g"},
            {"name": "Sodium", "value": flattened_nutrients["sodium"], "unit": "mg"},
            {"name": "Fiber", "value": flattened_nutrients["fiber"], "unit": "g"},
        ],
        "explanation": explanation,
        "cons": dna_res.get("extra_flags", []),
        "summary": dna_res["reason"],
        "disclaimer": MEDICAL_DISCLAIMER,
    }
    
    # 9. WhatsApp Formatter (Tiered Content)
    final_output["whatsapp_content"] = get_whatsapp_tiered_content(final_output)

    # 10. Alternatives
    final_output["better_alternative"] = get_healthy_alternative(final_output["product_category"], persona)

    # 11. Cache Table
    cacheable = {k: v for k, v in final_output.items() if k not in ("scan_meta")}
    set_ai_cache(cache_key, cacheable)

    return final_output
