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
1. PRODUCT NAME: The product name is ALWAYS in the first few lines of the text. Extract it from there (e.g., "MAGGI Masala Noodles", "Parle-G Gluco Biscuit"). NEVER output "Unknown Product" if any text is provided.
2. SINGLE INGREDIENTS: For pure products (Salt, Sugar, Oil), it is NORMAL to have 0g for everything except the main ingredient. Do NOT flag this as an error.
3. ZERO CREATIVITY: Output EXACTLY the numbers printed. If it says "Sodium 39,100mg", output 39100. 
4. MISSING DATA: If a macro is not mentioned, output 0.
5. CATEGORIES: ONLY use: ['biscuit', 'noodle', 'chip', 'beverage', 'chocolate', 'snack', 'dairy', 'salt', 'sugar', 'oil', 'spice', 'spread', 'unknown'].
6. TABLE LOGIC (STRICT): Food labels often have multiple columns (e.g., "Per 100g", "Per Serve", "% RDA"). 
   - RULE 0 (ISOLATION): If you see two numbers for one nutrient (e.g. "Fat 13.5 14.5"), the first is Per 100g. The second is Per Serve. NEVER SUM THEM. NEVER ADD THEM. 
   - You MUST ONLY extract the "Per 100g" column.
   - IGNORE any numbers followed by a '%' sign (e.g. "14%" is NOT a nutrient value).
   - If a line has multiple numbers (e.g. "Energy 389 272 14%"), the FIRST number is "Per 100g". Extract 389. 
   - NEVER ADD OR SUM numbers from different columns together.
   - If you see a number that looks like a sum of others (e.g. Carbs 59 + 41 = 100), you HAVE FAILED. Only take the first number.
   - If a row contains only numbers (e.g. "389 272 14%"), align them with the header row above it.
8. PHYSICS CHECK: Before outputting, do a quick mental check. 
   - Calculation: (Protein * 4) + (Carbs * 4) + (Fat * 9) = Computed Calories.
   - Computed Calories should be within 25% of the "calories" you extracted. 
   - If "calories" is 389 but Protein is 82g and Fat is 28g, your calculation says 580. 580 is 49% higher than 389. This is WRONG. Re-read the numbers. Protein is likely 8.2g, not 82g.
   - Adjust your extraction to maintain physical reality.
10. NUTRIENT HIERARCHY (CRITICAL): 
    - Sugar and Fiber are SUB-COMPONENTS of Carbohydrates. NEVER extract values that make Sugar + Fiber > Total Carbohydrates.
    - Saturated and Trans Fat are SUB-COMPONENTS of Total Fat. NEVER extract values that make them > Total Fat.
    - If a label says "Carbohydrates 59.6g" and below it says "Sugar 1.8g", the 59.6g ALREADY includes the 1.8g. 
11. Output ONLY valid JSON. No markdown, no chatting.

HINT: If you see separate lines with numbers like "384" after a header like "Energy", assume they belong together. For example:
"Energy (kcal) per 100g per serve (70g)"
"384 288"
This means 384 kcal per 100g. Extract 384 as "calories".

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
) -> dict:
    """The Master Pipeline."""

    # 1. Cache Check
    cache_key = hashlib.md5(
        f"{extracted_text[:100]}:{persona}:{language}".encode()
    ).hexdigest()
    cached = get_ai_cache(cache_key)
    if cached:
        cached["scan_meta"] = {
            "cached": True,
            "scans_remaining": 0,
            "is_pro": False,
            "scans_used": 0,
        }
        return cached

    # 2. THE TRASH COMPACTOR (Filter out FSSAI, marketing, etc.)
    from app.services.ocr import universal_label_filter

    filter_result = universal_label_filter(extracted_text)

    if not filter_result["is_valid"]:
        return {
            "error": "no_label",
            "message": "No nutrition table found. Ensure the numbers (e.g., 10g, 50kcal) are visible.",
        }

    clean_text = filter_result["clean_text"]
    logger.info(
        f"Original OCR length: {len(extracted_text)}, Cleaned length: {len(clean_text)}"
    )

    # 3. THE DATA ENTRY CLERK (Strict Extraction)
    prompt = f"{STRICT_EXTRACTION_PROMPT}\n\n[LABEL TEXT]:\n{clean_text}"
    raw_json_str = await asyncio.to_thread(call_llm, prompt, 1000)

    try:
        result = parse_llm_response(raw_json_str)
    except Exception as e:
        logger.error(f"Parse Error: {e} | Raw: {raw_json_str}")
        return {
            "error": "server_busy",
            "message": "⚠️ Analysis failed due to AI formatting. Please try again.",
        }

    # 4. Format for DNA Engine (Map sodium_mg -> sodium)
    flattened_nutrients = {
        "calories": float(result.get("calories", 0) or 0),
        "protein": float(result.get("protein", 0) or 0),
        "carbs": float(result.get("carbs", 0) or 0),
        "fat": float(result.get("fat", 0) or 0),
        "sugar": float(result.get("sugar", 0) or 0),
        "sodium": float(result.get("sodium_mg", 0) or 0),
        "fiber": float(result.get("fiber", 0) or 0),
    }

    # 5. THE STRICT BOSS (DNA Overrides - Math check, Lie Detector, NOVA 4)
    dna_res = apply_dna_overrides(
        full_ocr_text=extracted_text,
        nutrients=flattened_nutrients,
        ingredients_raw=result.get("ingredients_raw", ""),
        base_score=5,
        category=result.get("product_category", "unknown"),
        front_text=front_text,
    )

    # 5b. SELF-CORRECTION LOOP: If Math Mismatch detected, retry once with feedback
    if dna_res["action"] == "BLOCK" and "Math Mismatch" in dna_res["reason"]:
        logger.info(f"🔄 Math Mismatch detected ({dna_res['reason']}). Triggering AI Self-Correction...")
        
        correction_prompt = f"""
{STRICT_EXTRACTION_PROMPT}

[ERROR FEEDBACK]:
Your previous extraction FAILED our Physics-Sanity-Gate.
Problem: {dna_res['reason']}
Likely cause: You might have summed 'Per 100g' and 'Per Serve' columns together, or misread a decimal.

[LABEL TEXT]:
{clean_text}

Analyze the columns again. IGNORE the 'Per Serve' column. Extract ONLY the 'Per 100g' values.
"""
        retry_json_str = await asyncio.to_thread(call_llm, correction_prompt, 1000)
        try:
            retry_result = parse_llm_response(retry_json_str)
            # Re-flatten and re-calculate
            retry_nutrients = {
                "calories": float(retry_result.get("calories", 0) or 0),
                "protein": float(retry_result.get("protein", 0) or 0),
                "carbs": float(retry_result.get("carbs", 0) or 0),
                "fat": float(retry_result.get("fat", 0) or 0),
                "sugar": float(retry_result.get("sugar", 0) or 0),
                "sodium": float(retry_result.get("sodium_mg", 0) or 0),
                "fiber": float(retry_result.get("fiber", 0) or 0),
            }
            retry_dna = apply_dna_overrides(
                full_ocr_text=extracted_text,
                nutrients=retry_nutrients,
                ingredients_raw=retry_result.get("ingredients_raw", ""),
                base_score=5,
                category=retry_result.get("product_category", "unknown"),
                front_text=front_text,
            )
            # If retry succeeded or improved, use it
            if retry_dna["action"] != "BLOCK" or "Math Mismatch" not in retry_dna["reason"]:
                logger.info("✅ AI successfully self-corrected.")
                result = retry_result
                flattened_nutrients = retry_nutrients
                dna_res = retry_dna
        except Exception as e:
            logger.error(f"Retry Parse Fail: {e}")

    # 6. Build Final Output for Frontend
    final_output = {
        "product_name": result.get("product_name", "Unknown Product"),
        "product_category": result.get("product_category", "unknown"),
        "score": dna_res["score"],
        "verdict": dna_res["reason"] if dna_res["reason"] else "Analyzed",
        "ingredients_raw": result.get("ingredients_raw", ""),
        "nutrient_breakdown": [
            {
                "name": "Calories",
                "value": flattened_nutrients["calories"],
                "unit": "kcal",
            },
            {"name": "Protein", "value": flattened_nutrients["protein"], "unit": "g"},
            {"name": "Carbs", "value": flattened_nutrients["carbs"], "unit": "g"},
            {"name": "Fat", "value": flattened_nutrients["fat"], "unit": "g"},
            {"name": "Sugar", "value": flattened_nutrients["sugar"], "unit": "g"},
            {"name": "Sodium", "value": flattened_nutrients["sodium"], "unit": "mg"},
            {"name": "Fiber", "value": flattened_nutrients["fiber"], "unit": "g"},
        ],
        "chart_data": [33, 33, 34],
        "pros": [],
        "age_warnings": [],
        "cons": dna_res.get("extra_flags", []),
        "summary": dna_res["reason"],
        "disclaimer": MEDICAL_DISCLAIMER,
    }

    # 7. Alternative Engine
    final_output["better_alternative"] = get_healthy_alternative(
        final_output["product_category"], persona
    )

    # 8. Cache success
    cacheable = {k: v for k, v in final_output.items() if k not in ("scan_meta")}
    set_ai_cache(cache_key, cacheable)

    return final_output
