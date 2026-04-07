import os
import re
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
    """Provider-agnostic LLM call. Swap Groq → Anthropic → Ollama here."""
    if not _groq_client:
        logger.error("GROQ_API_KEY is not set in llm.py environment")
        raise RuntimeError("AI Configuration Error: Please check GROQ_API_KEY")

    # Priority: 70b (expert), Fallback: 8b (fast)
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
You are a strict data extraction bot. You will be given messy OCR text from a food label.
Extract the nutritional data into EXACTLY this JSON format.

CRITICAL RULES FOR EXTRACTION (YOU ARE A PHOTOCOPIER):
1. ZERO CREATIVITY: You must output EXACTLY the numbers printed on the label.
2. NO ASSUMPTIONS: If the label says "Protein 0g" on a piece of meat, you MUST output "Protein 0". Do NOT assume the label is wrong. Do NOT try to fix it.
3. NO GUESSING: If you cannot read a number clearly, output 0. Do not guess what the blurry number might be.
4. For 'product_category', ONLY use one of these exact words: ['biscuit', 'noodle', 'chip', 'beverage', 'chocolate', 'snack', 'dairy', 'unknown']. Do not invent categories.
5. Output ONLY valid JSON. No markdown formatting, no chatting, no explanations.

{
  "product_name": "string",
  "product_category": "string",
  "calories": float,
  "protein": float,
  "carbs": float,
  "fat": float,
  "sugar": float,
  "sodium": float,
  "fiber": float,
  "ingredients_raw": "string"
}
"""

def parse_llm_response(llm_output_string: str) -> dict:
    """Strip markdown code blocks if the LLM ignores instructions."""
    clean_string = llm_output_string.strip()
    if clean_string.startswith("```"):
        # Remove first and last lines (the backticks and 'json' tag)
        lines = clean_string.split("\n")
        if len(lines) >= 3:
            clean_string = "\n".join(lines[1:-1])
        else:
            # Fallback for inline backticks
            clean_string = clean_string.replace("```json", "").replace("```", "").strip()
        
    return json.loads(clean_string)


def run_sanity_check(nutrients: dict) -> dict:
    """
    Catches obvious physics-defying LLM hallucinations.
    Following 'Blind Photocopier' rule, this is now an ADVISORY check.
    It returns a warning string instead of blocking the result.
    """
    try:
        fat = float(nutrients.get('fat', 0) or 0)
        protein = float(nutrients.get('protein', 0) or 0)
        carbs = float(nutrients.get('carbs', 0) or 0)
        calories = float(nutrients.get('calories', 0) or 0)
        
        # Advisory Physics check: A food item CANNOT have >100 calories and 0g fat/protein/carbs.
        # If it does, the LLM might have failed to read the macros properly.
        if calories > 100 and fat == 0 and protein == 0 and carbs == 0:
            return {"warning": "⚠️ Extraction Note: Detected calories but 0g macros. Please verify label manually if blurry."}
            
        return {} # Passes check
    except (ValueError, TypeError):
        return {"warning": "⚠️ Data Note: Potential non-numeric values detected in extraction."}


def _validate_atwater_math(
    nutrient_breakdown: list, tolerance: float = 0.25
) -> dict | None:
    """Verify calories match macros via Atwater factors. Returns error dict on mismatch."""

    def _get(key):
        for n in nutrient_breakdown:
            if key in n.get("name", "").lower():
                v = n.get("value", 0)
                return float(v) if isinstance(v, (int, float)) else 0
        return 0

    stated_cal = _get("calorie") or _get("energy") or _get("kcal")
    protein = _get("protein")
    carbs = _get("carbohydrate") or _get("carbs")
    fat = _get("fat")

    if stated_cal == 0 or (protein == 0 and carbs == 0 and fat == 0):
        return None

    calculated_cal = (protein * 4) + (carbs * 4) + (fat * 9)
    margin = stated_cal * tolerance

    if abs(calculated_cal - stated_cal) > margin:
        return {
            "error": "atwater_mismatch",
            "message": f"Nutrient math mismatch: stated {stated_cal} kcal vs calculated {calculated_cal:.0f} kcal from macros.",
            "stated_calories": stated_cal,
            "calculated_calories": round(calculated_cal, 1),
        }
    return None


def _flatten_nutrients(nutrient_list: list) -> dict:
    """Helper to convert LLM nutrient list to flat dict for DNA check."""
    flat = {}
    for n in nutrient_list:
        name = n.get("name", "").lower()
        val = n.get("value", 0)
        # Handle cases where value might be a string or None
        try:
            val = float(val) if val is not None else 0
        except (ValueError, TypeError):
            val = 0

        if any(x in name for x in ["calorie", "energy", "kcal"]):
            flat["calories"] = val
        elif "protein" in name:
            flat["protein"] = val
        elif "saturated" in name:
            flat.setdefault("sat_fat", val)
        elif "trans" in name:
            flat.setdefault("trans_fat", val)
        elif name == "fat" or name == "total fat":
            flat["fat"] = val
        elif "carb" in name and "fiber" not in name:
            flat.setdefault("carbs", val)
        elif "sugar" in name and "added" not in name:
            flat.setdefault("sugar", val)
        elif "sodium" in name:
            flat["sodium"] = val
    return flat


def sanitise_result(result: dict) -> dict:
    """Fix all known LLM output issues: chart rounding, unit strings, defaults, Atwater math."""

    # 1. Ensure all expected fields exist with defaults
    result.setdefault("score", 5)
    result.setdefault("verdict", "Analyzed")
    result.setdefault("product_name", "Unknown Product")
    result.setdefault("product_category", "general")
    result.setdefault("nutrient_breakdown", [])
    result.setdefault("pros", [])
    result.setdefault("cons", [])
    result.setdefault("age_warnings", [])
    result.setdefault("ingredients_raw", "")
    result.setdefault("summary", "")
    result.setdefault("better_alternative", "")

    # 2. Fix chart_data proportions
    cd = result.get("chart_data")
    if (
        isinstance(cd, list)
        and len(cd) == 3
        and all(isinstance(x, (int, float)) for x in cd)
    ):
        total = sum(cd)
        if total > 0 and total != 100:
            scaled = [round(v * 100 / total) for v in cd]
            scaled[scaled.index(max(scaled))] += 100 - sum(scaled)
            result["chart_data"] = scaled
    else:
        result["chart_data"] = [33, 33, 34]  # Neutral default

    # 3. Clean up nutrient values (numeric extraction)
    for n in result.get("nutrient_breakdown", []):
        raw_val = str(n.get("value", "")).replace(",", ".")
        # Detect inequality prefixes to flag trace amounts
        if raw_val.strip().startswith("<"):
            n["is_max_value"] = True
        m = re.search(r"[\d]+\.?[\d]*", raw_val)
        if m:
            n["value"] = float(m.group())
        else:
            n["value"] = 0

    # 4. Atwater check (internal warning only)
    atwater_error = _validate_atwater_math(result.get("nutrient_breakdown", []))
    if atwater_error:
        result["atwater_warning"] = atwater_error
        result["is_low_confidence"] = True

    if "is_low_confidence" not in result:
        result["is_low_confidence"] = False

    return result


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
    """
    Unified high-quality analysis pipeline used by Web, WhatsApp, and B2B.
    Ensures DNA overrides and Alternative Engine are applied consistently.
    """
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

    # 2. Strict Extraction Call (Step 2 Implementation)
    prompt = f"{STRICT_EXTRACTION_PROMPT}\n\n[LABEL TEXT]:\n{extracted_text}"
    raw_json_str = await asyncio.to_thread(call_llm, prompt, 1000)
    
    try:
        # Step 2: Markdown-safe parsing
        result = parse_llm_response(raw_json_str)
    except Exception as e:
        logger.error(f"P0 Parse Error: {e} | Raw: {raw_json_str}")
        return {"error": "server_busy", "message": "⚠️ Analysis failed due to AI data format mismatch. Please try again."}

    # Step 3: Physics Sanity Check
    sanity = run_sanity_check(result)
    if "error" in sanity:
        return sanity

    # Convert strict output to internal flattened format for Score legacy logic
    flattened_nutrients = {
        "calories": float(result.get("calories", 0) or 0),
        "protein": float(result.get("protein", 0) or 0),
        "carbs": float(result.get("carbs", 0) or 0),
        "fat": float(result.get("fat", 0) or 0),
        "sugar": float(result.get("sugar", 0) or 0),
        "sodium": float(result.get("sodium", 0) or 0),
        "fiber": float(result.get("fiber", 0) or 0),
    }

    # 4. Run Physics Check (Advisory)
    extraction_note = run_sanity_check(flattened_nutrients)
    
    # Run DNA Overrides and Scoring logic
    dna_res = apply_dna_overrides(
        full_ocr_text=extracted_text,
        nutrients=flattened_nutrients,
        ingredients_raw=result.get("ingredients_raw", ""),
        base_score=5, # Strict extraction means we re-calculate score locally
        front_text=front_text,
        product_name=result.get("product_name", "product")
    )

    # Collect all warnings (Photocopier Rule)
    all_anomaly_warnings = dna_res.get("anomaly_warnings", [])
    if extraction_note.get("warning"):
        all_anomaly_warnings.append(extraction_note["warning"])

    # Re-map result to match frontend expectation for UI display
    final_output = {
        "product_name": result.get("product_name", "Unknown Product"),
        "product_category": result.get("product_category", "unknown"),
        "score": dna_res["score"] if dna_res["action"] == "OVERRIDE" else 5,
        "verdict": dna_res["reason"] if dna_res["action"] == "OVERRIDE" else "Analyzed",
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
        "chart_data": [33, 33, 34],  # Default macro pie chart proportions (protein, carbs, fat)
        "pros": [],  # Initialize pros list for PDF export and frontend
        "age_warnings": [],  # Initialize age warnings for frontend display
        "cons": dna_res.get("extra_flags", []),
        "summary": dna_res["reason"],
        "anomaly_warnings": all_anomaly_warnings, # Surfaced to Web/WhatsApp
        "disclaimer": MEDICAL_DISCLAIMER
    }

    # 5. Healthy Alternative Engine
    final_output["better_alternative"] = get_healthy_alternative(final_output["product_category"], persona)

    # 7. Cache success
    cacheable = {k: v for k, v in final_output.items() if k not in ("scan_meta")}
    set_ai_cache(cache_key, cacheable)

    return final_output


def upsert_food_product(
    name: str,
    nutrients: list,
    score: int,
    barcode: str = "",
    brand: str = "",
    category: str = "",
    ingredients_raw: str = "",
) -> int:
    """Insert a food product or increment scan_count if barcode already exists."""
    from app.models.db import db_conn

    def _get_nut(key):
        for n in nutrients:
            if key in n.get("name", "").lower():
                v = n.get("value", 0)
                return float(v) if isinstance(v, (int, float)) else 0
        return 0

    cal = _get_nut("calorie") or _get_nut("energy") or _get_nut("kcal")
    pro = _get_nut("protein")
    carb = _get_nut("carbohydrate") or _get_nut("carbs")
    fat = _get_nut("fat")
    sod = _get_nut("sodium")
    fib = _get_nut("fiber")
    sug = _get_nut("sugar")

    with db_conn() as conn:
        if barcode:
            existing = conn.execute(
                "SELECT id FROM food_products WHERE barcode=?", (barcode,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE food_products SET scan_count=scan_count+1 WHERE id=?",
                    (existing["id"],),
                )
                return existing["id"]

        cursor = conn.execute(
            """INSERT INTO food_products(
                name, brand, category, barcode,
                calories_100g, protein_100g, carbs_100g, fat_100g,
                sodium_100g, fiber_100g, sugar_100g,
                eatlytic_score, ingredients_raw, scan_count
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (
                name,
                brand,
                category,
                barcode,
                cal,
                pro,
                carb,
                fat,
                sod,
                fib,
                sug,
                score,
                ingredients_raw,
            ),
        )
        return cursor.lastrowid
