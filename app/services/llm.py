"""
app/services/llm.py
LLM abstraction layer + proprietary food database population.
"""

import os
import re
import json
import logging
import asyncio
from app.models.db import get_ai_cache, set_ai_cache, db_conn

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
        raise RuntimeError("GROQ_API_KEY not set")
    for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
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
            logger.warning("LLM %s failed: %s", model, exc)
    raise RuntimeError("All LLM models failed")


def build_analysis_prompt(
    extracted_text: str,
    persona: str,
    age_group: str,
    product_category: str,
    language: str,
    web_context: str,
    label_confidence: str,
    blur_info: dict,
) -> str:
    lang_name = LANGUAGE_MAP.get(language, "English")
    conf_note = (
        "⚠️ Label text may be partial — only list nutrients you can read confidently."
        if label_confidence == "low"
        else ""
    )
    blur_ctx = ""
    if blur_info.get("detected"):
        verb = (
            "enhanced via Wiener deconvolution"
            if blur_info.get("deblurred")
            else "blurry, used original"
        )
        blur_ctx = f"IMAGE: {blur_info['severity']}ly blurry ({verb}). Only report confident values."

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

    # Sanitise extracted_text to prevent prompt injection
    safe_text = extracted_text.replace('"', "'").replace("\n", " ").strip()
    
    return f"""[INST]
You are an expert nutritional scientist and food safety auditor.
CRITICAL: Respond ENTIRELY in {lang_name}. Every text field MUST be in {lang_name}.
Persona: {persona} | Age: {age_group} | Category: {product_category}
{persona_rules}
{conf_note}
{blur_ctx}
Label Text: "{safe_text}"
Web Context: "{web_context}"

Return ONLY valid JSON — no markdown, no preamble:
{{
  "product_name"      : "Short name from label",
  "product_category"  : "Snack|Dairy|Beverage|Cereal|Supplement|etc.",
  "score"             : <INTEGER 1-10 per SCORING RUBRIC — modified by persona rules>,
  "verdict"           : "Two-word verdict in {lang_name}",
  "fake_claim_detected": <true if text claims 'No Added Sugar'/'Sugar-Free' BUT ingredients have Maltodextrin, Dextrose, Fructose, Corn Syrup, Date Syrup>,
  "ingredients_raw"   : "Comma, separated, list, of, ingredients, extracted",
  "chart_data"        : [<Safe%>, <Moderate%>, <Risky%>],
  "summary"           : "2-sentence professional summary in {lang_name}.",
  "eli5_explanation"  : "Child-friendly explanation with emojis in {lang_name}.",
  "molecular_insight" : "1-2 sentences on biochemical body impact in {lang_name}.",
  "paragraph_benefits": "Full paragraph on genuine benefits in {lang_name}.",
  "paragraph_uniqueness": "Unique characteristics OR 2 better alternatives in {lang_name}.",
  "is_unique"         : true,
  "nutrient_breakdown": [
    {{"name":"Protein","value":<ACTUAL g from label>,"unit":"g","rating":"good","impact":"brief note in {lang_name}"}},
    {{"name":"Sugar","value":<ACTUAL g>,"unit":"g","rating":"moderate","impact":"brief note"}},
    {{"name":"Fat","value":<ACTUAL g>,"unit":"g","rating":"good","impact":"brief note"}},
    {{"name":"Sodium","value":<ACTUAL mg>,"unit":"mg","rating":"caution","impact":"brief note"}},
    {{"name":"Fiber","value":<ACTUAL g>,"unit":"g","rating":"good","impact":"brief note"}}
  ],
  "pros"           : ["Benefit 1 in {lang_name}", "Benefit 2", "Benefit 3"],
  "cons"           : ["Risk 1 in {lang_name}", "Risk 2"],
  "age_warnings"   : [
    {{"group":"Children","emoji":"👶","status":"warning","message":"in {lang_name}"}},
    {{"group":"Adults","emoji":"🧑","status":"good","message":"in {lang_name}"}},
    {{"group":"Seniors","emoji":"👴","status":"caution","message":"in {lang_name}"}},
    {{"group":"Pregnant","emoji":"🤰","status":"caution","message":"in {lang_name}"}}
  ],
  "better_alternative": "A specific healthier alternative in {lang_name}.",
  "is_low_confidence" : false
}}

SCORING RUBRIC — MANDATORY, never use 6 or 7 as defaults:
  9-10: Whole food, no added sugar, low sodium, high fibre/protein
  7-8 : Mildly processed, sugar <5g/100g, reasonable sodium
  5-6 : Processed, sugar 5-15g/100g OR sodium 400-700mg/100g
  3-4 : High sugar >15g/100g OR sodium >700mg/100g OR poor profile
  1-2 : Ultra-processed, very high sugar/sodium/sat-fat
RULES: chart_data sums to 100 | rating: good|moderate|caution|bad | status: good|caution|warning
[/INST]"""


def _validate_atwater_math(
    nutrient_breakdown: list, tolerance: float = 0.15
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


def sanitise_result(result: dict) -> dict:
    """Fix all known LLM output issues: chart rounding, unit strings, defaults, Atwater math."""
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
        # Default to a neutral state if LLM fails to provide chart data
        result["chart_data"] = [0, 0, 0]

    for n in result.get("nutrient_breakdown", []):
        m = re.search(r"[\d]+\.?[\d]*", str(n.get("value", "")).replace(",", "."))
        if m:
            n["value"] = float(m.group())

    atwater_error = _validate_atwater_math(result.get("nutrient_breakdown", []))
    if atwater_error:
        result["atwater_warning"] = atwater_error
        result["is_low_confidence"] = True

    # NOTE: Lie Detector and NOVA 4 checks are now handled exclusively
    # by app/services/fake_detector.py apply_dna_overrides().
    # They are intentionally NOT duplicated here.

    result.setdefault("score", 5)
    result.setdefault("verdict", "Analyzed")
    result.setdefault("product_name", "Unknown Product")
    result.setdefault("nutrient_breakdown", [])
    result.setdefault("pros", [])
    result.setdefault("cons", [])
    result.setdefault("age_warnings", [])
    if "is_low_confidence" not in result:
        result["is_low_confidence"] = False
    return result


async def analyse_label(
    extracted_text: str,
    persona: str,
    age_group: str,
    product_category: str,
    language: str,
    web_context: str,
    blur_info: dict,
    label_confidence: str,
) -> dict:
    cache_key = f"v4:{language}:{persona}:{age_group}:{extracted_text[:80]}"
    cached = get_ai_cache(cache_key)
    if cached:
        return cached

    prompt = build_analysis_prompt(
        extracted_text,
        persona,
        age_group,
        product_category,
        language,
        web_context,
        label_confidence,
        blur_info,
    )
    raw = await asyncio.to_thread(call_llm, prompt, 2500)
    result = sanitise_result(json.loads(raw))
    result["disclaimer"] = MEDICAL_DISCLAIMER

    cacheable = {
        k: v
        for k, v in result.items()
        if k not in ("blur_info", "scan_meta", "allergen_warning")
    }
    set_ai_cache(cache_key, cacheable)
    return result


def upsert_food_product(
    name: str,
    nutrients: list,
    score: int,
    ingredients_raw: str = "",
    barcode: str | None = None,
    brand: str = "",
    category: str = "",
    source: str = "llm_scan",
) -> int:
    def _get(key):
        for n in nutrients:
            if key in n.get("name", "").lower():
                v = n.get("value", 0)
                return float(v) if isinstance(v, (int, float)) else 0
        return 0

    cal = _get("calorie") or _get("energy") or _get("kcal")
    prot = _get("protein")
    carb = _get("carbohydrate") or _get("carbs")
    fat = _get("fat")
    sod = _get("sodium")
    fib = _get("fiber") or _get("fibre")
    sug = _get("sugar")
    sat = _get("saturated")

    with db_conn() as conn:
        if barcode:
            existing = conn.execute(
                "SELECT id, scan_count FROM food_products WHERE barcode=?", (barcode,)
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT id, scan_count FROM food_products WHERE name=? AND brand=?",
                (name.strip(), brand.strip()),
            ).fetchone()

        if existing:
            conn.execute(
                "UPDATE food_products SET scan_count=scan_count+1, updated_at=datetime('now') WHERE id=?",
                (existing["id"],),
            )
            return existing["id"]
        else:
            cursor = conn.execute(
                """INSERT INTO food_products
                   (name,brand,category,barcode,calories_100g,protein_100g,carbs_100g,
                    fat_100g,sodium_100g,fiber_100g,sugar_100g,sat_fat_100g,
                    eatlytic_score,ingredients_raw,source,scan_count)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                (
                    name.strip(),
                    brand.strip(),
                    category.strip(),
                    barcode.strip() if barcode else None,
                    cal,
                    prot,
                    carb,
                    fat,
                    sod,
                    fib,
                    sug,
                    sat,
                    score,
                    ingredients_raw,
                    source,
                ),
            )
            return cursor.lastrowid


def get_food_from_db(name: str = "", barcode: str = "") -> dict | None:
    with db_conn() as conn:
        if barcode:
            row = conn.execute(
                "SELECT * FROM food_products WHERE barcode=? AND verified=1", (barcode,)
            ).fetchone()
        elif name:
            row = conn.execute(
                "SELECT * FROM food_products WHERE name LIKE ? AND verified=1 ORDER BY scan_count DESC LIMIT 1",
                (f"%{name}%",),
            ).fetchone()
        else:
            return None
    return dict(row) if row else None
