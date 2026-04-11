"""
app/services/llm.py
Complete analysis pipeline:
  Step 1 — Extract ALL nutrients from label (dynamic, not hardcoded 7)
  Step 2 — Score + full analysis (real 1-10, pros, cons, age warnings)
  Step 3 — DNA / Atwater physics override
  Step 4 — Humanized insights + WhatsApp formatter
"""

import os
import re
import json
import logging
import asyncio
import hashlib
from app.models.db import get_ai_cache, set_ai_cache
from app.services.fake_detector import apply_dna_overrides, atwater_math_check
from app.services.alternatives import get_healthy_alternative
from app.services.label_detector import process_image_for_ocr
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

# ─── Rule-based nutrient rating (guaranteed ratings, no LLM needed) ────────
def _rule_rate(name: str, value: float, unit: str) -> dict:
    """
    Returns {rating, impact} for a single nutrient using Indian health standards.
    Works on ANY product worldwide — rates purely from the value+unit+name.
    """
    n = name.lower().replace("of which", "").replace("total", "").strip()

    # ── Beneficial nutrients (more = better) ──────────────────────────
    if "protein" in n:
        if value >= 15: return {"rating": "good",     "impact": f"High protein ({value}g/100g) — great for muscle repair and satiety."}
        if value >= 8:  return {"rating": "moderate", "impact": f"Decent protein ({value}g/100g)."}
        if value >= 3:  return {"rating": "caution",  "impact": f"Low protein ({value}g/100g) — won't keep you full."}
        return             {"rating": "bad",      "impact": f"Very low protein ({value}g/100g) — mainly empty calories."}

    if "fiber" in n or "fibre" in n:
        if value >= 6:  return {"rating": "good",     "impact": f"High dietary fiber ({value}g/100g) — great for gut health."}
        if value >= 3:  return {"rating": "moderate", "impact": f"Moderate fiber ({value}g/100g)."}
        return             {"rating": "caution",  "impact": f"Low fiber ({value}g/100g) — may cause blood sugar spikes."}

    if "calcium" in n:
        if value >= 300: return {"rating": "good",    "impact": f"Rich in calcium ({value}mg/100g) — good for bones."}
        if value >= 100: return {"rating": "moderate","impact": f"Some calcium ({value}mg/100g)."}
        return              {"rating": "caution", "impact": f"Low calcium ({value}mg/100g)."}

    if "iron" in n:
        if value >= 5:  return {"rating": "good",     "impact": f"Good iron source ({value}mg/100g) — helps prevent anaemia."}
        if value >= 2:  return {"rating": "moderate", "impact": f"Some iron ({value}mg/100g)."}
        return             {"rating": "caution",  "impact": f"Low iron ({value}mg/100g)."}

    if "potassium" in n:
        if value >= 500: return {"rating": "good",    "impact": f"Good potassium ({value}mg/100g) — counters sodium effects."}
        if value >= 200: return {"rating": "moderate","impact": f"Some potassium ({value}mg/100g)."}
        return              {"rating": "caution", "impact": f"Low potassium ({value}mg/100g)."}

    if "vitamin" in n:
        return {"rating": "good", "impact": f"{name}: {value}{unit}/100g — contributes to daily vitamin intake."}

    # ── Energy ────────────────────────────────────────────────────────
    if "energy" in n or "calorie" in n or "kcal" in unit.lower():
        if value > 500: return {"rating": "bad",      "impact": f"Very high energy density ({value} kcal/100g)."}
        if value > 400: return {"rating": "caution",  "impact": f"High energy ({value} kcal/100g) — watch portion size."}
        if value > 250: return {"rating": "moderate", "impact": f"Moderate energy ({value} kcal/100g)."}
        return             {"rating": "good",     "impact": f"Lower-calorie product ({value} kcal/100g)."}

    # ── Harmful nutrients (less = better) ────────────────────────────
    if "trans" in n and "fat" in n:
        if value >= 0.5: return {"rating": "bad",     "impact": f"Trans fat present ({value}g/100g) — NO safe level. Raises heart disease risk."}
        if value > 0:    return {"rating": "caution", "impact": f"Trace trans fat ({value}g/100g) — ideally 0."}
        return              {"rating": "good",    "impact": "No trans fat detected."}

    if "saturated" in n and "fat" in n:
        if value >= 10: return {"rating": "bad",      "impact": f"Very high saturated fat ({value}g/100g) — raises LDL cholesterol."}
        if value >= 5:  return {"rating": "caution",  "impact": f"High saturated fat ({value}g/100g)."}
        if value >= 2:  return {"rating": "moderate", "impact": f"Moderate saturated fat ({value}g/100g)."}
        return             {"rating": "good",     "impact": f"Low saturated fat ({value}g/100g)."}

    if "fat" in n:  # catches 'fat', 'total fat'
        if value >= 30: return {"rating": "bad",      "impact": f"Very high fat ({value}g/100g)."}
        if value >= 17: return {"rating": "caution",  "impact": f"High fat ({value}g/100g)."}
        if value >= 8:  return {"rating": "moderate", "impact": f"Moderate fat ({value}g/100g)."}
        return             {"rating": "good",     "impact": f"Low fat ({value}g/100g)."}

    if "added sugar" in n:
        if value >= 15: return {"rating": "bad",      "impact": f"Very high added sugar ({value}g/100g) — no nutrition, pure calories."}
        if value >= 5:  return {"rating": "caution",  "impact": f"High added sugar ({value}g/100g)."}
        if value >= 2:  return {"rating": "moderate", "impact": f"Some added sugar ({value}g/100g)."}
        return             {"rating": "good",     "impact": f"Low added sugar ({value}g/100g)."}

    if "sugar" in n:
        if value >= 22.5: return {"rating": "bad",    "impact": f"Very high sugar ({value}g/100g) — WHO daily limit is 25g."}
        if value >= 15:   return {"rating": "caution","impact": f"High sugar ({value}g/100g)."}
        if value >= 5:    return {"rating": "moderate","impact": f"Moderate sugar ({value}g/100g)."}
        return               {"rating": "good",   "impact": f"Low sugar ({value}g/100g)."}

    if "sodium" in n or "salt" in n:
        if value >= 1000: return {"rating": "bad",    "impact": f"Dangerously high sodium ({value}mg/100g) — over 50% of Indian daily limit."}
        if value >= 700:  return {"rating": "caution","impact": f"High sodium ({value}mg/100g) — raises blood pressure."}
        if value >= 200:  return {"rating": "moderate","impact": f"Moderate sodium ({value}mg/100g)."}
        return               {"rating": "good",   "impact": f"Low sodium ({value}mg/100g)."}

    if "cholesterol" in n:
        if value >= 100: return {"rating": "bad",     "impact": f"High cholesterol ({value}mg/100g)."}
        if value >= 60:  return {"rating": "caution", "impact": f"Moderate cholesterol ({value}mg/100g)."}
        return              {"rating": "good",    "impact": f"Low cholesterol ({value}mg/100g)."}

    if "carb" in n:
        if value >= 65: return {"rating": "caution",  "impact": f"High carbohydrates ({value}g/100g) — check sugar/fiber breakdown."}
        if value >= 35: return {"rating": "moderate", "impact": f"Moderate carbohydrates ({value}g/100g)."}
        return             {"rating": "good",     "impact": f"Lower carbohydrates ({value}g/100g)."}

    # Default — unknown nutrient, no opinion
    return {"rating": "moderate", "impact": f"{name}: {value}{unit} per 100g."}


def _fuzzy_rating(nutrient_name: str, rating_map: dict, value: float, unit: str) -> dict:
    """
    Try LLM-provided rating first (exact → lowercase → keyword fuzzy).
    Fall back to rule_rate if nothing matches.
    """
    # 1. Exact
    if nutrient_name in rating_map:
        return rating_map[nutrient_name]
    # 2. Lowercase exact
    nm_l = nutrient_name.lower().strip()
    for k, v in rating_map.items():
        if k.lower().strip() == nm_l:
            return v
    # 3. Keyword fuzzy — strip noise words and check overlap
    stop = {"of", "which", "total", "added", "dietary", "per", "100g", "the"}
    nm_words = set(nm_l.split()) - stop
    for k, v in rating_map.items():
        k_words = set(k.lower().split()) - stop
        if nm_words and nm_words & k_words:
            return v
    # 4. Rule-based fallback — guaranteed result
    return _rule_rate(nutrient_name, value, unit)


LANGUAGE_MAP = {
    "en": "English", "zh": "Simplified Chinese", "es": "Spanish",
    "ar": "Arabic",  "fr": "French",             "hi": "Hindi (हिन्दी)",
    "pt": "Portuguese", "de": "German",
}


# ── LLM caller ─────────────────────────────────────────────────────────
def call_llm(prompt: str, max_tokens: int = 3000) -> str:
    if not _groq_client:
        raise RuntimeError("AI Configuration Error: GROQ_API_KEY not set")
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
    raise RuntimeError(f"All LLM models failed. Last: {str(last_err)[:100]}")


def parse_llm_response(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1]) if len(lines) >= 3 else s.replace("```json","").replace("```","").strip()
    return json.loads(s)


# ── Prompt 1: Extract EVERY nutrient on the label ──────────────────────
EXTRACTION_PROMPT = """\
You are a precise nutrition label reader for ANY food product worldwide.
Extract EVERY single nutrient row exactly as printed on the label.

STRICT RULES:
1. Prefer "Per 100g" values. If only "Per Serve" exists, use those and note the serving size.
2. Extract ALL rows — not just common ones. Include EVERY line in the nutrition table:
   Energy, Protein, Carbohydrate, Sugars, Added Sugars, Dietary Fiber, Total Fat,
   Saturated Fat, Mono/Polyunsaturated Fat, Trans Fat, Cholesterol, Sodium, Salt,
   Potassium, Calcium, Iron, Magnesium, Zinc, Phosphorus, Selenium,
   Vitamin A/B1/B2/B6/B12/C/D/E/K, Folate, Niacin, Biotin, Pantothenic Acid,
   Moisture, Ash, Starch, Maltodextrin — WHATEVER IS PRINTED.
3. Output EXACT numbers from the label. Never invent or estimate values.
4. Omit nutrients NOT present on the label.
5. Sub-components use prefix "  of which " (e.g., "  of which Sugar").
6. If the label is in another language, translate nutrient names to English.
7. If OCR text is messy, use context to reconstruct numbers (e.g., "13.5g fat" → 13.5).
8. PRODUCT NAME: Read the product name from the label. If not obvious from nutrition label,
   infer it from brand names, packaging words, or ingredients context.
   e.g. "Tata Salt", "Peanut Butter", "Maggi Noodles", etc. NEVER output "Unknown Product".
9. INGREDIENTS: Transcribe the full ingredients list if present.

Return ONLY this JSON (no markdown, no extra text):
{
  "product_name": "string — REQUIRED, infer from context if not explicit",
  "product_category": "Snack|Dairy|Beverage|Cereal|Noodle|Biscuit|Supplement|Spice|Oil|Sauce|Salt|Other",
  "serving_size": "string or null",
  "calories": <number or null>,
  "protein": <number or null>,
  "carbs": <number or null>,
  "fat": <number or null>,
  "sugar": <number or null>,
  "fiber": <number or null>,
  "sodium_mg": <number or null>,
  "saturated_fat": <number or null>,
  "trans_fat": <number or null>,
  "cholesterol_mg": <number or null>,
  "potassium_mg": <number or null>,
  "calcium_mg": <number or null>,
  "iron_mg": <number or null>,
  "nutrients": [
    {"name": "Energy", "value": 389.0, "unit": "kcal"},
    {"name": "Protein", "value": 8.2, "unit": "g"},
    {"name": "Total Carbohydrate", "value": 59.6, "unit": "g"},
    {"name": "  of which Sugar", "value": 1.8, "unit": "g"},
    {"name": "  of which Dietary Fiber", "value": 2.0, "unit": "g"},
    {"name": "Total Fat", "value": 13.5, "unit": "g"},
    {"name": "  of which Saturated Fat", "value": 8.2, "unit": "g"},
    {"name": "  of which Trans Fat", "value": 0.1, "unit": "g"},
    {"name": "Sodium", "value": 920.0, "unit": "mg"}
  ],
  "ingredients_raw": "full ingredients text or empty string"
}
"""


# ── Prompt 2: Full health analysis + real score ────────────────────────
def build_analysis_prompt(
    product_name, category, nutrients_list, ingredients_raw,
    persona, language, nova_level, dna_flags, research_context="",
    blur_info=None
) -> str:
    lang_name = LANGUAGE_MAP.get(language, "English")
    nut_text = "\n".join(
        f"  {n['name']}: {n['value']} {n['unit']}" for n in nutrients_list
    )
    flags_text = "\n".join(f"  - {f}" for f in dna_flags) if dna_flags else "  None"
    
    blur_context = ""
    if blur_info and blur_info.get("detected"):
        if blur_info.get("deblurred"):
            blur_context = (
                f"Note: The image was detected as {blur_info['severity']}ly blurry and "
                f"has been enhanced. OCR text was extracted from the deblurred image. "
                "Prioritize identifiable keywords."
            )
        else:
            blur_context = (
                f"Note: Image has some blur ({blur_info['severity']}). "
                "OCR results might be slightly ambiguous. Use context to infer values."
            )

    return f"""\
You are an expert Indian nutritionist and food safety auditor.
Respond ENTIRELY in {lang_name}.

PRODUCT: {product_name}
CATEGORY: {category}
NOVA LEVEL: {nova_level} (1=whole food, 4=ultra-processed)
RISK FLAGS ALREADY DETECTED: {flags_text}

NUTRIENTS FOUND ON LABEL (per 100g):
{nut_text}

INGREDIENTS: {ingredients_raw or "Not listed"}

PERSONA: {persona}
WEB CONTEXT (Live health research): {research_context or "No specific web data found."}

SCORING RUBRIC — assign the EXACT right score based on Indian health standards:
  9-10 → Whole food / minimal processing. Sugar <2g/100g, sodium <200mg.
  7-8  → Moderately processed. Sugar <5g, sodium <400mg, decent nutrients.
  5-6  → Processed. Sugar 5-15g OR sodium 400-700mg.
  3-4  → High sugar (>15g/100g) OR high sodium (>700mg/100g) OR poor nutrient profile.
  1-2  → Ultra-processed (NOVA 4) OR very high sugar/sodium/saturated fat.
  HARD CAPS: NOVA 4 → max score 4. Sodium >1000mg → max score 5.

Return ONLY this JSON (no markdown):
{{
  "score": <integer 1-10, REQUIRED>,
  "verdict": "<Two-word verdict in {lang_name}>",
  "summary": "<2-sentence professional summary in {lang_name}>",
  "eli5_explanation": "<Child-friendly 1-sentence with one emoji in {lang_name}>",
  "pros": ["<Genuine benefit 1>", "<Genuine benefit 2>", "<Genuine benefit 3>"],
  "cons": ["<Health concern 1>", "<Health concern 2>"],
  "age_warnings": [
    {{"group": "Children (under 12)", "emoji": "👶", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Adults (18-60)",       "emoji": "🧑", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Seniors (60+)",        "emoji": "👴", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Diabetics",            "emoji": "🩸", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Pregnant",             "emoji": "🤰", "status": "warning|caution|good", "message": "<in {lang_name}>"}}
  ],
  "molecular_insight": "<1 sentence on biochemical impact in {lang_name}>",
  "chart_data": [<Safe%>, <Moderate%>, <Risky%>],
  "nutrient_ratings": [
    {{"name": "<same nutrient name from list>", "rating": "good|moderate|caution|bad", "impact": "<1 short sentence on this nutrient level in {lang_name}>"}}
  ],
  "ingredients_spotlight": [
    {{"name": "<ingredient name>", "type": "natural|additive|preservative|emulsifier|vitamin|seasoning", "safety_rating": "safe|moderate|concern", "what_it_is": "<one sentence>", "health_impact": "<one sentence>", "curiosity_fact": "<interesting fact>"}}
  ]
}}
RULES:
- score MUST match actual nutrient values — NEVER default to middle scores.
- chart_data must be [Safe%, Moderate%, Risky%] summing to exactly 100%.
- nutrient_ratings: rate EVERY nutrient from the list above as good/moderate/caution/bad based on Indian health standards.
- ingredients_spotlight: list the TOP 8 most noteworthy ingredients (additives, preservatives, major ingredients).
  If fewer than 8 exist, list ALL of them. NEVER return an empty list if ingredients are present.
- Extract ACTUAL values from the label text.
{blur_context}
"""


# ── Fallback scoring (pure math, used when LLM analysis fails) ────────
def compute_rule_based_score(nutrients: dict, nova_level: int) -> int:
    score = 10
    sugar    = nutrients.get("sugar",         0) or 0
    sodium   = nutrients.get("sodium",        0) or 0
    sat_fat  = nutrients.get("saturated_fat", 0) or 0
    trans    = nutrients.get("trans_fat",     0) or 0
    fiber    = nutrients.get("fiber",         0) or 0
    protein  = nutrients.get("protein",       0) or 0
    calories = nutrients.get("calories",      0) or 0

    if sugar > 22.5:  score -= 4
    elif sugar > 15:  score -= 3
    elif sugar > 5:   score -= 1
    if sodium > 1000: score -= 4
    elif sodium > 700:score -= 3
    elif sodium > 400:score -= 1
    if sat_fat > 10:  score -= 2
    elif sat_fat > 5: score -= 1
    if trans > 0.5:   score -= 3
    elif trans > 0:   score -= 1
    if fiber >= 5:    score += 1
    if protein >= 10: score += 1
    if calories > 500:score -= 1
    if nova_level == 4: score -= 2
    elif nova_level == 3: score -= 1

    return max(1, min(10, score))


async def recover_label_with_ai(raw_text: str) -> dict:
    """
    Final recovery step: asks the LLM if the text looks like a nutrition label.
    This handles stylized mockups and messy OCR that deterministic filters miss.
    """
    if not raw_text or len(raw_text) < 20:
        return {"is_valid": False, "clean_text": ""}

    # Only send top 1000 chars to be efficient
    sample = raw_text[:1000]
    prompt = f"""
    The following text was extracted via OCR from a food product. 
    Analyze if this contains a nutrition table or ingredients. 
    Stylized fonts or minor OCR errors are expected.
    
    TEXT:
    {sample}
    
    RULES:
    1. If you see words like 'Fat', 'Energy', 'Calories', 'Sugar', 'Protein', 'Serving', 'Saturated',
       OR a list of ingredients / chemical names, respond with "VALID".
    2. If it is ONLY marketing text (e.g. "Tasty", "Natural", "Best in India") with absolutely no
       numbers or nutrient names, respond with "INVALID".
    3. When in doubt, respond VALID — it is better to analyze than to reject.
    
    Return ONLY this JSON:
    {{
      "status": "VALID" | "INVALID",
      "cleaned_text": "<Full relevant text if valid, else empty>"
    }}
    """
    try:
        raw = await asyncio.to_thread(call_llm, prompt, 1200)
        res = parse_llm_response(raw)
        return {
            "is_valid": res.get("status") == "VALID",
            "clean_text": res.get("cleaned_text") or raw_text
        }
    except Exception as e:
        logger.error("AI recovery failed: %s", e)
        return {"is_valid": False, "clean_text": ""}


# ── Master pipeline ────────────────────────────────────────────────────
async def unified_analyze_flow(
    extracted_text: str,
    persona: str,
    age_group: str,
    product_category_hint: str,
    language: str,
    web_context: str,  # kept for API signature parity
    blur_info: dict,
    label_confidence: str,
    front_text: str = "",
    image_content: bytes = None,  # DEPRECATED: OCR is done in main.py before this call
) -> dict:
    """
    Full pipeline:
      1. ROI crop (if image given)
      2. Label filter
      3. LLM Step 1 — extract ALL nutrients from label
      4. Atwater physics check + auto-retry on mismatch
      5. DNA overrides (NOVA 4, fake claims, lie detector)
      6. LLM Step 2 — real score + pros/cons/age warnings
      7. Humanized insights (RDA%, teaspoons, walking minutes)
      8. Assemble rich final output
    """

    # Step 1: LAST-RESORT fallback OCR - only fires when caller did not pre-extract text.
    # Normal flow: main.py runs OCR first then passes extracted_text here.
    if image_content and not extracted_text:
        from app.services.ocr import run_ocr
        logger.warning("image_content passed to unified_analyze_flow - OCR should be done in caller.")
        cropped = process_image_for_ocr(image_content)
        ocr_res = run_ocr(cropped, language)
        extracted_text = ocr_res["text"]

    # Step 2: Cache check ─────────────────────────────────────────────
    cache_key = hashlib.md5(
        f"v6:{extracted_text[:120]}:{persona}:{language}".encode()
    ).hexdigest()
    cached = get_ai_cache(cache_key)
    if cached:
        cached["scan_meta"] = {"cached": True, "scans_remaining": 0, "is_pro": False}
        return cached

    # Step 3: Label filter & Fallback ──────────────────────────────────
    from app.services.ocr import universal_label_filter, run_ocr
    filter_result = universal_label_filter(extracted_text)
    
    # SMART FALLBACK: If crop failed to find a label, try the FULL image
    if not filter_result["is_valid"] and image_content:
        logger.info("Smart Crop failed to find label. Falling back to Full Image...")
        full_ocr_res = run_ocr(image_content, language)
        fallback_filter = universal_label_filter(full_ocr_res["text"])
        
        if fallback_filter["is_valid"]:
            logger.info("Full Image fallback SUCCEEDED.")
            filter_result = fallback_filter
            extracted_text = full_ocr_res["text"]
        else:
            logger.warning("Full Image fallback also failed. Attempting AI Recovery...")
            # FINAL TRIPLE-CHECK: Ask AI if it sees a label in the messy text
            ai_recovery = await recover_label_with_ai(full_ocr_res["text"])
            if ai_recovery["is_valid"]:
                logger.info("AI Recovery SUCCEEDED!")
                filter_result = ai_recovery
                extracted_text = ai_recovery["clean_text"]
            else:
                logger.error("AI Recovery also failed. Rejecting image.")
    
    # ADDITIONAL FALLBACK: Even without image_content, try AI recovery on the text we have
    if not filter_result["is_valid"] and extracted_text and len(extracted_text) > 30:
        logger.warning("Filter failed on pre-extracted text. Trying AI Recovery on extracted text...")
        ai_recovery = await recover_label_with_ai(extracted_text)
        if ai_recovery["is_valid"]:
            logger.info("AI Recovery on extracted text SUCCEEDED!")
            filter_result = {"is_valid": True, "clean_text": ai_recovery["clean_text"] or extracted_text}

    if not filter_result["is_valid"]:
        return {
            "error": "no_label",
            "message": "⚠️ No nutrition table detected. Please photograph the back of the package.",
        }
    
    clean_text = filter_result["clean_text"]

    # Step 4: LLM — extract ALL nutrients ─────────────────────────────
    async def _extract(text: str, extra: str = "") -> dict:
        prompt = f"{EXTRACTION_PROMPT}{extra}\n\n[LABEL TEXT]:\n{text}"
        raw = await asyncio.to_thread(call_llm, prompt, 1500)
        return parse_llm_response(raw)

    try:
        extracted = await _extract(clean_text)
    except Exception as e:
        logger.error("Extraction LLM failed: %s", e)
        return {"error": "server_busy", "message": "⚠️ Analysis failed. Please try again."}

    # FIX: If product_name came back as unknown, try to infer from original text
    if not extracted.get("product_name") or extracted.get("product_name", "").lower() in ("unknown", "unknown product", ""):
        logger.warning("Product name extraction failed. Retrying with hint...")
        hint = f"\n\nIMPORTANT: The product name is visible in the image. Please infer it from brand names, packaging words, or product type visible in the text. Do NOT return 'Unknown'. Even a generic name like 'Table Salt' or 'Peanut Butter' is acceptable."
        try:
            extracted2 = await _extract(clean_text, hint)
            if extracted2.get("product_name") and extracted2["product_name"].lower() not in ("unknown", "unknown product", ""):
                extracted["product_name"] = extracted2["product_name"]
        except Exception:
            pass

    category = extracted.get("product_category") or product_category_hint or "unknown"

    # Step 5: Atwater check + retry (Max 2 Attempts) ───────────────────
    def _primary(ex):
        return {
            "calories":      float(ex.get("calories")      or 0),
            "protein":       float(ex.get("protein")       or 0),
            "carbs":         float(ex.get("carbs")          or 0),
            "fat":           float(ex.get("fat")            or 0),
            "sugar":         float(ex.get("sugar")          or 0),
            "fiber":         float(ex.get("fiber")          or 0),
            "saturated_fat": float(ex.get("saturated_fat") or 0),
        }

    for attempt in range(2):
        math_ok = atwater_math_check(_primary(extracted), category)
        if math_ok["is_valid"]:
            break
            
        logger.warning(f"Math mismatch (Attempt {attempt+1}): {math_ok['reason']}")
        correction = (
            f"\n\nERROR IN PREVIOUS EXTRACTION: {math_ok['reason']}\n"
            "CRITICAL: Re-read the label carefully. Ensure sugar/fiber are part of carbs, "
            "and saturated fat is part of total fat. Do NOT double-count. "
            "If the numbers on the label are inconsistent, prioritize the macro gram values over the calorie counts."
        )
        try:
            extracted = await _extract(clean_text, correction)
            category = extracted.get("product_category") or category
        except Exception as re_err:
            logger.error(f"Retry {attempt+1} extraction failed: {re_err}")
            break

    # Step 6: Build rich nutrients dict ───────────────────────────────
    ingredients_raw = extracted.get("ingredients_raw", "") or ""

    # BUG FIX: Removed duckduckgo-search (causes deployment failures on HuggingFace/Docker).
    # Web research is now optional. If a working search module is present in the env, use it.
    # Otherwise skip silently to keep the service stable.
    internal_web_context = ""
    try:
        p_name = extracted.get("product_name", "")
        if p_name and p_name.lower() not in ("unknown", "unknown product"):
            from app.services.research_engine import get_live_search
            internal_web_context = get_live_search(f"health analysis ingredients {p_name} {category}")
            logger.info("Internal Live Research succeeded for: %s", p_name)
    except Exception as e:
        logger.warning("Internal research skipped (module unavailable or search failed): %s", e)

    # Step 7: DNA overrides ───────────────────────────────────────────
    rich = _primary(extracted)
    rich["sodium"] = float(extracted.get("sodium_mg") or 0)
    rich["trans_fat"] = float(extracted.get("trans_fat") or 0)
    rich["cholesterol"] = float(extracted.get("cholesterol_mg") or 0)
    rich["potassium"] = float(extracted.get("potassium_mg") or 0)
    rich["calcium"] = float(extracted.get("calcium_mg") or 0)
    rich["iron"] = float(extracted.get("iron_mg") or 0)

    dna_res = apply_dna_overrides(
        full_ocr_text=extracted_text,
        nutrients=rich,
        ingredients_raw=ingredients_raw,
        base_score=5,
        category=category,
        front_text=front_text,
    )

    if dna_res["action"] == "BLOCK":
        logger.error("DNA BLOCK: %s", dna_res["reason"])
        return {
            "error": "impossible_data",
            "message": dna_res["reason"],
            "dna_action": "BLOCK"
        }

    # Step 8: Explanation engine ──────────────────────────────────────
    explanation = get_explanation_report(rich, ingredients_raw)
    nova_level  = explanation["nova_level"]

    # Step 9: Build dynamic nutrient_breakdown ────────────────────────
    llm_list = extracted.get("nutrients", [])

    # Fallback: reconstruct from top-level fields if list is empty
    if not llm_list:
        _fields = [
            ("Energy",                 "calories",      "kcal"),
            ("Protein",                "protein",       "g"),
            ("Total Carbohydrate",     "carbs",         "g"),
            ("  of which Sugar",       "sugar",         "g"),
            ("  of which Fiber",       "fiber",         "g"),
            ("Total Fat",              "fat",           "g"),
            ("  of which Saturated Fat","saturated_fat","g"),
            ("  of which Trans Fat",   "trans_fat",     "g"),
            ("Sodium",                 "sodium_mg",     "mg"),
            ("Cholesterol",            "cholesterol_mg","mg"),
            ("Potassium",              "potassium_mg",  "mg"),
            ("Calcium",                "calcium_mg",    "mg"),
            ("Iron",                   "iron_mg",       "mg"),
        ]
        for label, key, unit in _fields:
            val = extracted.get(key)
            if val is not None and float(val or 0) > 0:
                llm_list.append({"name": label, "value": float(val), "unit": unit})

    # Normalise: strip unit strings embedded in value field
    nutrient_breakdown = []
    for n in llm_list:
        raw_val = n.get("value", 0)
        if isinstance(raw_val, str):
            m = re.search(r"[\d]+\.?[\d]*", raw_val.replace(",", "."))
            raw_val = float(m.group()) if m else 0.0
        nutrient_breakdown.append({
            "name":  n.get("name", "?"),
            "value": round(float(raw_val or 0), 2),
            "unit":  n.get("unit", ""),
        })

    # Step 10: LLM Step 2 — full analysis ────────────────────────────
    dna_flags = dna_res.get("extra_flags", [])
    if dna_res["action"] == "OVERRIDE":
        dna_flags = [dna_res.get("reason", "")] + dna_flags

    # Build the high-precision nutritionist prompt
    analysis_prompt = build_analysis_prompt(
        product_name=extracted.get("product_name", "Unknown"),
        category=category,
        nutrients_list=nutrient_breakdown,
        ingredients_raw=ingredients_raw,
        persona=persona,
        language=language,
        nova_level=nova_level,
        dna_flags=dna_flags,
        research_context=internal_web_context or web_context,
        blur_info=blur_info,
    )

    analysis = {}
    try:
        raw_analysis = await asyncio.to_thread(call_llm, analysis_prompt, 3000)
        analysis = parse_llm_response(raw_analysis)
        
        # Validate chart_data (sums to 100)
        if "chart_data" in analysis:
            cd = analysis["chart_data"]
            if len(cd) == 3 and sum(cd) != 100 and sum(cd) > 0:
                total = sum(cd)
                analysis["chart_data"] = [round(v * 100 / total) for v in cd]
    except Exception as e:
        logger.error("Analysis LLM failed: %s", e)

    # Step 11: Final Scoring & Humanization ───────────────────────────
    
    # Priority Score logic
    if dna_res["action"] == "OVERRIDE":
        final_score = dna_res.get("base_score", 4)
    elif analysis.get("score"):
        final_score = analysis["score"]
        # Physics Sanity Gate: if it's NOVA 4, cap it even if LLM missed it
        if nova_level == 4 and final_score > 4:
            final_score = 4
    else:
        final_score = compute_rule_based_score(rich, nova_level)

    # Step 12: Merge AI Insights with Physics-based Logic ──────────────
    product_name = extracted.get("product_name") or "Unknown Product"
    # Clean up common bad values
    if product_name.lower() in ("unknown", "unknown product", "", "n/a"):
        product_name = "Unknown Product"
    
    verdict      = analysis.get("verdict") or dna_res.get("reason") or "Analyzed"
    summary      = analysis.get("summary") or dna_res.get("reason") or ""
    pros         = analysis.get("pros", [])
    cons_llm     = analysis.get("cons", [])
    cons         = dna_flags + [c for c in cons_llm if c not in dna_flags]
    
    # Merge Age Warnings: Explanation Engine (Physics) + LLM (AI Context)
    age_warnings = analysis.get("age_warnings", [])
    phys_warnings = explanation.get("persona_warnings", [])
    
    # Create a merged set of warnings keyed by persona
    merged_warnings = {w["group"].lower(): w for w in age_warnings}
    for pw in phys_warnings:
        key = pw["persona"].lower()
        if key in merged_warnings:
            # Append physics-based fact to AI message
            merged_warnings[key]["message"] = f"{merged_warnings[key]['message']}. {pw['msg']}"
            if pw["type"] == "WARNING":
                merged_warnings[key]["status"] = "warning"
        else:
            merged_warnings[key] = {"group": pw["persona"], "status": pw["type"].lower(), "message": pw["msg"], "emoji": "⚠️"}

    # BUG FIX: field was "eli5" in old prompt but frontend expects "eli5_explanation"
    eli5         = analysis.get("eli5_explanation") or analysis.get("eli5", "")
    mol_insight  = analysis.get("molecular_insight", "")
    score_color  = "#22c55e" if final_score >= 7 else "#f59e0b" if final_score >= 4 else "#ef4444"

    # FIX: Merge nutrient ratings using fuzzy matching + guaranteed rule-based fallback
    rating_map = {r["name"]: r for r in analysis.get("nutrient_ratings", [])}
    for n in nutrient_breakdown:
        r = _fuzzy_rating(n["name"], rating_map, float(n.get("value") or 0), n.get("unit", ""))
        n["rating"] = r.get("rating", "moderate")
        n["impact"] = r.get("impact") or f"{n['name']}: {n['value']}{n.get('unit', '')} per 100g."

    # BUG FIX: ingredients_spotlight was never populated from LLM
    ingredients_spotlight = analysis.get("ingredients_spotlight", [])
    
    # FIX: If spotlight is empty but ingredients exist, generate rule-based fallback cards
    if not ingredients_spotlight and ingredients_raw:
        # Parse top ingredients from raw text and create simple cards
        ing_list = [i.strip() for i in re.split(r'[,;]', ingredients_raw) if i.strip()][:8]
        for ing in ing_list:
            if len(ing) > 2:
                ingredients_spotlight.append({
                    "name": ing.title(),
                    "type": "natural",
                    "safety_rating": "safe",
                    "what_it_is": f"{ing.title()} is a food ingredient.",
                    "health_impact": "Part of the product formulation.",
                    "curiosity_fact": "Check the full ingredients list for details."
                })

    # BUG FIX: merged_warnings was built but never converted back to list
    age_warnings_final = list(merged_warnings.values())

    final_output = {
        "product_name":          product_name,
        "product_category":      category,
        "serving_size":          extracted.get("serving_size"),
        "score":                 final_score,
        "score_color":           score_color,
        "verdict":               verdict,
        "summary":               summary,
        "nutrient_breakdown":    nutrient_breakdown,
        "pros":                  pros,
        "cons":                  cons,
        "age_warnings":          age_warnings_final,
        "eli5_explanation":      eli5,
        "molecular_insight":     mol_insight,
        "ingredients_raw":       ingredients_raw,
        "ingredients_spotlight": ingredients_spotlight,
        "explanation":           explanation,
        "better_alternative":    get_healthy_alternative(category, persona),
        "whatsapp_content":      {},
        "disclaimer":            MEDICAL_DISCLAIMER,
    }

    try:
        final_output["whatsapp_content"] = get_whatsapp_tiered_content(final_output)
    except Exception:
        pass

    cacheable = {k: v for k, v in final_output.items() if k != "scan_meta"}
    set_ai_cache(cache_key, cacheable)
    return final_output


# ── Legacy shim ───────────────────────────────────────────────────────
def upsert_food_product(
    name, nutrients, score, ingredients_raw="",
    barcode=None, brand="", category="", source="llm_scan",
) -> int:
    from app.models.db import db_conn

    def _get(key):
        for n in nutrients:
            if key in n.get("name", "").lower():
                v = n.get("value", 0)
                return float(v) if isinstance(v, (int, float)) else 0
        return 0

    cal = _get("calorie") or _get("energy")
    with db_conn() as conn:
        existing = (
            conn.execute(
                "SELECT id FROM food_products WHERE barcode=?", (barcode,)
            ).fetchone() if barcode else
            conn.execute(
                "SELECT id FROM food_products WHERE LOWER(name)=LOWER(?) AND LOWER(brand)=LOWER(?)",
                (name.strip(), brand.strip()),
            ).fetchone()
        )
        if existing:
            conn.execute(
                "UPDATE food_products SET scan_count=scan_count+1, updated_at=datetime('now') WHERE id=?",
                (existing["id"],),
            )
            return existing["id"]
        cursor = conn.execute(
            """INSERT INTO food_products
               (name,brand,category,barcode,calories_100g,protein_100g,carbs_100g,
                fat_100g,sodium_100g,fiber_100g,sugar_100g,sat_fat_100g,
                eatlytic_score,ingredients_raw,source,scan_count)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (name.strip(), brand, category, barcode,
             cal, _get("protein"), _get("carb"), _get("fat"),
             _get("sodium"), _get("fiber") or _get("fibre"),
             _get("sugar"), _get("saturated"),
             score, ingredients_raw, source),
        )
        return cursor.lastrowid
