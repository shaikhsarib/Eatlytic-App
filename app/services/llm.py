"""
app/services/llm.py  — Single-Pass Architecture (v7)
ONE LLM call does EVERYTHING:
  - Reads every nutrient row from the label (dynamic, not hardcoded)
  - Scores the product 1-10
  - Writes pros / cons / age warnings
  - Fills ingredient spotlight cards
  - Returns chart data + molecular insight

This eliminates the old "Extract → wait → Analyze → wait" double-call
and cuts total latency by ~40%.
"""

import os
import re
import json
import logging
import asyncio
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
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

LANGUAGE_MAP = {
    "en": "English", "zh": "Simplified Chinese", "es": "Spanish",
    "ar": "Arabic",  "fr": "French",             "hi": "Hindi (हिन्दी)",
    "pt": "Portuguese", "de": "German",
}


# ── LLM caller (Robust with Retries) ────────────────────────────────────—
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception), # Categorized retries below
    reraise=True
)
def call_llm(prompt: str, max_tokens: int = 4000) -> str:
    if not _groq_client:
        raise RuntimeError("AI Configuration Error: GROQ_API_KEY not set")
        
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    last_err = None
    
    for model in models:
        try:
            current_max = 2000 if "8b" in model else max_tokens
            comp = _groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=current_max,
                response_format={"type": "json_object"},
            )
            return comp.choices[0].message.content
        except Exception as exc:
            err_msg = str(exc).lower()
            # If it's a 413 (Payload Too Large), don't retry this model, try the next
            if "413" in err_msg or "payload too large" in err_msg:
                logger.warning("Groq model %s payload too large (413).", model)
                last_err = exc
                continue
                
            # If it's a 429 (Rate Limit), don't raise immediately! Try the fallback model.
            if "429" in err_msg or "rate limit" in err_msg:
                logger.warning("Groq model %s rate limited (429). Trying fallback...", model)
                last_err = exc
                continue
                
            logger.warning("Groq model %s failed: %s", model, exc)
            last_err = exc
            
    # If we reach here, ALL models failed. Ensure we bubble up the error to trigger Tenacity.
    raise last_err


def parse_llm_response(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1]) if len(lines) >= 3 else s.replace("```json", "").replace("```", "").strip()
    return json.loads(s)


# ── Rule-based nutrient rating (guaranteed, no second LLM needed) ────────
def _rule_rate(name: str, value: float, unit: str) -> dict:
    n = name.lower().replace("of which", "").replace("total", "").strip()

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

    if "energy" in n or "calorie" in n or "kcal" in unit.lower():
        if value > 500: return {"rating": "bad",      "impact": f"Very high energy density ({value} kcal/100g)."}
        if value > 400: return {"rating": "caution",  "impact": f"High energy ({value} kcal/100g) — watch portion size."}
        if value > 250: return {"rating": "moderate", "impact": f"Moderate energy ({value} kcal/100g)."}
        return             {"rating": "good",     "impact": f"Lower-calorie product ({value} kcal/100g)."}

    if "trans" in n and "fat" in n:
        if value >= 0.5: return {"rating": "bad",     "impact": f"Trans fat present ({value}g/100g) — NO safe level. Raises heart disease risk."}
        if value > 0:    return {"rating": "caution", "impact": f"Trace trans fat ({value}g/100g) — ideally 0."}
        return              {"rating": "good",    "impact": "No trans fat detected."}

    if "saturated" in n and "fat" in n:
        if value >= 10: return {"rating": "bad",      "impact": f"Very high saturated fat ({value}g/100g) — raises LDL cholesterol."}
        if value >= 5:  return {"rating": "caution",  "impact": f"High saturated fat ({value}g/100g)."}
        if value >= 2:  return {"rating": "moderate", "impact": f"Moderate saturated fat ({value}g/100g)."}
        return             {"rating": "good",     "impact": f"Low saturated fat ({value}g/100g)."}

    if "fat" in n:
        if value >= 30: return {"rating": "bad",      "impact": f"Very high fat ({value}g/100g)."}
        if value >= 17: return {"rating": "caution",  "impact": f"High fat ({value}g/100g)."}
        if value >= 8:  return {"rating": "moderate", "impact": f"Moderate fat ({value}g/100g)."}
        return             {"rating": "good",     "impact": f"Low fat ({value}g/100g)."}

    if "added sugar" in n:
        if value >= 15: return {"rating": "bad",      "impact": f"Very high added sugar ({value}g/100g)."}
        if value >= 5:  return {"rating": "caution",  "impact": f"High added sugar ({value}g/100g)."}
        if value >= 2:  return {"rating": "moderate", "impact": f"Some added sugar ({value}g/100g)."}
        return             {"rating": "good",     "impact": f"Low added sugar ({value}g/100g)."}

    if "sugar" in n:
        if value >= 22.5: return {"rating": "bad",   "impact": f"Very high sugar ({value}g/100g) — WHO daily limit is 25g."}
        if value >= 15:   return {"rating": "caution","impact": f"High sugar ({value}g/100g)."}
        if value >= 5:    return {"rating": "moderate","impact": f"Moderate sugar ({value}g/100g)."}
        return               {"rating": "good",   "impact": f"Low sugar ({value}g/100g)."}

    if "sodium" in n or "salt" in n:
        if value >= 1000: return {"rating": "bad",   "impact": f"Dangerously high sodium ({value}mg/100g) — over 50% of Indian daily limit."}
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

    return {"rating": "moderate", "impact": f"{name}: {value}{unit} per 100g."}


def _fuzzy_rating(nutrient_name: str, rating_map: dict, value: float, unit: str) -> dict:
    if nutrient_name in rating_map:
        return rating_map[nutrient_name]
    nm_l = nutrient_name.lower().strip()
    for k, v in rating_map.items():
        if k.lower().strip() == nm_l:
            return v
    stop = {"of", "which", "total", "added", "dietary", "per", "100g", "the"}
    nm_words = set(nm_l.split()) - stop
    for k, v in rating_map.items():
        k_words = set(k.lower().split()) - stop
        if nm_words and nm_words & k_words:
            return v
    return _rule_rate(nutrient_name, value, unit)


# ── SINGLE SUPER-PROMPT (extract + analyze in one call) ─────────────────
def build_super_prompt(
    label_text: str,
    persona: str,
    language: str,
    blur_info: dict = None,
    dna_flags: list = None,
    nova_level: int = 1,
    research_context: str = "",
) -> str:
    lang_name = LANGUAGE_MAP.get(language, "English")
    flags_text = "\n".join(f"  - {f}" for f in (dna_flags or [])) or "  None"
    # AGGRESSIVE TRUNCATION to stay within 6,000 TPM limit
    # 3,000 chars of OCR text + 2,000 chars of research context = ~2,000 tokens
    safe_label_text = label_text[:3000]
    safe_research = research_context[:2000]

    # FIX: build blur_context string from blur_info dict (was NameError before)
    blur_context = ""
    if blur_info and blur_info.get("detected"):
        if blur_info.get("deblurred"):
            blur_context = (
                f"NOTE: Image was blurry ({blur_info.get('severity','moderate')}) "
                "and has been AI-enhanced. OCR may have minor errors — use context clues."
            )
        else:
            blur_context = (
                f"NOTE: Image has some blur ({blur_info.get('severity','mild')}). "
                "OCR results may be slightly ambiguous — use context to infer values."
            )

    return f"""\
You are the Eatlytic "Nutrition Label Analysis and Card Generation Specialist".
You are a world-class food scientist with expertise across global food standards
(Indian FSSAI, US FDA, UK FSA, EU, Codex Alimentarius).
Your task: read ANY food label from ANY country and produce both a high-fidelity
extraction AND a scored health analysis — all in ONE response.

## STEP 1 — READ THE LABEL (Blind Photocopier)
1. **Extract ALL nutritional rows** present on the label EXACTLY as printed.
   - Prefer "Per 100g" / "Per 100ml" column. If ONLY "Per Serve" exists, use that and note it.
   - **Multi-column labels** (Indian format): If the table has columns like
     "Per 100g | Per Serve | %GDA" — extract ONLY the "Per 100g" column values.
     The OCR text may mix columns using tabs or spaces — the FIRST number after
     the nutrient name is usually Per 100g. Example:
       "Energy (kcal)\t389\t272\t14%" → extract 389
       "Protein (g)\t8.2\t5.7\t11%" → extract 8.2
   - Include EVERY row without exception: Energy (kcal/kJ), Protein, Total Carbohydrate,
     of which Sugars, of which Added Sugars, Dietary Fibre/Fiber, Total Fat,
     of which Saturated Fat, of which Trans Fat, of which Mono-unsaturated Fat,
     of which Poly-unsaturated Fat, Cholesterol, Sodium/Salt, Potassium, Calcium,
     Iron, Magnesium, Iodine, Selenium, Vitamins (A/B1/B2/B6/B12/C/D/E/K),
     Moisture, Ash — EVERYTHING visible in the nutrition table.
   - Do NOT skip any row. Do NOT hallucinate rows not present on the label.
   - If 7 rows are on the label → 7 nutrient cards. If 14 → 14. NEVER force a template.
   - OCR errors: numbers like "l3.5" (letter l) = 13.5, "O.1" (letter O) = 0.1.
     Use nutritional context to identify and correct obvious OCR character errors.
2. **Product name**: Read from label text. If not explicit, infer from ingredients
   or product type. NEVER output "Unknown Product" or leave blank.
3. **Ingredients**: Transcribe the complete ingredients list if printed on the label.

## STEP 2 — ANALYZE (use global health standards)
Respond text fields ENTIRELY in {lang_name}.
PERSONA: {persona}
NOVA LEVEL: {nova_level} (1=whole food, 4=ultra-processed)
RISK FLAGS: {flags_text}
{safe_research or ""}
{blur_context}

SCORING RUBRIC (score 1-10, REQUIRED, NEVER default to 5):
  9-10 → Whole/minimally processed. Sugar <2g/100g, Sodium <200mg/100g.
  7-8  → Moderately processed. Sugar <5g, Sodium <400mg.
  5-6  → Processed. Sugar 5-15g OR Sodium 400-700mg.
  3-4  → High sugar (>15g) OR high sodium (>700mg) OR poor nutritional profile.
  1-2  → Ultra-processed (NOVA 4) OR very high sugar/sodium/saturated fat.
  HARD CAPS: NOVA 4 → max score 4. Sodium >1000mg/100g → max score 5.

[LABEL TEXT]:
{safe_label_text}

Return ONLY this single JSON object (no markdown, no extra text):
{{
  "product_name": "string — REQUIRED, infer from context, NEVER 'Unknown Product'",
  "product_category": "Snack|Dairy|Beverage|Cereal|Noodle|Biscuit|Supplement|Spice|Oil|Sauce|Salt|Cheese|Nuts|Bread|Chocolate|Candy|Meat|Seafood|Fruit|Vegetable|Other",
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
  "ingredients_raw": "full ingredients text or empty string",
  "nutrients": [
    {{"name": "ENERGY", "value": 384.0, "unit": "kcal", "rating": "caution", "impact": "High energy density."}},
    {{"name": "PROTEIN", "value": 8.2, "unit": "g", "rating": "moderate", "impact": "Decent protein."}},
    {{"name": "TOTAL CARBOHYDRATE", "value": 59.6, "unit": "g", "rating": "moderate", "impact": "Moderate carbs."}},
    {{"name": "  of which SUGARS", "value": 1.8, "unit": "g", "rating": "good", "impact": "Low sugar."}},
    {{"name": "  of which DIETARY FIBRE", "value": 2.0, "unit": "g", "rating": "moderate", "impact": "Some fiber."}},
    {{"name": "TOTAL FAT", "value": 12.5, "unit": "g", "rating": "moderate", "impact": "Moderate fat."}},
    {{"name": "  of which SATURATES", "value": 8.2, "unit": "g", "rating": "caution", "impact": "High sat fat."}},
    {{"name": "  of which TRANS FAT", "value": 0.13, "unit": "g", "rating": "caution", "impact": "Trace trans fat."}},
    {{"name": "SODIUM", "value": 1000.0, "unit": "mg", "rating": "bad", "impact": "Dangerously high sodium."}}
  ],
  "better_alternative": "string — suggest a healthier alternative product or category",
  "score": <integer 1-10, REQUIRED — use scoring rubric above>,
  "verdict": "<Two-word verdict in {lang_name}>",
  "summary": "<2-sentence professional summary in {lang_name}>",
  "eli5_explanation": "<Child-friendly 1-sentence with emoji in {lang_name}>",
  "pros": ["<benefit 1>", "<benefit 2>", "<benefit 3>"],
  "cons": ["<concern 1>", "<concern 2>"],
  "age_warnings": [
    {{"group": "Children (under 12)", "emoji": "👶", "status": "warning|caution|good", "message": ""}},
    {{"group": "Adults (18-60)", "emoji": "🧑", "status": "warning|caution|good", "message": ""}},
    {{"group": "Seniors (60+)", "emoji": "👴", "status": "warning|caution|good", "message": ""}},
    {{"group": "Diabetics", "emoji": "🩸", "status": "warning|caution|good", "message": ""}},
    {{"group": "Pregnant", "emoji": "🤰", "status": "warning|caution|good", "message": ""}}
  ],
  "molecular_insight": "<1 sentence on biochemical impact in {lang_name}>",
  "chart_data": [<Safe%>, <Moderate%>, <Risky%>],
  "ingredients_spotlight": [
    {{"name": "<ingredient>", "type": "natural|additive|preservative|emulsifier|vitamin|seasoning", "safety_rating": "safe|moderate|concern", "what_it_is": "<one sentence>", "health_impact": "<one sentence>", "curiosity_fact": "<interesting fact>"}}
  ]
}}
- nutrients array: include EVERY row visible in the label text — no skipping.
  Add "rating" (good|moderate|caution|bad) and "impact" on EACH nutrient entry.
  Nutrient names MUST be in CAPITALS (e.g. "ENERGY", "TOTAL FAT", "OF WHICH SUGARS").
- product_name: REQUIRED. Infer if not explicit. NEVER output "Unknown Product".
- better_alternative: REQUIRED. Suggest a 100% healthier alternative (e.g. "Try 100% Whole Wheat Oats" instead of "Maida Noodles").
- score MUST reflect actual nutritional values using the rubric above. NEVER default to 5.
- chart_data: [Safe%, Moderate%, Risky%] must sum to exactly 100.
- ingredients_spotlight: TOP 8 notable ingredients. NEVER return empty if ingredients exist.
- verdict, summary, eli5_explanation: MUST be in {lang_name}.
"""


# ── Fallback scoring ───────────────────────────────────────────────────
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
    """Ask AI if the OCR text looks like a nutrition label — final safety net."""
    if not raw_text or len(raw_text) < 20:
        return {"is_valid": False, "clean_text": ""}
    sample = raw_text[:1000]
    prompt = f"""
The following was OCR-extracted from a food product image.
Does it contain a nutrition table or ingredient list?

TEXT:
{sample}

Rules:
1. VALID if you see ANY of: Fat, Energy, Calories, Sugar, Protein, Serving, Saturated,
   Carbohydrate, Sodium, or a list of chemical/food ingredient names.
2. INVALID only if it is PURELY marketing text with zero numbers or nutrient words.
3. When in doubt → respond VALID.

Return ONLY:
{{"status": "VALID" | "INVALID", "cleaned_text": "<relevant text or empty>"}}
"""
    try:
        raw = await asyncio.to_thread(call_llm, prompt, 800)
        res = parse_llm_response(raw)
        return {
            "is_valid": res.get("status") == "VALID",
            "clean_text": res.get("cleaned_text") or raw_text
        }
    except Exception as e:
        logger.error("AI recovery failed: %s", e)
        return {"is_valid": False, "clean_text": ""}


# ── Master pipeline (single-pass) ────────────────────────────────────────
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
    image_content: bytes = None,
) -> dict:
    """
    Single-pass pipeline — ONE LLM call does extraction + analysis together.
    Steps:
      1. Label filter / AI recovery gate
      2. ONE super-prompt call → nutrients + score + pros/cons/age warnings
      3. Atwater physics validation (retry if mismatch)
      4. DNA overrides (NOVA 4, fake claims)
      5. Explanation engine (humanized insights)
      6. Rule-based rating fallback for every nutrient card
      7. Assemble final output
    """

    # Step 1: fallback OCR if image passed directly
    if image_content and not extracted_text:
        from app.services.ocr import run_ocr
        logger.warning("image_content passed — OCR should be done in caller.")
        cropped = process_image_for_ocr(image_content)
        ocr_res = run_ocr(cropped, language)
        extracted_text = ocr_res["text"]

    # Step 2: Cache check
    cache_key = hashlib.md5(
        f"v7:{extracted_text[:120]}:{persona}:{language}".encode()
    ).hexdigest()
    cached = get_ai_cache(cache_key)
    if cached:
        cached["scan_meta"] = {"cached": True, "scans_remaining": 0, "is_pro": False}
        return cached

    # Step 3: Label filter
    from app.services.ocr import universal_label_filter, run_ocr
    filter_result = universal_label_filter(extracted_text)

    # Fallback: try full image OCR
    if not filter_result["is_valid"] and image_content:
        logger.info("Crop OCR failed — retrying on full image...")
        full_ocr = run_ocr(image_content, language)
        ff = universal_label_filter(full_ocr["text"])
        if ff["is_valid"]:
            filter_result = ff
            extracted_text = full_ocr["text"]
        else:
            ai_rec = await recover_label_with_ai(full_ocr["text"])
            if ai_rec["is_valid"]:
                filter_result = {"is_valid": True, "clean_text": ai_rec["clean_text"] or full_ocr["text"]}
                extracted_text = filter_result["clean_text"]

    # Fallback: AI recovery on already-extracted text
    if not filter_result["is_valid"] and extracted_text and len(extracted_text) > 30:
        ai_rec2 = await recover_label_with_ai(extracted_text)
        if ai_rec2["is_valid"]:
            filter_result = {"is_valid": True, "clean_text": ai_rec2["clean_text"] or extracted_text}

    if not filter_result["is_valid"]:
        return {
            "error": "no_label",
            "message": "⚠️ No nutrition table detected. Please photograph the back of the package.",
        }

    clean_text = filter_result["clean_text"]

    # Step 4: Optional live web research (non-blocking, fast timeout)
    # BUG FIX: Build a targeted search query from product keywords, not raw OCR text
    internal_web_context = ""
    try:
        from app.services.research_engine import get_live_search
        # Extract the most nutrition-relevant words for a targeted search
        _search_words = [w for w in clean_text.split() if len(w) >= 4 and w.isalpha()][:8]
        _search_query = f"nutrition health analysis {' '.join(_search_words[:5])}" if _search_words else f"food label nutrition {clean_text[:40]}"
        # Run search asynchronously without blocking the main flow
        internal_web_context = await asyncio.wait_for(
            asyncio.to_thread(get_live_search, _search_query),
            timeout=3.0  # Hard 3-second cap so it never slows down the response
        )
    except Exception as e:
        logger.debug("Web research skipped: %s", e)

    # Step 5: ── SINGLE SUPER-PROMPT (extract + analyze in ONE call) ──────
    super_prompt = build_super_prompt(
        label_text=clean_text,
        persona=persona,
        language=language,
        blur_info=blur_info,
        dna_flags=[],
        nova_level=1,  # Will be recalculated after extraction
        research_context=internal_web_context or web_context,
    )

    try:
        raw_response = await asyncio.to_thread(call_llm, super_prompt, 4000)
        result_data = parse_llm_response(raw_response)
    except Exception as e:
        logger.error("Super-prompt LLM failed: %s", e)
        err_msg = str(e).lower()
        if "rate limit" in err_msg or "429" in err_msg:
            return {"error": "server_busy", "message": "⚠️ High traffic on AI servers. Please try again in a few minutes."}
        if "payload too large" in err_msg or "413" in err_msg:
             return {"error": "impossible_data", "message": "⚠️ Label text too complex to analyze. Please try a closer crop."}
        return {"error": "server_busy", "message": "⚠️ Analysis failed. Please try again."}

    # Step 6: Atwater physics check on extracted macros
    def _float(val):
        if not val: return 0.0
        try: return float(str(val).replace(',', '.'))
        except: return 0.0

    def _primary(d):
        base_cal = _float(d.get("calories")) or _float(d.get("energy")) or _float(d.get("kcal")) or _float(d.get("energy_kcal"))
        kj = _float(d.get("energy_kj"))
        if not base_cal and kj: base_cal = kj / 4.184
        return {
            "calories":      base_cal,
            "protein":       _float(d.get("protein")),
            "carbs":         _float(d.get("carbs")) or _float(d.get("carbohydrate")),
            "fat":           _float(d.get("fat")) or _float(d.get("total_fat")),
            "sugar":         _float(d.get("sugar")) or _float(d.get("total_sugars")),
            "fiber":         _float(d.get("fiber")) or _float(d.get("dietary_fiber")),
            "saturated_fat": _float(d.get("saturated_fat")) or _float(d.get("saturated")),
        }

    category = result_data.get("product_category") or product_category_hint or "unknown"
    rich = _primary(result_data)
    rich["sodium"]      = _float(result_data.get("sodium_mg"))
    rich["trans_fat"]   = _float(result_data.get("trans_fat"))
    rich["cholesterol"] = _float(result_data.get("cholesterol_mg"))
    rich["potassium"]   = _float(result_data.get("potassium_mg"))
    rich["calcium"]     = _float(result_data.get("calcium_mg"))
    rich["iron"]        = _float(result_data.get("iron_mg"))

    math_ok = atwater_math_check(_primary(result_data), category)
    if not math_ok["is_valid"]:
        # One retry with explicit correction hint
        logger.warning("Atwater mismatch: %s — retrying with correction.", math_ok["reason"])
        correction = (
            f"\n\nPREVIOUS EXTRACTION ERROR: {math_ok['reason']}\n"
            "Re-read the label. Ensure sugar/fiber are PARTS of carbs (not extra). "
            "Saturated fat is PART of total fat. Do NOT add them separately."
        )
        try:
            retry_prompt = build_super_prompt(
                label_text=clean_text + correction,
                persona=persona,
                language=language,
                blur_info=blur_info,
                dna_flags=[],
                nova_level=1,
                research_context=internal_web_context or web_context,
            )
            raw2 = await asyncio.to_thread(call_llm, retry_prompt, 4000)
            result_data = parse_llm_response(raw2)
            category = result_data.get("product_category") or category
            rich = _primary(result_data)
            rich["sodium"]      = float(result_data.get("sodium_mg")      or 0)
            rich["trans_fat"]   = float(result_data.get("trans_fat")      or 0)
            rich["cholesterol"] = float(result_data.get("cholesterol_mg") or 0)
            rich["potassium"]   = float(result_data.get("potassium_mg")   or 0)
            rich["calcium"]     = float(result_data.get("calcium_mg")     or 0)
            rich["iron"]        = float(result_data.get("iron_mg")        or 0)
        except Exception as retry_err:
            logger.error("Retry failed: %s", retry_err)

    ingredients_raw = result_data.get("ingredients_raw", "") or ""

    # Step 7: DNA overrides
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

    # Step 8: Explanation engine (NOVA, RDA%, humanized insights)
    explanation = get_explanation_report(rich, ingredients_raw)
    nova_level  = explanation["nova_level"]

    # Step 9: Build final nutrient_breakdown — SPECIALIST DYNAMIC CARDS
    # We generate cards ONLY for nutrients actually LISTED on the label.
    # No fixed templates, no forced priority fields (per Specialist instructions).
    llm_list = result_data.get("nutrients", [])
    if not isinstance(llm_list, list) or not llm_list:
        # Emergency Fallback ONLY if LLM returned nothing
        _backup = [
            ("ENERGY", "calories", "kcal"), ("PROTEIN", "protein", "g"), 
            ("CARBS", "carbs", "g"), ("FAT", "fat", "g")
        ]
        llm_list = []
        for label, key, unit in _backup:
            val = result_data.get(key)
            if val is not None and float(val or 0) >= 0:
                llm_list.append({"name": label, "value": float(val), "unit": unit})

    # Normalise values (strip embedded unit strings) + title-case names
    def _title_nutrient(raw_name: str) -> str:
        """Convert 'TOTAL FAT' or '  of which SUGARS' to 'Total Fat' / '  of which Sugars'."""
        stripped = raw_name.strip()
        prefix = "  of which " if raw_name.lower().lstrip().startswith("of which") else ""
        core = stripped
        if prefix:
            core = re.sub(r"(?i)of\s+which\s+", "", core).strip()
        return (prefix + core.title()).rstrip()

    nutrient_breakdown = []
    for n in llm_list:
        raw_val = n.get("value", 0)
        if isinstance(raw_val, str):
            m = re.search(r"[\d]+\.?[\d]*", raw_val.replace(",", "."))
            raw_val = float(m.group()) if m else 0.0
        nutrient_breakdown.append({
            "name":   _title_nutrient(n.get("name", "?")),
            "value":  round(float(raw_val or 0), 2),
            "unit":   n.get("unit", ""),
            "rating": n.get("rating", ""),
            "impact": n.get("impact", ""),
        })

    # Guarantee every card has a rating — rule-based fills any gap
    for n in nutrient_breakdown:
        if not n.get("rating"):
            r = _rule_rate(n["name"], float(n.get("value") or 0), n.get("unit", ""))
            n["rating"] = r["rating"]
            n["impact"]  = r["impact"]

    # Step 10: Final score
    # BUG FIX: dna_res uses key "score" not "base_score" — was always defaulting to 4
    dna_flags = dna_res.get("extra_flags", [])
    if dna_res["action"] == "OVERRIDE":
        final_score = dna_res.get("score") or dna_res.get("base_score") or 4
        dna_flags   = [dna_res.get("reason", "")] + dna_flags
    elif result_data.get("score"):
        try:
            final_score = max(1, min(10, int(result_data["score"])))
        except (ValueError, TypeError):
            final_score = compute_rule_based_score(rich, nova_level)
        if nova_level == 4 and final_score > 4:
            final_score = 4
    else:
        final_score = compute_rule_based_score(rich, nova_level)
    # Safety clamp — score must always be 1-10
    final_score = max(1, min(10, int(final_score or 1)))

    # Step 11: Assemble output
    # BUG FIX: Better product name fallback — never show "Unknown Product" to users.
    # If the LLM couldn't extract a name, try to infer one from available data.
    product_name = result_data.get("product_name") or ""
    _bad_names = {"unknown", "unknown product", "", "n/a", "food product", "product"}
    if product_name.lower().strip() in _bad_names:
        # Try to build a name from category + ingredients hint
        cat = (result_data.get("product_category") or "").capitalize()
        ings = (result_data.get("ingredients_raw") or "").split(",")
        first_ing = ings[0].strip().title() if ings and ings[0].strip() else ""
        if cat and cat.lower() not in ("other", ""):
            product_name = f"{cat} Product"
        elif first_ing:
            product_name = f"{first_ing}-based Product"
        else:
            product_name = "Food Product"

    verdict     = result_data.get("verdict")  or dna_res.get("reason") or "Analyzed"
    summary     = result_data.get("summary")  or dna_res.get("reason") or ""
    pros        = result_data.get("pros", [])
    cons_llm    = result_data.get("cons", [])
    cons        = dna_flags + [c for c in cons_llm if c not in dna_flags]
    eli5        = result_data.get("eli5_explanation") or result_data.get("eli5", "")
    mol_insight = result_data.get("molecular_insight", "")
    score_color = "#22c55e" if final_score >= 7 else "#f59e0b" if final_score >= 4 else "#ef4444"

    # Merge age warnings (AI + physics engine)
    age_warnings   = result_data.get("age_warnings", [])
    phys_warnings  = explanation.get("persona_warnings", [])
    merged_warnings = {w["group"].lower(): w for w in age_warnings}
    for pw in phys_warnings:
        key = pw["persona"].lower()
        if key in merged_warnings:
            merged_warnings[key]["message"] = f"{merged_warnings[key]['message']}. {pw['msg']}"
            if pw["type"] == "WARNING":
                merged_warnings[key]["status"] = "warning"
        else:
            merged_warnings[key] = {
                "group": pw["persona"], "status": pw["type"].lower(),
                "message": pw["msg"], "emoji": "⚠️"
            }

    # chart_data normalisation
    chart_data = result_data.get("chart_data", [50, 30, 20])
    if len(chart_data) == 3 and sum(chart_data) != 100 and sum(chart_data) > 0:
        t = sum(chart_data)
        chart_data = [round(v * 100 / t) for v in chart_data]

    # Ingredient spotlight — fallback if LLM skipped it
    ingredients_spotlight = result_data.get("ingredients_spotlight", [])
    if not ingredients_spotlight and ingredients_raw:
        ing_list = [i.strip() for i in re.split(r"[,;]", ingredients_raw) if i.strip()][:8]
        for ing in ing_list:
            if len(ing) > 2:
                ingredients_spotlight.append({
                    "name": ing.title(), "type": "natural", "safety_rating": "safe",
                    "what_it_is": f"{ing.title()} is a food ingredient.",
                    "health_impact": "Part of the product formulation.",
                    "curiosity_fact": "Check the full ingredients list for details."
                })

    final_output = {
        "product_name":          product_name,
        "product_category":      category,
        "serving_size":          result_data.get("serving_size"),
        "score":                 final_score,
        "score_color":           score_color,
        "verdict":               verdict,
        "summary":               summary,
        "nutrient_breakdown":    nutrient_breakdown,
        "pros":                  pros,
        "cons":                  cons,
        "age_warnings":          list(merged_warnings.values()),
        "eli5_explanation":      eli5,
        "molecular_insight":     mol_insight,
        "chart_data":            chart_data,
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


# ── Legacy shim ────────────────────────────────────────────────────────
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
            conn.execute("SELECT id FROM food_products WHERE barcode=?", (barcode,)).fetchone()
            if barcode else
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
