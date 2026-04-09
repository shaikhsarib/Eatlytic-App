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
You are a precise nutrition label reader. Scan the label text and extract
EVERY single nutrient listed — not just the common ones.

STRICT RULES:
1. Extract the "Per 100g" column only. Skip "Per Serve" and "% RDA" columns.
2. Include ALL rows: Protein, Carbohydrate, Sugar, Added Sugar, Fiber,
   Fat, Saturated Fat, Trans Fat, Cholesterol, Sodium, Potassium,
   Calcium, Iron, Vitamins A/C/D/B12, Moisture, Ash — EVERYTHING on the label.
3. Output EXACT numbers from the label. No guessing or inventing.
4. If a nutrient is missing from the label, omit it from "nutrients" array.
5. Sub-components get an indent prefix "  of which" in their name.

Return ONLY this JSON (no markdown, no extra text):
{
  "product_name": "string",
  "product_category": "Snack|Dairy|Beverage|Cereal|Noodle|Biscuit|Supplement|Spice|Oil|Sauce|Other",
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
    persona, language, nova_level, dna_flags,
) -> str:
    lang_name = LANGUAGE_MAP.get(language, "English")
    nut_text = "\n".join(
        f"  {n['name']}: {n['value']} {n['unit']}" for n in nutrients_list
    )
    flags_text = "\n".join(f"  - {f}" for f in dna_flags) if dna_flags else "  None"

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

SCORING RUBRIC — assign the EXACT right score, NEVER default to 5, 6, or 7:
  9-10 → Whole food. Sugar <2g/100g, sodium <200mg, high protein or fiber.
  7-8  → Mildly processed. Sugar <5g, sodium <400mg, decent macros.
  5-6  → Processed. Sugar 5-15g OR sodium 400-700mg.
  3-4  → High sugar >15g OR sodium >700mg OR poor fat profile.
  1-2  → Ultra-processed (NOVA 4) OR very high sugar/sodium/sat-fat.
  HARD CAPS: NOVA 4 → max score 4. Sodium >1000mg → max score 5.

Return ONLY this JSON (no markdown):
{{
  "score": <integer 1-10, REQUIRED>,
  "verdict": "<Two-word verdict in {lang_name}>",
  "summary": "<2-sentence professional summary in {lang_name}>",
  "eli5": "<Child-friendly 1-sentence with one emoji in {lang_name}>",
  "pros": ["<Genuine benefit 1>", "<Genuine benefit 2>", "<Genuine benefit 3>"],
  "cons": ["<Health concern 1>", "<Health concern 2>"],
  "age_warnings": [
    {{"group": "Children (under 12)", "emoji": "👶", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Adults (18-60)",       "emoji": "🧑", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Seniors (60+)",        "emoji": "👴", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Diabetics",            "emoji": "🩸", "status": "warning|caution|good", "message": "<in {lang_name}>"}},
    {{"group": "Pregnant",             "emoji": "🤰", "status": "warning|caution|good", "message": "<in {lang_name}>"}}
  ],
  "molecular_insight": "<1 sentence on biochemical impact in {lang_name}>"
}}"""


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


# ── Master pipeline ────────────────────────────────────────────────────
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

    # Step 1: Image pipeline (optional) ──────────────────────────────
    if image_content:
        from app.services.ocr import run_ocr
        logger.info("Running intelligent image pipeline...")
        cropped = process_image_for_ocr(image_content)
        ocr_res = run_ocr(cropped, language)
        extracted_text = ocr_res["text"]

    # Step 2: Cache check ─────────────────────────────────────────────
    cache_key = hashlib.md5(
        f"v5:{extracted_text[:120]}:{persona}:{language}".encode()
    ).hexdigest()
    cached = get_ai_cache(cache_key)
    if cached:
        cached["scan_meta"] = {"cached": True, "scans_remaining": 0, "is_pro": False}
        return cached

    # Step 3: Label filter & Fallback ──────────────────────────────────
    from app.services.ocr import universal_label_filter
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
            logger.warning("Full Image fallback also failed.")

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

    category = extracted.get("product_category") or product_category_hint or "unknown"

    # Step 5: Atwater check + retry ───────────────────────────────────
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

    math_ok = atwater_math_check(_primary(extracted), category)
    if not math_ok["is_valid"]:
        logger.warning("Math mismatch, retrying: %s", math_ok["reason"])
        correction = (
            f"\n\nERROR IN PREVIOUS EXTRACTION: {math_ok['reason']}\n"
            "Re-read the label. Sugar/Fiber are sub-sets of Carbs; "
            "Sat-Fat is a sub-set of Fat. Do NOT double-count."
        )
        try:
            extracted = await _extract(clean_text, correction)
            category = extracted.get("product_category") or category
        except Exception as re_err:
            logger.error("Retry extraction failed: %s", re_err)

    # Step 6: Build rich nutrients dict ───────────────────────────────
    rich = {
        "calories":      float(extracted.get("calories")      or 0),
        "protein":       float(extracted.get("protein")       or 0),
        "carbs":         float(extracted.get("carbs")          or 0),
        "fat":           float(extracted.get("fat")            or 0),
        "sugar":         float(extracted.get("sugar")          or 0),
        "fiber":         float(extracted.get("fiber")          or 0),
        "sodium":        float(extracted.get("sodium_mg")      or 0),
        "saturated_fat": float(extracted.get("saturated_fat") or 0),
        "trans_fat":     float(extracted.get("trans_fat")     or 0),
        "cholesterol":   float(extracted.get("cholesterol_mg")or 0),
        "potassium":     float(extracted.get("potassium_mg")  or 0),
        "calcium":       float(extracted.get("calcium_mg")    or 0),
        "iron":          float(extracted.get("iron_mg")       or 0),
    }
    ingredients_raw = extracted.get("ingredients_raw", "") or ""

    # Step 7: DNA overrides ───────────────────────────────────────────
    dna_res = apply_dna_overrides(
        full_ocr_text=extracted_text,
        nutrients=rich,
        ingredients_raw=ingredients_raw,
        base_score=5,
        category=category,
        front_text=front_text,
    )

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
    )

    analysis = {}
    try:
        raw_analysis = await asyncio.to_thread(call_llm, analysis_prompt, 2000)
        analysis = parse_llm_response(raw_analysis)
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
    product_name = extracted.get("product_name", "Unknown Product")
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

    eli5         = analysis.get("eli5", "")
    mol_insight  = analysis.get("molecular_insight", "")
    score_color  = "#22c55e" if final_score >= 7 else "#f59e0b" if final_score >= 4 else "#ef4444"

    final_output = {
        "product_name":      product_name,
        "product_category":  category,
        "serving_size":      extracted.get("serving_size"),
        "score":             final_score,
        "score_color":       score_color,
        "verdict":           verdict,
        "summary":           summary,
        "nutrient_breakdown": nutrient_breakdown,
        "pros":              pros,
        "cons":              cons,
        "age_warnings":      age_warnings,
        "eli5":              eli5,
        "molecular_insight": mol_insight,
        "ingredients_raw":   ingredients_raw,
        "explanation":       explanation,
        "better_alternative": get_healthy_alternative(category, persona),
        "whatsapp_content":  {},
        "disclaimer":        MEDICAL_DISCLAIMER,
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
