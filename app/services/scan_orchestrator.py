"""
app/services/scan_orchestrator.py
─────────────────────────────────────────────────────────────────────────────
The Scan Orchestration Service of Eatlytic.
Integrates OCR, LLM fallbacks, database caching, local verified lookups,
Atwater physics checks, DNA rules overrides, and persona-based scoring.
Moved to Services layer to enforce Onion Architecture rules.
"""

import os
import re
import time
import logging
import asyncio
import hashlib
from typing import Dict, List, Tuple

from app.database.connection import get_ai_cache, set_ai_cache, db_conn
from app.services.fake_detector import apply_dna_overrides, atwater_math_check
from app.services.alternatives import get_healthy_alternative
from app.services.explanation_engine import get_explanation_report, adjust_score_for_persona, verify_atwater_math
from app.services.formatter import get_whatsapp_tiered_content
from app.ai.llm.prompts import build_super_prompt, MEDICAL_DISCLAIMER
from app.ai.llm.validators import _rule_rate, compute_rule_based_score, compute_extraction_confidence

logger = logging.getLogger(__name__)

def normalize_group_name(name: str) -> str:
    if not name:
        return ""
    n = name.lower()
    if "child" in n:
        return "children"
    if "diab" in n:
        return "diabetics"
    if "senior" in n or "60+" in n:
        return "seniors"
    if "preg" in n or "matern" in n:
        return "pregnant"
    if "heart" in n:
        return "heart"
    if "athlet" in n or "fit" in n or "gym" in n:
        return "athletes"
    return n

def build_merged_warnings(age_warnings: list, phys_warnings: list) -> list:
    merged = {}
    
    # 1. Add age warnings from LLM first (these already have nice emojis from LLM)
    for w in age_warnings:
        if not isinstance(w, dict) or "group" not in w:
            continue
        norm_key = normalize_group_name(w["group"])
        merged[norm_key] = {
            "group": w["group"],
            "status": w.get("status", "good").lower(),
            "message": w.get("message", ""),
            "emoji": w.get("emoji", "⚠️")
        }
        
    # 2. Add or merge physical advice from the explanation engine
    for pw in phys_warnings:
        if not isinstance(pw, dict) or "persona" not in pw:
            continue
        norm_key = normalize_group_name(pw["persona"])
        pw_msg = pw.get("msg", "")
        pw_type = pw.get("type", "caution").lower()
        
        if norm_key in merged:
            existing = merged[norm_key]
            existing_msg = existing.get("message", "")
            # Prevent double appending
            if pw_msg and pw_msg not in existing_msg:
                if existing_msg:
                    existing["message"] = f"{existing_msg}. {pw_msg}"
                else:
                    existing["message"] = pw_msg
            # Update status if the new warning is more critical
            if pw_type == "warning":
                existing["status"] = "warning"
            elif pw_type == "caution" and existing.get("status") not in ["warning", "avoid"]:
                existing["status"] = "caution"
        else:
            emoji_map = {
                "children": "👶",
                "diabetics": "🩸",
                "seniors": "👴",
                "pregnant": "🤰",
                "athletes": "💪",
                "heart": "❤️"
            }
            emoji = emoji_map.get(norm_key, "⚠️")
            merged[norm_key] = {
                "group": pw["persona"],
                "status": pw_type,
                "message": pw_msg,
                "emoji": emoji
            }
            
    return list(merged.values())

def find_db_product_match(extracted_text: str) -> dict:
    """
    Looks up a food product in local database based on OCR text keywords or barcode.
    """
    if not extracted_text or len(extracted_text.strip()) < 10:
        return None
        
    lower_text = extracted_text.lower()
    
    # 1. Barcode check (regex matches 8 to 13 digits)
    barcodes = re.findall(r"\b\d{8,13}\b", extracted_text)
    if barcodes:
        with db_conn() as conn:
            for bc in barcodes:
                row = conn.execute("""
                    SELECT fp.*, fpe.sat_fat_100g, fpe.ingredients_raw, fpe.source
                    FROM food_products fp
                    LEFT JOIN food_products_extra fpe ON fp.id = fpe.id
                    WHERE fp.barcode = ? AND fp.verified = 1
                """, (bc,)).fetchone()
                if row:
                    return dict(row)
                    
    # 2. Text Keyword brand + name check — push filtering to SQLite
    words = list({w for w in re.split(r"\W+", lower_text) if len(w) >= 3})
    if not words:
        return None

    like_clauses = " OR ".join(["fp.name LIKE ? OR fp.brand LIKE ?"] * min(len(words), 8))
    like_params = []
    for w in words[:8]:
        like_params.extend([f"%{w}%", f"%{w}%"])
    like_params.append(50)  # LIMIT

    with db_conn() as conn:
        rows = conn.execute(f"""
            SELECT fp.*, fpe.sat_fat_100g, fpe.ingredients_raw, fpe.source
            FROM food_products fp
            LEFT JOIN food_products_extra fpe ON fp.id = fpe.id
            WHERE fp.verified = 1 AND ({like_clauses})
            LIMIT ?
        """, like_params).fetchall()

    best_match = None
    best_score = 0

    for row in rows:
        prod = dict(row)
        brand = (prod.get("brand") or "").lower().strip()
        name = (prod.get("name") or "").lower().strip()

        if not brand or not name:
            continue

        # Brand must be present in the OCR text
        if brand not in lower_text:
            continue

        # Split product name into significant words
        name_words = [w for w in re.split(r"\W+", name) if len(w) > 2]
        if not name_words:
            if name in lower_text:
                score = 10.0
                if score > best_score:
                    best_score = score
                    best_match = prod
            continue

        # Count matches
        matches = sum(1 for w in name_words if w in lower_text)
        match_ratio = matches / len(name_words)

        # At least 70% matching words to qualify
        if match_ratio >= 0.7:
            score = match_ratio * 10 + matches
            if score > best_score:
                best_score = score
                best_match = prod

    return best_match

def build_offline_match_response(db_match: dict, persona: str, language: str, t_pipeline_start: float, device_key: str = None) -> dict:
    """
    Constructs the exact same response schema as the LLM-based unified_analyze_flow
    using local verified product data and offline calculations.
    """
    logger.info("Building offline response for verified match: %s (%s)", db_match["name"], db_match["brand"])
    from app.services.brain import EatlyticBrain
    brain = EatlyticBrain()
    
    nutrients = {
        "calories": float(db_match.get("calories_100g") or 0.0),
        "protein": float(db_match.get("protein_100g") or 0.0),
        "carbs": float(db_match.get("carbs_100g") or 0.0),
        "fat": float(db_match.get("fat_100g") or 0.0),
        "sugar": float(db_match.get("sugar_100g") or 0.0),
        "fiber": float(db_match.get("fiber_100g") or 0.0),
        "saturated_fat": float(db_match.get("sat_fat_100g") or 0.0),
        "sodium": float(db_match.get("sodium_100g") or 0.0),
        "trans_fat": 0.0,
        "cholesterol": 0.0,
    }
    ingredients_raw = db_match.get("ingredients_raw") or ""
    
    final_output = brain.compile_local_report(
        product_name=db_match["name"],
        brand=db_match["brand"],
        category=db_match.get("category") or "unknown",
        nutrients=nutrients,
        ingredients_raw=ingredients_raw,
        persona=persona,
        eatlytic_score=int(db_match.get("eatlytic_score")) if db_match.get("eatlytic_score") is not None else None,
        device_key=device_key
    )
    
    latency_ms = int((time.time() - t_pipeline_start) * 1000)
    final_output["perf_metrics"] = {"latency_ms": latency_ms}
    final_output["disclaimer"] = MEDICAL_DISCLAIMER
    
    try:
        final_output["whatsapp_content"] = get_whatsapp_tiered_content(final_output)
    except:
        pass
        
    return final_output

async def unified_analyze_flow(
    extracted_text: str,
    persona: str = "adult",
    age_group: str = "adult",
    product_category_hint: str = "unknown",
    language: str = "en",
    web_context: str = "",
    blur_info: dict = {},
    label_confidence: str = "high",
    front_text: str = "",
    image_content: bytes = None,
    device_key: str = None,
) -> dict:
    """THE CORE ENGINE: Coordinates OCR, LLM, DNA validation, and persona scoring."""
    t_pipeline_start = time.time()
    extracted_text = extracted_text or ""
    from app.ai.llm.engine import recover_label_with_ai
    import app.ai.llm.engine as llm_engine

    from app.services.label_classifier import classify_label_type
    label_type = classify_label_type(extracted_text)
    if label_type["is_non_food"]:
        return {
            "error": "non_food_label",
            "message": (
                f"⚠️ This looks like a {label_type['detected_type']} label, not a food label. "
                "Eatlytic analyzes food & beverage nutrition labels. "
                "Please photograph the nutrition facts panel on a food or drink product."
            ),
            "detected_type": label_type["detected_type"],
        }

    # ENTRY POINT A: DB-First Hybrid Lookup on initial OCR text
    db_match = find_db_product_match(extracted_text)
    if db_match:
        return build_offline_match_response(db_match, persona, language, t_pipeline_start, device_key)

    cache_key = hashlib.md5(
        f"v7:{extracted_text[:500]}:{persona}:{language}:{blur_info.get('score',0)}".encode()
    ).hexdigest()
    cached = get_ai_cache(cache_key)
    cached_tier = (cached or {}).get("extraction_confidence", {}).get("tier", "")
    if cached and "error" not in cached and cached_tier != "UNRELIABLE" and extracted_text:
        cached["scan_meta"] = {"cached": True, "scans_remaining": 0, "is_pro": False}
        return cached

    from app.ai.ocr.client import universal_label_filter, run_ocr
    filter_result = universal_label_filter(extracted_text)

    if extracted_text and len(extracted_text.strip()) > 20:
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

        if not filter_result["is_valid"] and extracted_text and len(extracted_text) > 30:
            ai_rec2 = await recover_label_with_ai(extracted_text)
            if ai_rec2["is_valid"]:
                filter_result = {"is_valid": True, "clean_text": ai_rec2["clean_text"] or extracted_text}
    else:
        logger.info("No usable OCR text — skipping retries, going straight to Vision Fallback.")

    if not filter_result["is_valid"]:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        together_key = os.environ.get("TOGETHER_API_KEY", "")
        vision_text = ""

        if image_content and groq_key and not vision_text:
            import base64, requests as _req
            GROQ_VISION_MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct"]
            b64_img = base64.b64encode(image_content).decode("utf-8")
            vision_prompt = (
                "Extract ALL text from this food product label image. Focus on the nutrition facts table, "
                "ingredients list, and any nutritional information. "
                "Return ONLY the raw text as it appears — preserve all numbers, units (g, mg, kcal, kJ), "
                "and row labels. Do NOT summarize or interpret."
            )

            def _call_groq_vision(model: str) -> str:
                resp = _req.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        ]}],
                        "max_tokens": 1500,
                        "temperature": 0.0,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"] or ""
                return ""

            for vision_model in GROQ_VISION_MODELS:
                try:
                    vision_text = await asyncio.to_thread(_call_groq_vision, vision_model)
                    if vision_text and len(vision_text) > 50: break
                except: vision_text = ""

        if image_content and together_key and not vision_text:
            try:
                import base64, requests as _req
                b64_img = base64.b64encode(image_content).decode("utf-8")
                vision_resp = _req.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={"Authorization": f"Bearer {together_key}", "Content-Type": "application/json"},
                    json={
                        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": "Extract all text from this food nutrition label image. Return only raw text with numbers and units."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        ]}],
                        "max_tokens": 1500, "temperature": 0.0,
                    },
                    timeout=30,
                )
                if vision_resp.status_code == 200:
                    vision_text = vision_resp.json()["choices"][0]["message"]["content"] or ""
            except: pass

        if vision_text and len(vision_text) > 50:
            vision_filter = universal_label_filter(vision_text)
            if vision_filter["is_valid"] or len(vision_text) > 100:
                filter_result = vision_filter if vision_filter["is_valid"] else {"is_valid": True, "clean_text": vision_text}
                extracted_text = vision_text

    if not filter_result["is_valid"]:
        return {"error": "no_label", "message": "⚠️ No nutrition table detected. Please photograph the back of the package."}

    clean_text = filter_result["clean_text"]
    
    # ENTRY POINT B: DB-First Hybrid Lookup on enhanced/recovered OCR or vision text
    db_match = find_db_product_match(clean_text)
    if db_match:
        return build_offline_match_response(db_match, persona, language, t_pipeline_start, device_key)

    internal_web_context = ""
    try:
        from app.services.research_engine import get_live_search
        _relevant = [w for w in clean_text.split() if len(w) >= 4 and w.isalpha()]
        _brand_hint = next((w for w in _relevant if w.lower() in ["cadbury", "nestle", "amul", "britannia", "unilever"]), "")
        _search_query = f"nutrition facts label {' '.join(_relevant[:4])} {_brand_hint}".strip()
        internal_web_context = await asyncio.wait_for(asyncio.to_thread(get_live_search, _search_query), timeout=3.0)
    except: pass

    super_prompt = build_super_prompt(
        label_text=clean_text, persona=persona, language=language,
        blur_info=blur_info, dna_flags=[], nova_level=1,
        research_context=internal_web_context or web_context,
    )

    try:
        raw_response = await asyncio.to_thread(llm_engine.call_llm, super_prompt, 4000)
        result_data = llm_engine.parse_llm_response(raw_response)
    except Exception as e:
        logger.error("Super-prompt LLM failed: %s", e)
        err_msg = str(e).lower()
        if "rate limit" in err_msg or "429" in err_msg:
            return {"error": "server_busy", "message": "⚠️ High traffic on AI servers. Please try again in a few minutes."}
        return {"error": "server_busy", "message": "⚠️ Analysis failed. Please try again."}

    def _float(val):
        if not val: return 0.0
        try: return float(str(val).replace(',', '.'))
        except: return 0.0

    def _primary(d):
        base_cal = _float(d.get("calories")) or _float(d.get("energy")) or _float(d.get("kcal")) or _float(d.get("energy_kcal"))
        kj = _float(d.get("energy_kj"))
        if not base_cal and kj: base_cal = kj / 4.184
        return {
            "calories": base_cal, "protein": _float(d.get("protein")),
            "carbs": _float(d.get("carbs")) or _float(d.get("carbohydrate")),
            "fat": _float(d.get("fat")) or _float(d.get("total_fat")),
            "sugar": _float(d.get("sugar")) or _float(d.get("total_sugars")),
            "fiber": _float(d.get("fiber")) or _float(d.get("dietary_fiber")),
            "saturated_fat": _float(d.get("saturated_fat")) or _float(d.get("saturated")),
        }

    category = result_data.get("product_category") or product_category_hint or "unknown"
    rich = _primary(result_data)
    rich["sodium"] = _float(result_data.get("sodium_mg"))
    rich["trans_fat"] = _float(result_data.get("trans_fat"))
    rich["cholesterol"] = _float(result_data.get("cholesterol_mg"))
    rich["potassium"] = _float(result_data.get("potassium_mg"))
    rich["calcium"] = _float(result_data.get("calcium_mg"))
    rich["iron"] = _float(result_data.get("iron_mg"))
    math_ok = atwater_math_check(rich, category)
    if not math_ok["is_valid"]:
        logger.warning("Atwater mismatch: %s — triggering AI Self-Correction.", math_ok["reason"])
        error_hint = f"\n\n🚨 PHYSICS ERROR IN YOUR LAST EXTRACTION: {math_ok['reason']}\nRE-EXAMINE THE IMAGE. Ensure Total Carbs + Protein + Fat <= 100g."
        try:
            retry_prompt = build_super_prompt(
                label_text=clean_text + error_hint, persona=persona, language=language,
                blur_info=blur_info, dna_flags=[], nova_level=1,
                research_context=internal_web_context or web_context,
            )
            raw2 = await asyncio.to_thread(llm_engine.call_llm, retry_prompt, 4000)
            result_data = llm_engine.parse_llm_response(raw2)
            category = result_data.get("product_category") or category
            rich = _primary(result_data)
            math_ok = atwater_math_check(rich, category)
            rich["sodium"] = _float(result_data.get("sodium_mg"))
            rich["trans_fat"] = _float(result_data.get("trans_fat"))
            rich["cholesterol"] = _float(result_data.get("cholesterol_mg"))
            rich["potassium"] = _float(result_data.get("potassium_mg"))
            rich["calcium"] = _float(result_data.get("calcium_mg"))
            rich["iron"] = _float(result_data.get("iron_mg"))
        except: pass

    ingredients_raw = result_data.get("ingredients_raw", "") or ""
    dna_res = apply_dna_overrides(
        full_ocr_text=extracted_text, nutrients=rich, ingredients_raw=ingredients_raw,
        base_score=5, category=category, front_text=front_text,
    )

    if dna_res["action"] == "BLOCK":
        return {"error": "impossible_data", "message": dna_res["reason"], "dna_action": "BLOCK"}

    explanation = get_explanation_report(rich, ingredients_raw)
    nova_level = explanation["nova_level"]

    llm_list = result_data.get("nutrients") or result_data.get("nutrient_breakdown") or []
    
    # Filter llm_list: ensure items are dicts and have valid nutrient keywords in their 'name'
    valid_keywords = {
        "calorie", "energy", "kcal", "kj", "protein", "carbohydrate", "carb", "fat", "saturated", 
        "trans", "cholesterol", "sodium", "salt", "fiber", "fibre", "sugar", "potassium", "calcium", 
        "iron", "zinc", "vitamin", "folate", "folic", "niacin", "riboflavin", "thiamin", "biotin", 
        "iodine", "magnesium", "phosphorus", "copper", "manganese", "selenium"
    }
    if isinstance(llm_list, list):
        filtered_list = []
        for item in llm_list:
            if isinstance(item, dict) and "name" in item:
                name_lower = str(item["name"]).lower()
                if any(kw in name_lower for kw in valid_keywords):
                    filtered_list.append(item)
        llm_list = filtered_list

    if not isinstance(llm_list, list) or not llm_list:
        _backup = [
            ("Energy", ["calories", "energy", "kcal", "energy_kcal"], "kcal"),
            ("Protein", ["protein"], "g"),
            ("Carbohydrates", ["carbs", "carbohydrate"], "g"),
            ("Fat", ["fat", "total_fat"], "g"),
            ("Sugar", ["sugar", "total_sugars"], "g"),
            ("Fiber", ["fiber", "dietary_fiber", "fibre"], "g"),
            ("Sodium", ["sodium_mg", "sodium", "salt"], "mg"),
            ("Saturated Fat", ["saturated_fat", "saturated"], "g"),
            ("Trans Fat", ["trans_fat"], "g"),
            ("Cholesterol", ["cholesterol_mg", "cholesterol"], "mg"),
            ("Potassium", ["potassium_mg", "potassium"], "mg"),
            ("Calcium", ["calcium_mg", "calcium"], "mg"),
            ("Iron", ["iron_mg", "iron"], "mg")
        ]
        llm_list = []
        for label, keys, unit in _backup:
            val = None
            for k in keys:
                if result_data.get(k) is not None:
                    val = _float(result_data.get(k))
                    break
            if val is not None:
                llm_list.append({"name": label, "value": val, "unit": unit})

    def _title_nutrient(raw_name: str) -> str:
        stripped = raw_name.strip()
        prefix = "  of which " if raw_name.lower().lstrip().startswith("of which") else ""
        core = re.sub(r"(?i)of\s+which\s+", "", stripped).strip() if prefix else stripped
        return (prefix + core.title()).rstrip()

    nutrient_breakdown = []
    for n in llm_list:
        raw_val = n.get("value", 0)
        if isinstance(raw_val, str):
            m = re.search(r"[\d]+\.?[\d]*", raw_val.replace(",", "."))
            raw_val = float(m.group()) if m else 0.0
        nutrient_breakdown.append({
            "name": _title_nutrient(n.get("name", "?")), "value": round(float(raw_val or 0), 2),
            "unit": n.get("unit", ""), "rating": n.get("rating", ""), "impact": n.get("impact", ""),
        })

    for n in nutrient_breakdown:
        if not n.get("rating"):
            r = _rule_rate(n["name"], float(n.get("value") or 0), n.get("unit", ""))
            n["rating"], n["impact"] = r["rating"], r["impact"]

    dna_flags = dna_res.get("extra_flags", [])
    if dna_res["action"] == "OVERRIDE":
        final_score, dna_flags = int(dna_res.get("score") or 2), [dna_res.get("reason", "")] + dna_flags
    else:
        final_score = int(result_data.get("score") or compute_rule_based_score(rich, nova_level))
        if nova_level == 4 and final_score > 4: final_score = 4
    final_score = adjust_score_for_persona(final_score, rich, clean_text, persona)
    final_score = max(1, min(10, int(final_score or 1)))

    product_name = result_data.get("product_name") or ""
    if product_name.lower().strip() in {"unknown", "unknown product", "", "n/a", "food product", "product"}:
        cat = (result_data.get("product_category") or "").capitalize()
        ings = (result_data.get("ingredients_raw") or "").split(",")
        product_name = f"{cat} Product" if cat and cat.lower() not in ("other", "") else (f"{ings[0].strip().title()}-based Product" if ings and ings[0].strip() else "Food Product")

    confidence = compute_extraction_confidence(
        result_data=result_data, ocr_word_count=len(clean_text.split()),
        avg_ocr_confidence=0.85, atwater_valid=math_ok.get("is_valid", False),
        nutrient_count=len(nutrient_breakdown),
    )
    
    if confidence["tier"] == "UNRELIABLE":
        return {"error": "low_confidence", "message": "⚠️ " + confidence["message"], "confidence": confidence, "product_name": product_name}

    verdict, summary, pros, eli5 = result_data.get("verdict") or dna_res.get("reason") or "Analyzed", result_data.get("summary") or dna_res.get("reason") or "", result_data.get("pros", []), result_data.get("eli5_explanation") or result_data.get("eli5", "")
    cons = dna_flags + [c for c in result_data.get("cons", []) if c not in dna_flags]

    age_warnings, phys_warnings = result_data.get("age_warnings", []), explanation.get("persona_warnings", [])
    merged_warnings_list = build_merged_warnings(age_warnings, phys_warnings)

    ingredients_spotlight = result_data.get("ingredients_spotlight", [])
    if isinstance(ingredients_spotlight, list):
        sanitized_spotlight = []
        for ing in ingredients_spotlight:
            if isinstance(ing, dict):
                sanitized_spotlight.append(ing)
            elif isinstance(ing, str) and len(ing.strip()) > 0:
                name_title = ing.strip().title()
                sanitized_spotlight.append({
                    "name": name_title,
                    "type": "natural",
                    "safety_rating": "safe",
                    "what_it_is": f"{name_title} is a food ingredient.",
                    "health_impact": "Part of the product formulation.",
                    "curiosity_fact": "Check the full ingredients list for details."
                })
        ingredients_spotlight = sanitized_spotlight
    else:
        ingredients_spotlight = []

    if not ingredients_spotlight and ingredients_raw:
        for ing in [i.strip() for i in re.split(r"[,;]", ingredients_raw) if i.strip()][:8]:
            if len(ing) > 2: ingredients_spotlight.append({"name": ing.title(), "type": "natural", "safety_rating": "safe", "what_it_is": f"{ing.title()} is a food ingredient.", "health_impact": "Part of the product formulation.", "curiosity_fact": "Check the full ingredients list for details."})

    latency_ms = int((time.time() - t_pipeline_start) * 1000)

    from app.services.brain import EatlyticBrain
    brain = EatlyticBrain()
    brain_nutrients = {
        "calories": rich.get("calories", 0.0),
        "protein": rich.get("protein", 0.0),
        "carbs": rich.get("carbs", 0.0),
        "fat": rich.get("fat", 0.0),
        "sugar": rich.get("sugar", 0.0),
        "fiber": rich.get("fiber", 0.0),
        "saturated_fat": rich.get("saturated_fat", 0.0),
        "sodium": rich.get("sodium", 0.0),
        "trans_fat": rich.get("trans_fat", 0.0),
        "cholesterol": rich.get("cholesterol", 0.0),
    }
    
    final_output = brain.compile_local_report(
        product_name=product_name,
        brand="",
        category=category,
        nutrients=brain_nutrients,
        ingredients_raw=ingredients_raw,
        persona=persona,
        eatlytic_score=final_score,
        device_key=device_key
    )
    final_output["data_source"] = "llm_fallback"
    final_output["extraction_confidence"] = confidence
    final_output["perf_metrics"] = {"latency_ms": latency_ms}
    final_output["disclaimer"] = MEDICAL_DISCLAIMER

    try:
        final_output["whatsapp_content"] = get_whatsapp_tiered_content(final_output)
    except:
        pass

    try:
        upsert_food_product(name=product_name, nutrients=final_output["nutrient_breakdown"], score=final_output["score"], ingredients_raw=ingredients_raw, category=category, source="llm_scan")
    except Exception as dberr:
        logger.warning(f"upsert_food_product failed: {dberr}")

    if "error" not in final_output and final_output.get("extraction_confidence", {}).get("tier", "") != "UNRELIABLE":
        set_ai_cache(cache_key, {k: v for k, v in final_output.items() if k != "scan_meta"})
    return final_output

def upsert_food_product(name, nutrients, score, ingredients_raw="", barcode=None, brand="", category="", source="llm_scan") -> int:
    from app.database.connection import db_conn
    def _get(key):
        for n in nutrients:
            if key in n.get("name", "").lower():
                v = n.get("value", 0)
                return float(v) if isinstance(v, (int, float)) else 0
        return 0
    cal = _get("calorie") or _get("energy")
    with db_conn() as conn:
        existing = conn.execute("SELECT id FROM food_products WHERE barcode=?", (barcode,)).fetchone() if barcode else conn.execute("SELECT id FROM food_products WHERE LOWER(name)=LOWER(?) AND LOWER(brand)=LOWER(?)", (name.strip(), brand.strip())).fetchone()
        if existing:
            conn.execute("UPDATE food_products SET scan_count=scan_count+1, updated_at=datetime('now') WHERE id=?", (existing["id"],))
            prod_id = existing["id"]
        else:
            cursor = conn.execute("""
                INSERT INTO food_products (
                     name, brand, category, barcode, calories_100g, 
                     protein_100g, carbs_100g, fat_100g, sodium_100g, 
                     fiber_100g, sugar_100g, eatlytic_score, scan_count
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)
            """, (
                name.strip(), brand, category, barcode, cal, 
                _get("protein"), _get("carb"), _get("fat"), _get("sodium"), 
                _get("fiber") or _get("fibre"), _get("sugar"), score
            ))
            prod_id = cursor.lastrowid
        
        # Upsert extra metadata in secondary table
        sat_fat = _get("saturated")
        conn.execute("""
            INSERT OR REPLACE INTO food_products_extra (id, sat_fat_100g, ingredients_raw, source)
            VALUES (?, ?, ?, ?)
        """, (prod_id, sat_fat, ingredients_raw, source))
        return prod_id
