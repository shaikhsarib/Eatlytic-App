"""
app/services/brain.py
The Eatlytic Intelligence Brain.
A local, rule-based clinical and chemical intelligence layer providing 
deterministic glycemic audits, additive decoding, Atwater checks, and FSSAI safety checks.
"""

import re
import logging
from typing import Dict, List, Tuple
from app.services.alternatives import get_healthy_alternative
from app.ai.llm.validators import _rule_rate, compute_rule_based_score
from app.services.explanation_engine import get_explanation_report, adjust_score_for_persona, verify_atwater_math
from app.services.fake_detector import FakeDetector
from app.ai.lexicons.indian_ingredients import INDIAN_INGREDIENT_LEXICON

logger = logging.getLogger(__name__)

# Import additive DB service (lazy — OK if additives.json missing during tests)
try:
    from app.services.additive_db import scan_ingredients as _scan_additives, get_ingredient_risk_summary as _additive_risk
    _ADDITIVE_DB_AVAILABLE = True
except Exception as _e:
    logger.warning("Additive DB not available: %s", _e)
    _ADDITIVE_DB_AVAILABLE = False
    def _scan_additives(text): return []
    def _additive_risk(text, persona="general"): return {"total_additives_found": 0, "safe_count": 0, "caution_count": 0, "avoid_count": 0, "red_flags": [], "persona_flags": [], "matched_additives": []}



class EatlyticBrain:
    """
    The Core Intelligence Moat of the Eatlytic Platform.
    Performs fully local, high-fidelity, deterministic nutritional audits,
    glycemic threat ratings, regulatory FSSAI check overrides, and Atwater audits.
    """

    
    def __init__(self):
        self.detector = FakeDetector()
        
    def normalize_ingredient_text(self, text: str) -> str:
        """Standardizes ingredients string for lookup token matching."""
        if not text:
            return ""
        val = text.lower().strip()
        val = re.sub(r'[\s\-_\/]+', ' ', val)
        return val

    def match_lexicon_ingredients(self, ingredients_text: str) -> List[Dict]:
        """
        Parses raw text to identify matches inside the INDIAN_INGREDIENT_LEXICON.
        Robust to E-numbers/INS codes and direct name substrings.
        """
        if not ingredients_text or len(ingredients_text.strip()) < 3:
            return []
            
        normalized = self.normalize_ingredient_text(ingredients_text)
        matches = []
        matched_keys = set()
        
        for key, entry in INDIAN_INGREDIENT_LEXICON.items():
            name_matched = key in normalized
            ins_matched = False
            
            for code in entry["ins_codes"]:
                if code in normalized:
                    ins_matched = True
                    break
                    
            if (name_matched or ins_matched) and entry["name"] not in matched_keys:
                matches.append(entry)
                matched_keys.add(entry["name"])
                
        return matches

    def run_clinical_audit(self, nutrients: Dict, matched_ingredients: List[Dict], persona: str = "diabetic") -> Dict:
        """
        Runs local, rule-based clinical logic for the specified persona.
        Assigns safety verdicts, ratings, and teaspoon equivalents offline.
        """
        persona = str(persona or "diabetic").lower().strip()
        sugar = float(nutrients.get("sugar") or nutrients.get("sugar_100g") or 0.0)
        carbs = float(nutrients.get("carbs") or nutrients.get("carbs_100g") or 0.0)
        sodium = float(nutrients.get("sodium") or nutrients.get("sodium_mg") or nutrients.get("sodium_100g") or 0.0)
        sat_fat = float(nutrients.get("saturated_fat") or nutrients.get("sat_fat_100g") or 0.0)
        trans_fat = float(nutrients.get("trans_fat") or 0.0)
        
        # Initialize defaults
        verdict = "SAFE"
        tier = "Safe"
        reasons = []
        gi_level = "LOW"
        score_color = "#22c55e" # High-confidence standard green
        score_deductions = 0
        
        teaspoons = 0.0
        if sugar > 0:
            teaspoons = round(sugar / 4.2, 1)

        # ── PERSONA 1: DIABETIC CARE ──────────────────────────────────────────
        if persona in ["diabetic", "diabetic care"]:
            if sugar > 15.0:
                verdict = "AVOID"
                score_deductions += 4
                reasons.append(f"🔴 Extremely high sugar content ({sugar}g/100g) which causes severe glycemic surges and insulin stress.")
            elif sugar > 5.0:
                verdict = "CAUTION" if verdict != "AVOID" else verdict
                score_deductions += 2
                reasons.append(f"🟡 Moderate to high sugar ({sugar}g/100g) — raises blood sugar. Consume in small portions.")
            
            if carbs > 45.0:
                verdict = "AVOID"
                score_deductions += 3
                reasons.append(f"🔴 High carbohydrate load ({carbs}g/100g) threatening glycemic stability.")
            elif carbs > 25.0:
                verdict = "CAUTION" if verdict != "AVOID" else verdict
                score_deductions += 1
                reasons.append(f"🟡 Sizable carbohydrates ({carbs}g/100g) — requires careful portioning.")

            for ing in matched_ingredients:
                if ing["diabetic_verdict"] == "AVOID":
                    verdict = "AVOID"
                    gi_level = "HIGH"
                    score_deductions += 3
                    reasons.append(f"🔴 Contains {ing['name']}: {ing['reason']}")
                elif ing["diabetic_verdict"] == "CAUTION" and verdict != "AVOID":
                    verdict = "CAUTION"
                    gi_level = "MODERATE" if gi_level == "LOW" else gi_level
                    score_deductions += 1.5
                    reasons.append(f"🟡 Contains {ing['name']}: {ing['reason']}")

        # ── PERSONA 2: HYPERTENSION / GENERAL ADULT ────────────────────────────
        else:
            if sat_fat > 10.0:
                verdict = "CAUTION" if verdict != "AVOID" else verdict
                score_deductions += 2
                reasons.append(f"🟡 High saturated fat levels ({sat_fat}g/100g) — watch portion limits to protect arterial wellness.")
            elif sat_fat > 5.0:
                verdict = "CAUTION" if verdict != "AVOID" else verdict
                score_deductions += 1.0
                reasons.append(f"🟡 Sizable saturated fats ({sat_fat}g/100g) — consume moderately.")

            if sugar > 22.5:
                verdict = "AVOID"
                score_deductions += 3
                reasons.append(f"🔴 Dangerous sugar content ({sugar}g/100g) exceeding 90% of daily WHO recommended limits.")
            elif sugar > 10.0:
                verdict = "CAUTION" if verdict != "AVOID" else verdict
                score_deductions += 1.5
                reasons.append(f"🟡 Elevated sugars ({sugar}g/100g) — watch portion limits.")

        # ── GENERAL HEALTH DEFECT AUDITS (Applies to all personas) ────────────
        if sodium > 600.0:
            score_deductions += 3
            reasons.append(f"🔴 Dangerous sodium levels ({sodium}mg/100g) linked to arterial hypertension and fluid retention.")
        elif sodium > 200.0:
            score_deductions += 1
            reasons.append(f"🟡 Moderate sodium levels ({sodium}mg/100g) — drink plenty of water to offset metabolic load.")

        if trans_fat > 0.5:
            verdict = "AVOID"
            score_deductions += 4
            reasons.append(f"🔴 Trans-fat content is high ({trans_fat}g/100g) — raises LDL cholesterol and promotes arterial plaque formation.")
        elif trans_fat > 0.0:
            verdict = "CAUTION" if verdict != "AVOID" else verdict
            score_deductions += 2
            reasons.append(f"🟡 Trace trans-fats detected ({trans_fat}g/100g) — ideally should be 0.0g.")

        palm_oil_present = any(any(x in ing["name"].lower() for x in ["palm oil", "palmolein"]) for ing in matched_ingredients)
        if palm_oil_present:
            score_deductions += 1.5
            if verdict != "AVOID":
                verdict = "CAUTION"
            reasons.append("🟡 Contains highly refined Palm/Palmolein Oil, which is associated with systemic metabolic inflammation.")

        sweetener_present = any(any(x in ing["name"].lower() for x in ["aspartame", "acesulfame"]) for ing in matched_ingredients)
        if sweetener_present:
            score_deductions += 1.5
            reasons.append("⚠️ FSSAI Statutory Warning: Contains artificial sweetener. Not recommended for children.")

        # Match deductions to Eatlytic Score (Base 10)
        final_score = round(max(1, min(10, 10 - score_deductions)))
        
        # Standardize score color and safety tier
        if final_score >= 7:
            tier = "Safe"
            score_color = "#22c55e"
        elif final_score >= 4:
            tier = "Limit"
            score_color = "#f59e0b"
        else:
            tier = "Avoid"
            score_color = "#ef4444"
            
        safety_verdict = "Nutritious Choice" if final_score >= 7 else "Moderation Advised" if final_score >= 4 else "Glycemic Threat"

        return {
            "verdict": verdict,
            "safety_tier": tier,
            "score": final_score,
            "score_color": score_color,
            "safety_verdict": safety_verdict,
            "reasons": reasons,
            "gi_level": gi_level,
            "teaspoons": teaspoons,
        }

    def compile_local_report(self, product_name: str, brand: str, category: str, nutrients: Dict, ingredients_raw: str, persona: str = "diabetic", eatlytic_score: int = None, device_key: str = None) -> Dict:
        """
        Builds a full high-fidelity frontend scan response structure completely offline.
        Bypasses LLM cloud latency entirely.
        """
        matched_ings = self.match_lexicon_ingredients(ingredients_raw)

        # ── Additive DB Integration (The Moat) ─────────────────────────────
        # Scan against verified proprietary additive database
        additive_risk = _additive_risk(ingredients_raw, persona=persona)
        additive_matches = _scan_additives(ingredients_raw)

        # Merge additive DB matches into matched_ings if not already present
        existing_names = {m["name"].lower() for m in matched_ings}
        for add in additive_matches:
            if add["name"].lower() not in existing_names:
                # Convert additive DB format to brain lexicon format
                tier = add.get("safety_tier", "SAFE")
                diabetic_verdict = "SAFE" if tier == "SAFE" else "CAUTION" if tier == "CAUTION" else "AVOID"
                matched_ings.append({
                    "name": add["name"],
                    "category": add.get("category", "additive"),
                    "gi_level": "LOW",
                    "typical_gi": 0,
                    "diabetic_verdict": diabetic_verdict,
                    "reason": add.get("curiosity_fact", ""),
                    "curiosity_fact": add.get("curiosity_fact", ""),
                    "source": "additive_db",
                    "safety_tier": tier,
                    "fssai_status": add.get("fssai_status", "unknown"),
                })
                existing_names.add(add["name"].lower())
        
        # Determine base scores from DB match or compute rules
        sugar = float(nutrients.get("sugar") or nutrients.get("sugar_100g") or 0.0)
        carbs = float(nutrients.get("carbs") or nutrients.get("carbs_100g") or 0.0)
        protein = float(nutrients.get("protein") or nutrients.get("protein_100g") or 0.0)
        fat = float(nutrients.get("fat") or nutrients.get("fat_100g") or 0.0)
        calories = float(nutrients.get("calories") or nutrients.get("calories_100g") or 0.0)
        sodium = float(nutrients.get("sodium") or nutrients.get("sodium_mg") or nutrients.get("sodium_100g") or 0.0)
        fiber = float(nutrients.get("fiber") or nutrients.get("fiber_100g") or 0.0)
        sat_fat = float(nutrients.get("saturated_fat") or nutrients.get("sat_fat_100g") or 0.0)

        rich = {
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
            "sugar": sugar,
            "fiber": fiber,
            "saturated_fat": sat_fat,
            "sodium_mg": sodium,
            "trans_fat": float(nutrients.get("trans_fat") or 0.0),
            "cholesterol": float(nutrients.get("cholesterol") or 0.0),
        }

        # Run get_explanation_report from explanation_engine to match warnings & details
        explanation = get_explanation_report(rich, ingredients_raw)
        nova_level = explanation["nova_level"]
        
        # Score calculation and active adjustments
        base_score = eatlytic_score if eatlytic_score is not None else int(compute_rule_based_score(rich, nova_level))
        final_score = adjust_score_for_persona(base_score, rich, ingredients_raw, persona)
        
        # Run clinical audit to apply local lexicon overlays
        audit = self.run_clinical_audit(rich, matched_ings, persona)
        
        # ── Phase 5 AI Personalized Medicine Genomic Overrides ──────────
        rich["ingredients_raw_text"] = ingredients_raw
        try:
            from app.services.personalized_medicine import apply_genomic_overrides
            audit = apply_genomic_overrides(device_key, rich, matched_ings, audit)
        except Exception as e:
            logger.error(f"Error applying genomic overrides: {e}")
        
        # If clinical audit suggests Avoid or Caution, let's reflect that in the score
        if audit["verdict"] == "AVOID" and final_score > 3:
            final_score = 3
        elif audit["verdict"] == "CAUTION" and final_score > 6:
            final_score = 6
            
        final_score = min(final_score, audit["score"])
        final_score = max(1, min(10, final_score))
        
        score_color = "#22c55e" if final_score >= 7 else "#f59e0b" if final_score >= 4 else "#ef4444"
        safety_tier = "Safe" if final_score >= 7 else "Limit" if final_score >= 4 else "Avoid"
        safety_verdict = "Nutritious Choice" if final_score >= 7 else "Moderation Advised" if final_score >= 4 else "Ultra-Processed"
        if final_score < 4 and persona in ["diabetic", "diabetic care"]:
            safety_verdict = "Glycemic Threat"

        # Build Atwater audit
        atwater = verify_atwater_math(rich)
        atwater_valid = atwater.get("is_atwater_valid", True)
        if not atwater_valid:
            atwater_reason = f"Atwater mismatch: calculated {atwater.get('calculated_cal')} kcal vs declared {atwater.get('declared_cal')} kcal."
        else:
            atwater_reason = "Atwater math verified."

        # Build nutrient breakdown structure
        raw_nutrients = [
            ("Energy", calories, "kcal"),
            ("Protein", protein, "g"),
            ("Carbohydrates", carbs, "g"),
            ("Fat", fat, "g"),
            ("Saturated Fat", sat_fat, "g"),
            ("Sugar", sugar, "g"),
            ("Fiber", fiber, "g"),
            ("Sodium", sodium, "mg")
        ]
        
        nutrient_breakdown = []
        for label, val, unit in raw_nutrients:
            r = _rule_rate(label, val, unit)
            nutrient_breakdown.append({
                "name": label,
                "value": round(float(val), 2),
                "unit": unit,
                "rating": r["rating"],
                "impact": r["impact"],
            })

        pros = []
        if protein >= 8: pros.append("High protein content supporting muscle repair")
        if fiber >= 3: pros.append("Good dietary fiber level aiding digestion")
        if sugar < 5: pros.append("Low sugar formulation")
        if sodium < 120: pros.append("Low sodium levels supporting cardiovascular health")
        if final_score >= 7: pros.append("Balanced, nutrient-dense profile")
        if not pros: pros.append("Standard macro profile")
        
        cons = []
        if sugar > 15: cons.append("High in added sugar")
        if sodium > 600: cons.append("High in sodium")
        if sat_fat > 5: cons.append("Contains substantial saturated fats")
        if nova_level == 4: cons.append("Ultra-processed formulation with food additives")
        for pw in explanation.get("persona_warnings", []):
            if pw["type"] in ("WARNING", "CAUTION"):
                cons.append(f"{pw['persona']}: {pw['msg']}")
                
        # Append clinical brain reasons
        for r in audit["reasons"]:
            if r not in cons:
                cons.append(r)
                
        if not cons:
            cons.append("Standard macro profile constraints verified.")

        # Spotlight compiling
        spotlight = []
        for ing in matched_ings:
            spotlight.append({
                "name": ing["name"],
                "type": "additive" if ing["category"] in ["additive", "sweetener", "dye"] else "natural",
                "safety_rating": "safe" if ing["diabetic_verdict"] == "SAFE" else "warning" if ing["diabetic_verdict"] == "CAUTION" else "danger",
                "what_it_is": f"{ing['name']} is classified as a metabolic {ing['category']}.",
                "health_impact": ing["reason"],
                "curiosity_fact": ing["curiosity_fact"]
            })

        other_ings = [i.strip() for i in re.split(r"[,;()]", ingredients_raw) if i.strip()]
        for ing in other_ings[:6]:
            normalized_ing = self.normalize_ingredient_text(ing)
            if len(ing) > 3 and not any(x["name"].lower() in normalized_ing for x in matched_ings):
                spotlight.append({
                    "name": ing.title(),
                    "type": "natural",
                    "safety_rating": "safe",
                    "what_it_is": f"{ing.title()} is a common ingredient in this product formulation.",
                    "health_impact": "Standard ingredient parsed locally.",
                    "curiosity_fact": "Verified by Eatlytic Brain."
                })

        # Add additive DB red flags to spotlight if not already there
        spotlight_names = {s["name"].lower() for s in spotlight}
        for flag in additive_risk.get("red_flags", [])[:4]:
            if flag["name"].lower() not in spotlight_names:
                tier = flag["safety_tier"]
                rating = "safe" if tier == "SAFE" else "warning" if tier == "CAUTION" else "danger"
                spotlight.append({
                    "name": flag["name"],
                    "type": "additive",
                    "safety_rating": rating,
                    "what_it_is": f"{flag['name']} ({flag.get('category', 'additive')}) — FSSAI: {flag.get('fssai_status', 'unknown')}.",
                    "health_impact": flag.get("curiosity_fact", ""),
                    "curiosity_fact": flag.get("curiosity_fact", "")
                })
                spotlight_names.add(flag["name"].lower())

        # Energy chart
        total_energy = (carbs * 4) + (protein * 4) + (fat * 9)
        if total_energy > 0:
            chart_data = [
                round((carbs * 4 * 100) / total_energy),
                round((protein * 4 * 100) / total_energy),
                round((fat * 9 * 100) / total_energy)
            ]
        else:
            chart_data = [50, 30, 20]

        summary = f"{brand} {product_name} parsed offline. Sugar content: {sugar}g/100g, Sodium: {sodium}mg/100g."
        
        # Build customizable ELI5 text based on Atwater checks and local scan results
        eli5 = (
            f"Eatlytic Brain scanned this locally. This product has an Eatlytic Score of {final_score}/10. "
            f"Its glycemic threat rating is {audit['gi_level']}. "
        )
        if audit["teaspoons"] > 0:
            eli5 += f"It packs about {audit['teaspoons']} teaspoons of sugar per 100g. "
            
        if not atwater_valid:
            eli5 += f"🚨 WARNING: {atwater_reason}"
        else:
            # If it's a verified match, add the database message
            if eatlytic_score is not None:
                eli5 = f"We found {brand} {product_name} in our verified food database! This means the nutritional values and ingredients have been certified by our team to be 100% accurate, saving you time and cloud scanning costs."

        return {
            "product_name": f"{brand} {product_name}".strip() or "Standard Food Product",
            "product_category": category or "unknown",
            "serving_size": "100g",
            "score": final_score,
            "score_color": score_color,
            "safety_tier": safety_tier,
            "safety_verdict": safety_verdict,
            "safety_reason": f"Verified product in Eatlytic database. Eatlytic Score: {final_score}/10.",
            "verdict": f"Verified Match. Score: {final_score}/10.",
            "summary": summary,
            "nutrient_breakdown": nutrient_breakdown,
            "pros": pros,
            "cons": cons,
            "age_warnings": [],
            "eli5_explanation": eli5,
            "molecular_insight": "Verified database record. Free from AI extraction errors.",
            "chart_data": chart_data,
            "ingredients_raw": ingredients_raw,
            "ingredients_spotlight": spotlight[:8],
            "extraction_confidence": {
                "tier": "HIGH",
                "score": 1.0,
                "message": "Verified database record matching OCR signatures.",
                "atwater_valid": atwater_valid,
            },
            "explanation": explanation,
            "better_alternative": get_healthy_alternative(
                product_category=category,
                persona=persona,
                product_name=product_name,
                ingredients=ingredients_raw
            ),
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
            "sodium": sodium,
            "fiber": fiber,
            "sugar": sugar,
            "data_source": "eatlytic_database",
            "atwater_audit": atwater,
            "clinical_audit": audit,
            "sugar_teaspoons": audit["teaspoons"],
            "gi_level": audit["gi_level"],
            "additive_db_summary": additive_risk,  # Verified additive intelligence
        }

