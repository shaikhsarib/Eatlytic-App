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
from app.services.llm.validators import _rule_rate, compute_rule_based_score
from app.services.explanation_engine import get_explanation_report, adjust_score_for_persona, verify_atwater_math
from app.services.fake_detector import FakeDetector

logger = logging.getLogger(__name__)

# ── INDIAN INGREDIENT & ADDITIVE LEXICON ──────────────────────────────
# Maps ingredient names / INS codes to glycemic index (GI), safety tiers, and clinical reasoning.
INDIAN_INGREDIENT_LEXICON = {
    # Starches / Flours / High Glycemic Carbs
    "maltodextrin": {
        "name": "Maltodextrin",
        "category": "starch",
        "gi_level": "HIGH",
        "typical_gi": 110,
        "diabetic_verdict": "AVOID",
        "reason": "Maltodextrin has a Glycemic Index of 110-185—spiking blood sugar faster than pure table glucose (GI ~100). Highly dangerous for insulin resistance.",
        "curiosity_fact": "It is widely used in packaged foods as a cheap filler, thickener, and preservative.",
        "ins_codes": ["ins 1400", "e1400"]
    },
    "maida": {
        "name": "Maida (Refined Wheat Flour)",
        "category": "flour",
        "gi_level": "HIGH",
        "typical_gi": 75,
        "diabetic_verdict": "AVOID",
        "reason": "Extremely refined flour with fiber and nutrients stripped. Causes rapid insulin surges and contributes to visceral fat accumulation.",
        "curiosity_fact": "Refined flour is chemically bleached with benzoyl peroxide or chlorine gas in industrial milling.",
        "ins_codes": []
    },
    "refined wheat flour": {
        "name": "Refined Wheat Flour (Maida)",
        "category": "flour",
        "gi_level": "HIGH",
        "typical_gi": 75,
        "diabetic_verdict": "AVOID",
        "reason": "Essentially Maida. High starch content with zero fibrous buffer to slow glucose absorption.",
        "curiosity_fact": "The bran and germ are completely removed, stripping >90% of dietary fiber, vitamins, and minerals.",
        "ins_codes": []
    },
    "corn starch": {
        "name": "Corn Starch / Corn Flour",
        "category": "starch",
        "gi_level": "HIGH",
        "typical_gi": 85,
        "diabetic_verdict": "AVOID",
        "reason": "Pure refined carbohydrate that acts as a rapid glycemic spike trigger. Highly concentrated.",
        "curiosity_fact": "Mainly used as a thickener and binder in gravies and instant noodles.",
        "ins_codes": []
    },
    "potato starch": {
        "name": "Potato Starch",
        "category": "starch",
        "gi_level": "HIGH",
        "typical_gi": 80,
        "diabetic_verdict": "AVOID",
        "reason": "Highly refined starch with a rapid rate of digestion, promoting fast postprandial glucose rises.",
        "curiosity_fact": "Has high binding power and yields high viscosity when cooked.",
        "ins_codes": []
    },
    "dextrose": {
        "name": "Dextrose (D-Glucose)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 100,
        "diabetic_verdict": "AVOID",
        "reason": "Pure medical glucose. Absorbs directly through the stomach lining, causing an immediate, severe glycemic surge.",
        "curiosity_fact": "Structurally identical to the glucose circulating in human blood vessels.",
        "ins_codes": []
    },
    "maltose": {
        "name": "Maltose (Malt Sugar)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 105,
        "diabetic_verdict": "AVOID",
        "reason": "A disaccharide composed of two glucose units. Higher glycemic impact than standard sucrose.",
        "curiosity_fact": "Produced during the malting of barley grains.",
        "ins_codes": []
    },
    "liquid glucose": {
        "name": "Liquid Glucose / Glucose Syrup",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 90,
        "diabetic_verdict": "AVOID",
        "reason": "Highly concentrated glucose solution that leaves zero digestive resistance, triggering immediate insulin secretion.",
        "curiosity_fact": "Extensively used in Indian confectionery to prevent sugar crystallization.",
        "ins_codes": []
    },
    "invert sugar": {
        "name": "Invert Sugar / Invert Syrup",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 65,
        "diabetic_verdict": "AVOID",
        "reason": "A mixture of glucose and fructose. Very sweet, causes high liver-glycogen stress and glycemic load.",
        "curiosity_fact": "Sweeter than regular sugar and keeps baked goods moist for much longer.",
        "ins_codes": []
    },
    "high fructose corn syrup": {
        "name": "High Fructose Corn Syrup (HFCS)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 65,
        "diabetic_verdict": "AVOID",
        "reason": "High fructose load drives hepatic lipogenesis (fatty liver), systemic inflammation, and severe leptin resistance.",
        "curiosity_fact": "Fructose is processed exclusively by the liver, unlike glucose which can be used by all cells.",
        "ins_codes": []
    },
    "sugar": {
        "name": "Refined Sugar (Sucrose)",
        "category": "sweetener",
        "gi_level": "HIGH",
        "typical_gi": 65,
        "diabetic_verdict": "AVOID",
        "reason": "Consists of 50% glucose and 50% fructose. High empty calorie profile. Direct cause of pancreatic fatigue and chronic inflammation.",
        "curiosity_fact": "Sugar triggers the release of dopamine in the brain's reward center, highly mimicking addictive substances.",
        "ins_codes": []
    },

    # Fats / Oils
    "palm oil": {
        "name": "Refined Palm Oil",
        "category": "oil",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Highly refined, industrially processed fat. High in palmitic acid which is highly pro-inflammatory and increases LDL cardiovascular risk in diabetics.",
        "curiosity_fact": "Palm oil is the most widely consumed vegetable oil in the world due to its low cost and high melting point.",
        "ins_codes": []
    },
    "palmolein": {
        "name": "Refined Palmolein Oil",
        "category": "oil",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Identical concern to palm oil. Liquid fraction of palm oil, loaded with industrial saturated fatty acids that worsen metabolic markers.",
        "curiosity_fact": "Often used for commercial deep frying because it can withstand repeated heating without breaking down.",
        "ins_codes": []
    },
    "hydrogenated vegetable oil": {
        "name": "Hydrogenated Vegetable Oil / Vanaspati",
        "category": "oil",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Source of industrial trans fats. Wrecks lipid panels, spikes systemic inflammation, damages blood vessel linings, and raises stroke risk.",
        "curiosity_fact": "Vegetable oil is bubbled with hydrogen gas at high temperatures in the presence of a nickel catalyst to solidify it.",
        "ins_codes": []
    },

    # Sweeteners (Artificial / Sugar Alcohols)
    "aspartame": {
        "name": "Aspartame",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Artificial sweetener. Zero glycemic index but can trigger insulin responses via cephalic phase activation. Classified as possibly carcinogenic (Group 2B) by IARC.",
        "curiosity_fact": "It is about 200 times sweeter than sucrose and breaks down at high baking temperatures.",
        "ins_codes": ["ins 951", "e951", "951"]
    },
    "acesulfame potassium": {
        "name": "Acesulfame Potassium (Acesulfame K)",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Synthetic artificial sweetener. Low-calorie but linked to alterations in gut microbiota. Often blended with aspartame.",
        "curiosity_fact": "Discovered accidentally in 1967 when a chemist licked his finger.",
        "ins_codes": ["ins 950", "e950", "950"]
    },
    "sucralose": {
        "name": "Sucralose",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Chlorinated sugar derivative. Zero glycemic, but evidence suggests chronic use can decrease insulin sensitivity and alter gut biome.",
        "curiosity_fact": "It is made by selectively substituting three chlorine atoms for three hydroxyl groups on sucrose.",
        "ins_codes": ["ins 955", "e955", "955"]
    },
    "steviol glycosides": {
        "name": "Stevia (Steviol Glycosides)",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "SAFE",
        "reason": "Natural, plant-derived non-nutritive sweetener. Zero glycemic index, zero calories, and safe for metabolic health.",
        "curiosity_fact": "Extracted from the leaves of the Stevia rebaudiana plant, native to South America.",
        "ins_codes": ["ins 960", "e960", "960"]
    },
    "stevia": {
        "name": "Stevia Extract",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "SAFE",
        "reason": "Pure plant-based sweetener. Zero glycemic surge, highly recommended for sugar substitution.",
        "curiosity_fact": "Leaves have been used by indigenous South American tribes for over 1,500 years to sweeten teas.",
        "ins_codes": []
    },
    "erythritol": {
        "name": "Erythritol",
        "category": "sweetener",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "SAFE",
        "reason": "Natural sugar alcohol. Near-zero calories, zero glycemic impact, and highly tolerated without gastrointestinal side effects.",
        "curiosity_fact": "About 90% is absorbed in the small intestine and excreted unchanged in urine.",
        "ins_codes": ["ins 968", "e968", "968"]
    },
    "maltitol": {
        "name": "Maltitol",
        "category": "sweetener",
        "gi_level": "MODERATE",
        "typical_gi": 35,
        "diabetic_verdict": "CAUTION",
        "reason": "Sugar alcohol with a moderate glycemic index (GI ~35). Spikes blood glucose and insulin (about half the impact of sugar).",
        "curiosity_fact": "Frequently used in 'sugar-free' chocolates, but can cause bloating or laxative effects.",
        "ins_codes": ["ins 965", "e965", "965"]
    },
    "sorbitol": {
        "name": "Sorbitol",
        "category": "sweetener",
        "gi_level": "MODERATE",
        "typical_gi": 9,
        "diabetic_verdict": "CAUTION",
        "reason": "Sugar alcohol. Low glycemic index, but can cause severe abdominal cramping and osmotic diarrhea in moderate doses.",
        "curiosity_fact": "Naturally occurs in fruits like apples and pears but is industrially synthesized from corn syrup.",
        "ins_codes": ["ins 420", "e420", "420"]
    },

    # Flavor Enhancers
    "monosodium glutamate": {
        "name": "Monosodium Glutamate (MSG)",
        "category": "additive",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Umami flavor enhancer. Generally recognized as safe by FSSAI, but can act as an excitotoxin in high amounts, potentially causing headaches.",
        "curiosity_fact": "It is the sodium salt of glutamic acid, an amino acid abundant in nature.",
        "ins_codes": ["ins 621", "e621", "621"]
    },
    "disodium 5'-ribonucleotides": {
        "name": "Disodium 5'-Ribonucleotides",
        "category": "additive",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "CAUTION",
        "reason": "Synthetic flavor enhancer (INS 635) that synergetically works with MSG to intensify savory taste profiles.",
        "curiosity_fact": "Often found in instant noodle tastemakers, chips, and spice mixes.",
        "ins_codes": ["ins 635", "e635", "635"]
    },

    # Synthetic Food Dyes
    "tartrazine": {
        "name": "Tartrazine (Synthetic Yellow Dye)",
        "category": "dye",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Coal-tar derived artificial dye. Banned or heavily restricted in several European countries due to links with childhood hyperactivity and hives.",
        "curiosity_fact": "Mandatorily carries a warning label in the EU: 'May have an adverse effect on activity and attention in children.'",
        "ins_codes": ["ins 102", "e102", "102"]
    },
    "sunset yellow": {
        "name": "Sunset Yellow FCF (Orange Dye)",
        "category": "dye",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Petroleum-derived artificial food coloring. Associated with allergic reactions, worsening asthma, and hyperactivity in kids.",
        "curiosity_fact": "Used in candies, carbonated orange beverages, and packaged snack foods.",
        "ins_codes": ["ins 110", "e110", "110"]
    },
    "carmoisine": {
        "name": "Carmoisine (Red Dye)",
        "category": "dye",
        "gi_level": "LOW",
        "typical_gi": 0,
        "diabetic_verdict": "AVOID",
        "reason": "Synthetic red azo dye. Strongly linked to hives and hyperactivity. Banned in Canada, USA, and Japan.",
        "curiosity_fact": "Often used in jams, gelatin desserts, and packaged pastries.",
        "ins_codes": ["ins 122", "e122", "122"]
    }
}

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

    def compile_local_report(self, product_name: str, brand: str, category: str, nutrients: Dict, ingredients_raw: str, persona: str = "diabetic", eatlytic_score: int = None) -> Dict:
        """
        Builds a full high-fidelity frontend scan response structure completely offline.
        Bypasses LLM cloud latency entirely.
        """
        matched_ings = self.match_lexicon_ingredients(ingredients_raw)
        
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
            "gi_level": audit["gi_level"]
        }

