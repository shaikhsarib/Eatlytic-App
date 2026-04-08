"""
app/services/fake_detector.py
Eatlytic DNA: Proprietary detectors for Marketing Lies, NOVA 4, and Atwater Physics.
"""

import re
import logging

logger = logging.getLogger(__name__)

def atwater_math_check(nutrients: dict, category: str = "unknown") -> dict:
    """Fake Type 3: Prevents hallucinations and catches physically impossible labels."""
    try:
        protein = float(nutrients.get("protein", 0) or 0)
        carbs = float(nutrients.get("carbs", 0) or 0)
        fat = float(nutrients.get("fat", 0) or 0)
        stated_calories = float(nutrients.get("calories", 0) or 0)
    except (ValueError, TypeError):
        return {"is_valid": False, "reason": "Could not read macro numbers."}

    # 1. Macro Integrity Check (Hierarchy Protection)
    # Total Carbs must contain Sugar and Fiber
    sugar = float(nutrients.get("sugar", 0) or 0)
    fiber = float(nutrients.get("fiber", 0) or 0)
    if (sugar + fiber) > (carbs + 0.5): # 0.5 allowance for rounding
        return {
            "is_valid": False,
            "reason": f"Integrity Failure: Sugar ({sugar}g) + Fiber ({fiber}g) > Total Carbs ({carbs}g). Extraction error likely.",
        }

    # Total Fat must contain Saturated and Trans Fat
    sat_fat = float(nutrients.get("saturated_fat", 0) or 0)
    trans_fat = float(nutrients.get("trans_fat", 0) or 0)
    if (sat_fat + trans_fat) > (fat + 0.5):
        return {
            "is_valid": False,
            "reason": f"Integrity Failure: Sat Fat ({sat_fat}g) + Trans Fat ({trans_fat}g) > Total Fat ({fat}g). Extraction error likely.",
        }

    if stated_calories <= 0:
        # If both are 0, it's valid. If macros > 0 but calories=0, it's a mismatch.
        calculated_calories = (protein * 4) + (carbs * 4) + (fat * 9)
        if calculated_calories > 20: 
            return {"is_valid": False, "reason": f"Math Mismatch: Label says 0 kcal, but macros calculate to {calculated_calories:.0f} kcal."}
        return {"is_valid": True, "reason": None}

    calculated_calories = (protein * 4) + (carbs * 4) + (fat * 9)
    
    # Standard margin is 25% for all categories to prevent "hallucination bypass".
    # (Previously 35% for noodles, which was too loose).
    margin_percent = 0.25
        
    margin = stated_calories * margin_percent
    lower_bound = stated_calories - margin
    upper_bound = stated_calories + margin

    if not (lower_bound <= calculated_calories <= upper_bound):
        return {
            "is_valid": False,
            "reason": f"Math Mismatch: Label says {stated_calories} kcal, but macros calculate to {calculated_calories:.0f} kcal.",
        }
    return {"is_valid": True, "reason": None}


def detect_nova_4(ingredients_raw: str) -> dict:
    """Fake Type 2: Scans ingredients for ultra-processed markers."""
    if not ingredients_raw:
        return {"is_nova_4": False, "flags_found": []}

    upf_blacklist = [
        r"\be\d{3,4}[a-z]?\b", 
        r"ins\s*\d{3,4}[a-z]?", 
        r"flavor enhancer",
        r"emulsifier",
        r"humectant",
        r"stabiliz",
        r"hydrogenated",
        r"maltodextrin",
        r"high fructose corn syrup",
        r"artificial color",
        r"sweetener",
        r"aspartame",
        r"sucralose",
        r"acesulfame",
    ]

    flags_found = []
    ingredients_lower = ingredients_raw.lower()

    for pattern in upf_blacklist:
        matches = re.findall(pattern, ingredients_lower)
        if matches:
            f = (
                matches[0].upper()
                if ("e" in matches[0] or "ins" in matches[0])
                else matches[0].capitalize()
            )
            flags_found.append(f)

    flags_found = list(set(flags_found))
    is_nova_4 = len(flags_found) >= 2

    return {"is_nova_4": is_nova_4, "flags_found": flags_found}


def detect_fake_claims(full_ocr_text: str, ingredients_raw: str, front_text: str = "") -> dict:
    """Fake Type 1: Checks if the front claims "No Sugar" but back has hidden sugars."""
    combined_text = (full_ocr_text + " " + front_text).lower()
    if not combined_text or not ingredients_raw:
        return {"fake_claim_detected": False, "hidden_ingredients": []}

    text_lower = combined_text
    ingredients_lower = ingredients_raw.lower()

    marketing_claims = [
        "no added sugar", "zero sugar", "sugar free", "no sugar",
        "100% natural", "no artificial colors", "no preservatives",
    ]

    claim_found = False
    for claim in marketing_claims:
        if claim in text_lower:
            claim_found = True
            break

    if not claim_found:
        return {"fake_claim_detected": False, "hidden_ingredients": []}

    hidden_sugars = [
        "maltodextrin", "dextrose", "fructose", "glucose syrup",
        "corn syrup", "invert sugar", "date syrup", "cane juice", "jaggery", "honey",
    ]

    hidden_colors = [
        "e102", "e110", "e122", "e129", "e133", "e150", "e171",
        "ins 102", "ins 110", "ins 122", "ins 129", "ins 133",
        "tartrazine", "sunset yellow", "carmine",
    ]

    lies_found = []

    if any(c in text_lower for c in ["no added sugar", "sugar free", "zero sugar", "no sugar"]):
        for sugar in hidden_sugars:
            if sugar in ingredients_lower:
                lies_found.append(sugar.capitalize())

    if "no artificial colors" in text_lower:
        for color in hidden_colors:
            if color in ingredients_lower:
                lies_found.append(color.upper())

    if lies_found:
        return {"fake_claim_detected": True, "hidden_ingredients": lies_found}

    return {"fake_claim_detected": False, "hidden_ingredients": []}


def apply_dna_overrides(
    full_ocr_text: str,
    nutrients: dict,
    ingredients_raw: str,
    base_score: int,
    category: str = "unknown",
    front_text: str = "",
) -> dict:
    """THE MASTER OVERRIDE FUNCTION"""
    final_verdicts = []

    # 1. Atwater Math Check (BLOCK level)
    math_check = atwater_math_check(nutrients, category=category)
    if not math_check["is_valid"]:
        return {
            "action": "BLOCK",
            "score": 0,
            "reason": f"❌ CANNOT SCORE: {math_check['reason']}",
            "extra_flags": [],
        }

    # 2. Lie Detector (OVERRIDE level - Score 2)
    lie_check = detect_fake_claims(full_ocr_text, ingredients_raw, front_text=front_text)
    if lie_check["fake_claim_detected"]:
        hidden_str = ", ".join(lie_check["hidden_ingredients"])
        return {
            "action": "OVERRIDE",
            "score": 2,
            "reason": f"🚨 FAKE CLAIM: Brand claims healthy marketing, but contains {hidden_str}.",
            "extra_flags": [],
        }

    # 3. NOVA 4 (PASS level - Capped Score 3)
    nova_check = detect_nova_4(ingredients_raw)
    score = base_score
    if nova_check["is_nova_4"]:
        flags_str = ", ".join(nova_check["flags_found"])
        final_verdicts.append(f"⚠️ NOVA 4 Ultra-Processed (Contains: {flags_str})")
        if score > 3:
            score = 3

    return {
        "action": "PASS",
        "score": score,
        "reason": None,
        "extra_flags": final_verdicts,
    }
