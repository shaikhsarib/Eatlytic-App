"""
app/services/fake_detector.py
Eatlytic DNA: Proprietary detectors for Marketing Lies, NOVA 4, and Atwater Physics.
"""

import re
import logging

logger = logging.getLogger(__name__)


# ==========================================
# FAKE TYPE 3: THE PHYSICAL COUNTERFEIT CHECK (Atwater Physics)
# ==========================================
def atwater_math_check(nutrients: dict) -> dict:
    """
    Checks if label macro math follows physics (Atwater 4-4-9).
    Now returns a warning instead of a block to follow 'Blind Photocopier' rule.
    """
    try:
        protein = float(nutrients.get("protein", 0) or 0)
        carbs = float(nutrients.get("carbs", 0) or 0)
        fat = float(nutrients.get("fat", 0) or 0)
        stated_calories = float(nutrients.get("calories", 0) or 0)
    except (ValueError, TypeError):
        return {"is_valid": False, "reason": "Could not read macro numbers."}

    if stated_calories <= 0:
        return {"is_valid": True, "reason": None}

    calculated_calories = (protein * 4) + (carbs * 4) + (fat * 9)
    margin = stated_calories * 0.25
    lower_bound = stated_calories - margin
    upper_bound = stated_calories + margin

    if not (lower_bound <= calculated_calories <= upper_bound):
        return {
            "is_valid": False, 
            "reason": f"Atwater Mismatch: Label says {stated_calories} kcal, but macros calculate to {calculated_calories:.0f} kcal."
        }

    return {"is_valid": True, "reason": None}


def check_for_impossible_labels(nutrients: dict, product_name: str) -> list:
    """
    Identifies physically impossible data (e.g., fried food with 0g fat).
    Returns a list of warning strings. (CEO'S P0 'Blind Photocopier' Rule)
    """
    warnings = []
    
    try:
        fat = float(nutrients.get('fat', 0) or 0)
        calories = float(nutrients.get('calories', 0) or 0)
        sodium = float(nutrients.get('sodium', 0) or 0)
        protein = float(nutrients.get('protein', 0) or 0)
    except (ValueError, TypeError):
        return []

    # Example 1: Fried food with 0 fat
    # Indian context: 'noodle', 'chip', 'namkeen', 'bhujia'
    p_lower = product_name.lower()
    if any(k in p_lower for k in ["noodle", "chip", "namkeen", "bhujia", "fried"]):
        if calories > 200 and fat == 0:
            warnings.append("⚠️ Label Anomaly: High calories but 0g fat detected. This is physically impossible for fried snacks. Verify label manually.")
            
    # Example 2: Savory snack with 0 sodium
    if sodium == 0 and calories > 50 and any(k in p_lower for k in ["noodle", "chip", "namkeen", "salted", "savory"]):
        warnings.append("⚠️ Label Anomaly: 0mg sodium detected. This is extremeley rare for savory snacks. Verify label manually.")

    # Example 3: Essential logic for 'Photocopier' safety (Meat/Protein)
    if any(k in p_lower for k in ["chicken", "meat", "fish", "egg"]) and protein == 0 and calories > 50:
        warnings.append("⚠️ Label Anomaly: 0g protein detected in a meat-based product. Please double-check the label.")
        
    return warnings


# ==========================================
# FAKE TYPE 2: THE HIDDEN TRUTH (NOVA 4)
# ==========================================
def detect_nova_4(ingredients_raw: str) -> dict:
    """
    Scans ingredients for ultra-processed markers (Emulsifiers, Colorants, Flavor Enhancers).
    Uses regex for E-numbers and INS numbers common in the Indian market.
    """
    if not ingredients_raw:
        return {"is_nova_4": False, "flags_found": []}

    # Standardized list of Indian UPF markers (INS numbers and names)
    upf_blacklist = [
        r"\be\d{3,4}[a-z]?\b",  # Matches ANY E-number (e.g., E471, E621)
        r"ins\s*\d{3,4}[a-z]?",  # Matches INS numbers (Indian standard)
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
    # Make lowercase for matching
    ingredients_lower = ingredients_raw.lower()

    for pattern in upf_blacklist:
        matches = re.findall(pattern, ingredients_lower)
        if matches:
            # Capitalize the found match to look good in output
            f = (
                matches[0].upper()
                if ("e" in matches[0] or "ins" in matches[0])
                else matches[0].capitalize()
            )
            flags_found.append(f)

    # Deduplicate flags
    flags_found = list(set(flags_found))

    # NOVA 4 threshold: If 2 or more UPF markers are found
    is_nova_4 = len(flags_found) >= 2

    return {"is_nova_4": is_nova_4, "flags_found": flags_found}


# ==========================================
# FAKE TYPE 1: THE "LEGAL" LIE (Lie Detector)
# ==========================================
def detect_fake_claims(
    full_ocr_text: str, ingredients_raw: str, front_text: str = ""
) -> dict:
    """
    Checks if the front of the pack claims "No Sugar" but the back contains hidden sugars.
    front_text should contain any marketing claims from the front of the package.
    """
    combined_text = (full_ocr_text + " " + front_text).lower()
    if not combined_text or not ingredients_raw:
        return {"fake_claim_detected": False, "hidden_ingredients": []}

    text_lower = combined_text
    ingredients_lower = ingredients_raw.lower()

    # 1. What is the brand claiming?
    marketing_claims = [
        "no added sugar",
        "zero sugar",
        "sugar free",
        "no sugar",
        "100% natural",
        "no artificial colors",
        "no preservatives",
    ]

    claim_found = False
    for claim in marketing_claims:
        if claim in text_lower:
            claim_found = True
            break

    if not claim_found:
        return {"fake_claim_detected": False, "hidden_ingredients": []}

    # 2. What is actually hiding in the ingredients?
    hidden_sugars = [
        "maltodextrin",
        "dextrose",
        "fructose",
        "glucose syrup",
        "corn syrup",
        "invert sugar",
        "date syrup",
        "cane juice",
        "jaggery",
        "honey",
    ]

    hidden_colors = [
        "e102",
        "e110",
        "e122",
        "e129",
        "e133",
        "e150",
        "e171",
        "ins 102",
        "ins 110",
        "ins 122",
        "ins 129",
        "ins 133",
        "tartrazine",
        "sunset yellow",
        "carmine",
    ]

    lies_found = []

    # Check for sugar lies
    if any(
        c in text_lower
        for c in ["no added sugar", "sugar free", "zero sugar", "no sugar"]
    ):
        for sugar in hidden_sugars:
            if sugar in ingredients_lower:
                lies_found.append(sugar.capitalize())

    # Check for color lies
    if "no artificial colors" in text_lower:
        for color in hidden_colors:
            if color in ingredients_lower:
                lies_found.append(color.upper())

    # If they made a claim, but we found hidden bad stuff, it's a LIE
    if lies_found:
        return {"fake_claim_detected": True, "hidden_ingredients": lies_found}

    return {"fake_claim_detected": False, "hidden_ingredients": []}


# ==========================================
# THE MASTER OVERRIDE FUNCTION
# ==========================================
def apply_dna_overrides(
    full_ocr_text: str,
    nutrients: dict,
    ingredients_raw: str,
    base_score: int,
    front_text: str = "",
    product_name: str = "product",
) -> dict:
    """
    Master function to apply Eatlytic DNA rules.
    Following the 'Blind Photocopier' rule, we NO LONGER hard-block on math/anomalies.
    We instead surface clearly marked SYSTEM WARNINGS.
    """
    final_verdicts = []
    anomaly_warnings = []

    # 1. Check for Impossible Labels (Photocopier Rule Warnings)
    anomalies = check_for_impossible_labels(nutrients, product_name)
    if anomalies:
        anomaly_warnings.extend(anomalies)

    # 2. Run Math Check - Now a WARNING, not a BLOCK
    math_check = atwater_math_check(nutrients)
    if not math_check["is_valid"]:
        anomaly_warnings.append(f"⚠️ {math_check['reason']} (Data extraction matches printed label).")

    # 3. Run Fake Type 1 (Lie Detector) - OVERRIDE level (Score 2)
    lie_check = detect_fake_claims(
        full_ocr_text, ingredients_raw, front_text=front_text
    )
    if lie_check["fake_claim_detected"]:
        hidden_str = ", ".join(lie_check["hidden_ingredients"])
        return {
            "action": "OVERRIDE",
            "score": 2,
            "reason": f"🚨 FAKE CLAIM: Brand claims healthy marketing, but contains {hidden_str}.",
            "extra_flags": final_verdicts,
            "anomaly_warnings": anomaly_warnings,
        }

    # 4. Run Fake Type 2 (NOVA 4) - PASS level (Capped Score 3)
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
        "anomaly_warnings": anomaly_warnings,
    }
