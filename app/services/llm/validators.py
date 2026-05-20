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

def compute_extraction_confidence(
    result_data: dict,
    ocr_word_count: int,
    avg_ocr_confidence: float,
    atwater_valid: bool,
    nutrient_count: int,
) -> dict:
    score = 100

    if ocr_word_count < 10:
        score -= 25
    elif ocr_word_count < 20:
        score -= 10
    if avg_ocr_confidence < 0.5:
        score -= 25
    elif avg_ocr_confidence < 0.7:
        score -= 10

    if nutrient_count == 0:
        score -= 50
    elif nutrient_count < 4:
        score -= 25
    elif nutrient_count < 7:
        score -= 10

    if not atwater_valid:
        score -= 20

    required = ["energy", "protein", "fat", "carb"]
    names_lower = [n.get("name", "").lower() for n in result_data.get("nutrients", [])]
    missing = [r for r in required if not any(r in nm for nm in names_lower)]
    score -= len(missing) * 10

    bad_names = {"unknown", "unknown product", "food product", "", "n/a"}
    if result_data.get("product_name", "").lower().strip() in bad_names:
        score -= 15

    if nutrient_count >= 6 and atwater_valid:
        score += 25
    elif nutrient_count >= 4:
        score += 15
        
    if "No recent web data" not in result_data.get("summary", "") and nutrient_count >= 5:
        score += 15

    score = max(0, min(100, score))

    if score >= 80:
        tier, message = "HIGH", "Extracted from clear label — high confidence."
    elif score >= 55:
        tier, message = "MEDIUM", "Some uncertainty — verify key nutrients against label."
    elif score >= 18:
        tier, message = "LOW", "Partial extraction — image may be blurry or label poorly read."
    else:
        tier, message = "UNRELIABLE", "Could not reliably read this label. Please retake the photo."

    return {
        "score": score,
        "tier": tier,
        "message": message,
        "nutrient_count": nutrient_count,
        "atwater_valid": atwater_valid,
    }
