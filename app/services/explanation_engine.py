"""
app/services/explanation_engine.py
The Brain: ICMR 2020 RDA, NOVA Classification, and Humanized Insights.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ICMR 2020 Reference Values for Indian Adults
INDIAN_RDA = {
    "energy": 2000,          # kcal
    "protein": 54,           # g (avg for 60-70kg adult)
    "total_fat": 66,         # g (30% of energy)
    "added_sugar": 25,       # g (5-10% of energy)
    "sodium_mg": 2000,       # mg
}

# Cultural food benchmarks for humanized comparison
CULTURAL_EQUIVALENTS = {
    "chapati": {"name": "medium Chapatis", "calories": 100},
    "samosa": {"name": "Samosas", "calories": 250},
    "chai": {"name": "cups of cutting Chai", "calories": 70},
}

# Database of common Indian food additives and their health context
INS_DATABASE = {
    "635": "Flavor enhancer (Disodium 5'-ribonucleotides). Can trigger allergic reactions in sensitive individuals.",
    "150d": "Ammonia Caramel color. Highly processed coloring agent often found in colas and noodles.",
    "451": "Humectant/Stabilizer (Pentasodium triphosphate). Used to retain moisture but high in phosphorus.",
    "621": "MSG (Monosodium Glutamate). Potential trigger for headaches or sensitivity in some people.",
    "951": "Aspartame (Artificial sweetener). Flagged for long-term health monitoring.",
    "211": "Sodium Benzoate (Preservative). Common in beverages; best avoided in high quantities.",
}

from app.services.fake_detector import detect_nova_4

def get_nova_level(ingredients_raw: str) -> int:
    """
    Consolidated NOVA Classification via FakeDetector.
    """
    res = detect_nova_4(ingredients_raw)
    return 4 if res["is_nova_4"] else 3 if len(res["flags_found"]) > 0 else 1

def identify_additives(ingredients_raw: str) -> list:
    """Extracts INS/E numbers and provides context."""
    if not ingredients_raw:
        return []
    
    found = []
    # Match patterns like INS 635, E-150, or just 635 in ingredient lists
    for code, desc in INS_DATABASE.items():
        if re.search(rf"\b(ins|e)?\s*-?{code}\b", ingredients_raw.lower()):
            found.append(f"🧪 {desc}")
    return found

def get_persona_advice(nutrients: dict, ingredients: str) -> list:
    """Generates specific advice for different personas (Children, Diabetic, etc)."""
    advice = []
    sodium = nutrients.get("sodium_mg", 0) or nutrients.get("sodium", 0)
    sugar = nutrients.get("sugar", 0)
    ing_lower = str(ingredients or "").lower()

    # 1. Kids Advice
    if sodium > 600:
        advice.append({
            "persona": "Children under 12",
            "type": "WARNING",
            "msg": "Kidneys are still developing — high sodium is actively harmful. Limit to once a fortnight maximum."
        })
    
    # 2. Diabetic Advice
    if sugar > 15 or "maida" in ing_lower or "starch" in ing_lower:
        advice.append({
            "persona": "Diabetics",
            "type": "CAUTION",
            "msg": "High glycemic index (Sugar/Maida) causes sharp blood sugar spikes. Pair with vegetables and protein if eating."
        })

    # 3. Seniors Advice
    if sodium > 400:
        advice.append({
            "persona": "Seniors 60+",
            "type": "WARNING",
            "msg": "High sodium raises blood pressure directly. Strongly avoid if hypertensive or on heart medication."
        })

    # 4. Pregnant Advice
    if "trans fat" in ing_lower or nutrients.get("trans_fat", 0) > 0 or "vanaspati" in ing_lower:
        advice.append({
            "persona": "Pregnant",
            "type": "WARNING",
            "msg": "Trans fat (Vanaspati) crosses the placenta and is harmful. High sodium contributes to pregnancy hypertension. Strictly avoid."
        })

    # 5. Athlete Advice
    protein = nutrients.get("protein", 0)
    if protein < 5 and ("snack" in ing_lower or "noodle" in ing_lower):
        advice.append({
            "persona": "Athletes",
            "type": "CAUTION",
            "msg": "Empty calories. Very low protein content will not help with recovery or muscle synthesis."
        })
    elif protein >= 15:
        advice.append({
            "persona": "Athletes",
            "type": "GOOD",
            "msg": "High protein content detected. Good for muscle recovery."
        })

    return advice

def generate_humanized_insights(nutrients: dict, ingredients: str) -> list:
    """Conversion logic for human-readable insights with Indian context."""
    insights = []
    
    # 1. Sugar to Teaspoons
    sugar = nutrients.get("sugar", 0)
    if sugar > 0:
        tsps = round(sugar / 4, 1)
        insights.append(f"🍬 Contains ~{tsps} teaspoons of sugar per 100g.")
        
    # 2. Salt Limit (Sodium % of RDA)
    sodium_mg = nutrients.get("sodium_mg", 0) or nutrients.get("sodium", 0)
    if sodium_mg > 0:
        daily_limit_pct = round((sodium_mg / INDIAN_RDA["sodium_mg"]) * 100)
        insights.append(f"🧂 Provides {daily_limit_pct}% of your daily Indian salt limit per 100g.")
        
    # 3. Cultural calorie comparison (Dynamic)
    calories = nutrients.get("calories", 0)
    if calories > 70:
        # Choose the most relatable benchmark based on calories
        if calories > 250:
            key = "samosa"
        elif calories > 120:
            key = "chapati"
        else:
            key = "chai"
            
        bench = CULTURAL_EQUIVALENTS[key]
        count = round(calories / bench["calories"], 1)
        insights.append(f"⚖️ This portion is equivalent to ~{count} {bench['name']} in calories.")

    # 4. Additive scanning
    additives = identify_additives(ingredients)
    insights.extend(additives)
        
    return insights

def get_explanation_report(nutrients: dict, ingredients_raw: str) -> dict:
    """Assemble the full explanation with persona and cultural insights."""
    sugar = nutrients.get("sugar", 0)
    sodium = nutrients.get("sodium_mg", 0) or nutrients.get("sodium", 0)
    
    if sugar > 22.5 or sodium > 600:
        verdict = "🔴 RED (HIGH)"
    elif sugar > 5.0 or sodium > 100:
        verdict = "🟡 AMBER (MEDIUM)"
    else:
        verdict = "🟢 GREEN (LOW)"

    return {
        "nova_level": get_nova_level(ingredients_raw),
        "verdict": verdict,
        "humanized_insights": generate_humanized_insights(nutrients, ingredients_raw),
        "persona_warnings": get_persona_advice(nutrients, ingredients_raw),
        "rda_pct": {
            "energy": round((nutrients.get("calories", 0) / INDIAN_RDA["energy"]) * 100, 1),
            "protein": round((nutrients.get("protein", 0) / INDIAN_RDA["protein"]) * 100, 1),
            "sodium": round(((nutrients.get("sodium_mg", 0) or nutrients.get("sodium", 0)) / INDIAN_RDA["sodium_mg"]) * 100, 1),
        }
    }
