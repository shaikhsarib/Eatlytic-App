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

def get_nova_level(ingredients_raw: str) -> int:
    """
    NOVA Classification logic:
    1: Unprocessed / Minimally processed
    2: Processed culinary ingredients
    3: Processed foods
    4: Ultra-processed foods (UPF)
    """
    if not ingredients_raw:
        return 1
    
    ingredients_lower = ingredients_raw.lower()
    
    # Industrial markers (emulsifiers, flavors, etc.)
    upf_markers = [
        "emulsifier", "stabilizer", "maltodextrin", "high fructose",
        "artificial", "sweetener", "flavor enhancer", "ins", "e-",
        "hydrolyzed", "hydrogenated", "color", "thickener"
    ]
    
    marker_count = 0
    for marker in upf_markers:
        if marker in ingredients_lower:
            marker_count += 1
            
    # Simple heuristic
    if marker_count >= 5:
        return 4
    if marker_count >= 2:
        return 3
    if marker_count >= 1:
        return 2
    return 1

def get_nutriscore_verdict(nutrients: dict) -> str:
    """Returns a color indicator based on critical nutrients (Sugar/Salt)."""
    sugar = nutrients.get("sugar", 0)
    sodium = nutrients.get("sodium", 0) or nutrients.get("sodium_mg", 0)
    
    # Per 100g thresholds (Conservative Indian context)
    if sugar > 22.5 or sodium > 600:
        return "🔴 RED (HIGH)"
    if sugar > 5.0 or sodium > 100:
        return "🟡 AMBER (MEDIUM)"
    return "🟢 GREEN (LOW)"

def generate_humanized_insights(nutrients: dict) -> list:
    """
    Conversion logic for human-readable insights.
    """
    insights = []
    
    # 1. Sugar to Teaspoons (1 tsp = ~4g)
    sugar = nutrients.get("sugar", 0)
    if sugar > 0:
        tsps = round(sugar / 4, 1)
        insights.append(f"🍬 Contains ~{tsps} teaspoons of sugar per 100g.")
        
    # 2. Salt Limit (Sodium to Salt: 1g Sodium = 2.5g Salt)
    sodium_mg = nutrients.get("sodium", 0) or nutrients.get("sodium_mg", 0)
    if sodium_mg > 0:
        salt_g = (sodium_mg / 1000) * 2.5
        daily_limit_pct = round((sodium_mg / INDIAN_RDA["sodium_mg"]) * 100)
        insights.append(f"🧂 Provides {daily_limit_pct}% of your daily Indian salt limit per 100g.")
        
    # 3. Activity to Burn Calories (Walking: 100 kcal = ~20-30 mins)
    calories = nutrients.get("calories", 0)
    if calories > 0:
        walking_mins = round((calories / 100) * 25)
        insights.append(f"🚶 You'd need to walk for ~{walking_mins} minutes to burn this off.")
        
    return insights

def get_explanation_report(nutrients: dict, ingredients_raw: str) -> dict:
    """Assemble the full explanation."""
    return {
        "nova_level": get_nova_level(ingredients_raw),
        "verdict": get_nutriscore_verdict(nutrients),
        "humanized_insights": generate_humanized_insights(nutrients),
        "rda_pct": {
            "energy": round((nutrients.get("calories", 0) / INDIAN_RDA["energy"]) * 100, 1),
            "protein": round((nutrients.get("protein", 0) / INDIAN_RDA["protein"]) * 100, 1),
            "sodium": round(((nutrients.get("sodium", 0) or nutrients.get("sodium_mg", 0)) / INDIAN_RDA["sodium_mg"]) * 100, 1),
        }
    }
