"""
app/services/alternatives.py
The Eatlytic "Ingredient-Pivot" Alternative Engine. 
Converts product categories and risk flags into behavioral healthier swaps.
"""
import logging

logger = logging.getLogger(__name__)

def get_healthy_alternative(product_category: str, persona: str = "general") -> str:
    """
    Takes the category and returns an ingredient-pivot suggestion.
    Tailored for the Indian market and specific personas (Diabetic/Athlete).
    """
    
    # Standardize inputs
    p_cat = str(product_category or "").lower().strip()
    p_pers = str(persona or "general").lower().strip()

    # The Indian Alternative Matrix
    alternatives_map = {
        "biscuit": {
            "default_alt": "Look for biscuits made with Whole Wheat (Atta) or Oats, baked (not fried), with less than 5g added sugar per 100g.",
            "diabetic_alt": "Avoid biscuits entirely. Snack on a handful of roasted almonds or walnuts instead."
        },
        "noodle": {
            "default_alt": "Switch to plain Vermicelli (Sevai) or Oats Poha. Add your own vegetables and salt to control sodium.",
            "diabetic_alt": "Strictly avoid instant noodles. They spike blood sugar faster than table sugar."
        },
        "chip": {
            "default_alt": "Look for baked snacks, roasted Makhana (fox nuts), or roasted chana. Check for 'Vacuum Fried' on the label."
        },
        "beverage": {
            "default_alt": "Switch to plain water, lemon water, or green tea. If you need energy, eat a banana—it's healthier than any energy drink.",
            "athlete_alt": "Look for drinks with Electrolytes and <10g natural sugar, no Maltodextrin."
        },
        "juice": {
            "default_alt": "Eat the whole fruit instead. If you must buy juice, look for '100% Cold Pressed' with ZERO added sugar or preservatives."
        },
        "dairy": {
            "default_alt": "Switch to plain, unsweetened dahi (curd) or milk. Add fresh fruits at home for sweetness."
        },
        "chocolate": {
            "default_alt": "Switch to Dark Chocolate (>70% Cocoa) with less than 15g sugar per 100g. Avoid 'Milk Chocolate' with hydrogenated fats."
        },
        "protein_supplement": {
            "default_alt": "Look for 'Whey Protein Isolate' with <3g carbs/sugar per scoop, and no Maltodextrin in the ingredients.",
            "athlete_alt": "Ensure it has a complete Amino Acid profile. Avoid 'Protein Blends' that hide cheap fillers."
        },
        "ready_to_eat": {
            "default_alt": "Look for 'Retort Processed' meals without preservatives. Switch to fresh home-cooked Dal-Chawal for better fiber."
        },
        "sweet": {
            "default_alt": "Enjoy traditional sweets made with Jaggery or Dates in moderation. Avoid those with 'Liquid Glucose' or Artificial Colors."
        }
    }

    # 1. Check if we have a rule for this category
    category_data = alternatives_map.get(p_cat)
    
    if not category_data:
        return "🥗 Healthier Swap: Choose whole foods with <5 ingredients, no added sugar, and <400mg sodium per 100g."

    # 2. Check for Persona-Specific Override (Crucial for Diabetics!)
    if "diabetic" in p_pers and "diabetic_alt" in category_data:
        return f"🥗 Alternative (For Diabetics): {category_data['diabetic_alt']}"

    if ("athlete" in p_pers or "gym" in p_pers) and "athlete_alt" in category_data:
        return f"🥗 Alternative (For Athletes): {category_data['athlete_alt']}"

    # 3. Default category alternative
    return f"🥗 Healthier Swap: {category_data['default_alt']}"
