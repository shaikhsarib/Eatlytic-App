"""
app/services/duel_service.py
Head-to-head comparison logic to declare a winner between two products.
"""

import logging

logger = logging.getLogger(__name__)

def compare_nutrients(val_a, val_b, higher_is_better=False):
    """Simple comparison helper."""
    if val_a == val_b: return "equal"
    if higher_is_better:
        return "a" if val_a > val_b else "b"
    else:
        return "a" if val_a < val_b else "b"

def run_duel(product_a: dict, product_b: dict, persona: str = "general") -> dict:
    """
    Compare two products and return a winner with side-by-side metrics.
    product_a/b are the dictionaries returned by get_scan_by_id.
    """
    
    # Weighting matrix based on persona
    weights = {
        "muscle": {"protein": 3.0, "calories": 1.0, "sugar": -1.0},
        "weight_loss": {"calories": -3.0, "sugar": -2.0, "fiber": 1.0},
        "diabetic": {"sugar": -5.0, "carbs": -1.0, "fiber": 2.0},
        "general": {"score": 3.0, "sugar": -2.0, "protein": 1.0}
    }
    
    current_weights = weights.get(persona, weights["general"])
    
    score_a = 0
    score_b = 0
    
    # 1. Compare Eatlytic Scores
    if product_a["score"] > product_b["score"]: score_a += current_weights.get("score", 3.0)
    elif product_b["score"] > product_a["score"]: score_b += current_weights.get("score", 3.0)
    
    # 2. Compare Macros
    metrics = [
        ("protein", True),  # higher is better
        ("sugar", False),   # lower is better
        ("calories", False), # lower is better (usually)
        ("fiber", True),
        ("fat", False),
    ]
    
    comparison_details = []
    
    for key, higher_better in metrics:
        val_a = product_a.get(key, 0) or 0
        val_b = product_b.get(key, 0) or 0
        
        winner = compare_nutrients(val_a, val_b, higher_better)
        weight = abs(current_weights.get(key, 1.0))
        
        if winner == "a": score_a += weight
        elif winner == "b": score_b += weight
        
        comparison_details.append({
            "metric": key.capitalize(),
            "val_a": val_a,
            "val_b": val_b,
            "winner": winner,
            "unit": "g" if key != "calories" else "kcal"
        })

    # 3. Determine Final Winner
    winner_id = product_a["id"] if score_a >= score_b else product_b["id"]
    winner_name = product_a["product_name"] if score_a >= score_b else product_b["product_name"]
    
    # 4. Generate "The Edge" - Why did they win?
    edge = ""
    if winner_id == product_a["id"]:
        if product_a["score"] > product_b["score"] + 2:
            edge = f"{product_a['product_name']} is significantly cleaner."
        elif product_a["sugar"] < product_b["sugar"] - 5:
            edge = f"Much lower sugar content in {product_a['product_name']}."
        else:
            edge = f"{product_a['product_name']} has a better overall nutrient density."
    else:
        if product_b["score"] > product_a["score"] + 2:
            edge = f"{product_b['product_name']} is the clear technical winner."
        elif product_b["protein"] > product_a["protein"] + 2:
            edge = f"Better protein-to-calorie ratio in {product_b['product_name']}."
        else:
            edge = f"{product_b['product_name']} is slightly better for your goals."

    return {
        "winner_id": winner_id,
        "winner_name": winner_name,
        "edge": edge,
        "score_a": round(score_a, 1),
        "score_b": round(score_b, 1),
        "comparison": comparison_details,
        "side_by_side": {
            "a": {
                "id": product_a["id"],
                "name": product_a["product_name"],
                "score": product_a["score"],
                "verdict": product_a["verdict"]
            },
            "b": {
                "id": product_b["id"],
                "name": product_b["product_name"],
                "score": product_b["score"],
                "verdict": product_b["verdict"]
            }
        }
    }
