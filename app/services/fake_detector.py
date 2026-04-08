"""
app/services/fake_detector.py
Eatlytic DNA: Robust Atwater Physics, Marketing Lie Detector, and NOVA 4.
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class FakeDetector:
    """
    Atwater Physics Engine - Validates calorie math.
    
    CRITICAL RULE: Only use PRIMARY macronutrients for calculation.
    Sub-components (sugar, fiber, saturated fat, trans fat) are 
    ALREADY INCLUDED in their parent totals. Do NOT double-count them!
    """
    
    def __init__(self, tolerance_percent=35.0):
        # Tolerance for high-variance products (noodles, snacks)
        self.tolerance = tolerance_percent
        
        # Atwater energy conversion factors (kcal per gram)
        self.FACTORS = {
            'protein': 4.0,
            'carbohydrate': 4.0,
            'fat': 9.0,
            'alcohol': 7.0,
            'polyols': 2.4,
        }
        
        # List of sub-component keys to IGNORE (they're part of parent)
        self.SUB_COMPONENTS = [
            'sugar', 'total_sugars', 'added_sugars', 
            'added_sugar', 'fiber', 'dietary_fiber',
            'saturated_fat', 'saturated_fatty_acids',
            'trans_fat', 'trans_fatty_acids',
            'monounsaturated_fat', 'polyunsaturated_fat',
            'mufa', 'pufa',
            'starch', 'glycogen',
            'sodium', 'potassium', 'cholesterol',
            'iron', 'calcium', 'vitamin',  # Minerals/vitamins = 0 kcal
        ]
        
        # Aliases mapping (normalize key names)
        self.ALIASES = {
            'carbs': 'carbohydrate',
            'total_carbohydrate': 'carbohydrate',
            'total_fat': 'fat',
            'energy': 'calories',
            'kcal': 'calories',
            'kilocalories': 'calories',
        }
    
    def normalize_key(self, key: str) -> str:
        """Convert various naming conventions to standard form."""
        key_lower = key.lower().strip()
        key_lower = re.sub(r'[\s\-_]+', '_', key_lower)
        return self.ALIASES.get(key_lower, key_lower)
    
    def is_sub_component(self, key: str) -> bool:
        """Check if this nutrient is a sub-component of a primary macro."""
        normalized = self.normalize_key(key)
        for sub in self.SUB_COMPONENTS:
            if sub in normalized:
                return True
        return False
    
    def extract_primary_macros(self, nutrition_data: Dict) -> Dict[str, float]:
        """
        Extract only PRIMARY macronutrients for Atwater calculation.
        """
        primary_macros = {}
        
        for key, value in nutrition_data.items():
            if not isinstance(value, (int, float, str)):
                continue
            
            try:
                val_float = float(value)
            except (ValueError, TypeError):
                continue
                
            if val_float == 0:
                continue
            
            normalized_key = self.normalize_key(key)
            
            # SKIP sub-components to prevent double-counting!
            if self.is_sub_component(normalized_key):
                continue
            
            # Only keep primary macros
            if normalized_key in ['protein', 'carbohydrate', 'fat', 'alcohol', 'polyols']:
                primary_macros[normalized_key] = val_float
        
        return primary_macros
    
    def calculate_expected_calories(self, nutrition_data: Dict) -> float:
        """
        Calculate expected calories using Atwater equation.
        """
        primaries = self.extract_primary_macros(nutrition_data)
        
        total_calories = 0.0
        for macro, value in primaries.items():
            factor = self.FACTORS.get(macro, 0)
            total_calories += value * factor
        
        return round(total_calories, 1)
    
    def validate(self, nutrition_data: Dict) -> Dict:
        """Main validation function."""
        try:
            label_calories = float(nutrition_data.get('calories') 
                                or nutrition_data.get('energy') 
                                or nutrition_data.get('kcal') or 0)
            
            if label_calories == 0:
                # Special case: check if macros are also zero
                calculated = self.calculate_expected_calories(nutrition_data)
                if calculated > 20: # Small allowance
                    return {
                        'status': 'MATH_MISMATCH',
                        'label_calories': 0,
                        'calculated_calories': calculated,
                        'message': f"Math Mismatch: Label says 0 kcal, but macros calculate to {calculated:.0f} kcal."
                    }
                return {'status': 'VALID', 'message': 'Valid zero-calorie label.'}

            calculated = self.calculate_expected_calories(nutrition_data)
            
            diff_percent = abs(label_calories - calculated) / calculated * 100 if calculated > 0 else 100
            
            if diff_percent <= self.tolerance:
                status = 'VALID'
                message = f"✅ Calorie math checks out ({diff_percent:.1f}% within {self.tolerance}% tolerance)"
            elif diff_percent <= 100:
                status = 'MATH_MISMATCH'
                message = f"⚠️ Math mismatch: Label={label_calories}, Calculated={calculated} ({diff_percent:.1f}% off)"
            else:
                status = 'IMPOSSIBLE_DATA'
                message = f"❌ Impossible: Label={label_calories} but macros suggest {calculated}"
            
            return {
                'status': status,
                'label_calories': label_calories,
                'calculated_calories': calculated,
                'difference_percent': round(diff_percent, 2),
                'message': message
            }
        except Exception as e:
            return {'status': 'ERROR', 'message': f"Validation error: {str(e)}"}

def detect_nova_4(ingredients_raw: str) -> dict:
    """Scans ingredients for ultra-processed markers."""
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
            f = (matches[0].upper() if ("e" in matches[0] or "ins" in matches[0]) else matches[0].capitalize())
            flags_found.append(f)

    flags_found = list(set(flags_found))
    is_nova_4 = len(flags_found) >= 2

    return {"is_nova_4": is_nova_4, "flags_found": flags_found}

def detect_fake_claims(full_ocr_text: str, ingredients_raw: str, front_text: str = "") -> dict:
    """Checks for hidden sugars and colors despite healthy claims."""
    combined_text = (full_ocr_text + " " + front_text).lower()
    if not combined_text or not ingredients_raw:
        return {"fake_claim_detected": False, "hidden_ingredients": []}

    ingredients_lower = ingredients_raw.lower()
    marketing_claims = ["no added sugar", "zero sugar", "sugar free", "no sugar", "100% natural", "no artificial colors", "no preservatives"]

    claim_found = any(claim in combined_text for claim in marketing_claims)
    if not claim_found:
        return {"fake_claim_detected": False, "hidden_ingredients": []}

    hidden_sugars = ["maltodextrin", "dextrose", "fructose", "glucose syrup", "corn syrup", "invert sugar", "date syrup", "cane juice", "jaggery", "honey"]
    hidden_colors = ["e102", "e110", "e122", "e129", "e133", "e150", "e171", "ins 102", "ins 110", "ins 122", "ins 129", "ins 133", "tartrazine", "sunset yellow", "carmine"]

    lies_found = []
    if any(c in combined_text for c in ["no added sugar", "sugar free", "zero sugar", "no sugar"]):
        for sugar in hidden_sugars:
            if sugar in ingredients_lower: lies_found.append(sugar.capitalize())

    if "no artificial colors" in combined_text:
        for color in hidden_colors:
            if color in ingredients_lower: lies_found.append(color.upper())

    return {"fake_claim_detected": bool(lies_found), "hidden_ingredients": lies_found}

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
    
    # 1. New Robust Math Check
    detector = FakeDetector(tolerance_percent=35.0)
    validation = detector.validate(nutrients)
    
    if validation["status"] in ["MATH_MISMATCH", "IMPOSSIBLE_DATA"]:
        return {
            "action": "BLOCK",
            "score": 0,
            "reason": f"❌ CANNOT SCORE: {validation['message']}",
            "extra_flags": [],
        }

    # 2. Lie Detector
    lie_check = detect_fake_claims(full_ocr_text, ingredients_raw, front_text=front_text)
    if lie_check["fake_claim_detected"]:
        hidden_str = ", ".join(lie_check["hidden_ingredients"])
        return {
            "action": "OVERRIDE", "score": 2,
            "reason": f"🚨 FAKE CLAIM: Brand claims healthy marketing, but contains {hidden_str}.",
            "extra_flags": [],
        }

    # 3. NOVA 4
    nova_check = detect_nova_4(ingredients_raw)
    score = base_score
    if nova_check["is_nova_4"]:
        flags_str = ", ".join(nova_check["flags_found"])
        final_verdicts.append(f"⚠️ NOVA 4 Ultra-Processed (Contains: {flags_str})")
        if score > 3: score = 3

    return {
        "action": "PASS", "score": score,
        "reason": validation["message"] if validation["status"] == "VALID" else None,
        "extra_flags": final_verdicts,
    }
