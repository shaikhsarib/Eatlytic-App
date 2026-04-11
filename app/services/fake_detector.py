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
            
            diff_percent = abs(label_calories - calculated) / label_calories * 100 if label_calories > 0 else 100
            
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

# Per-category Atwater tolerance overrides
# High-variance products (noodles, snacks, spices) can have wider labeling error
CATEGORY_TOLERANCES: dict[str, float] = {
    "noodle": 35.0,  "noodles": 35.0, "instant noodles": 35.0,
    "snack":  35.0,  "chips":   35.0, "biscuit": 35.0, "cracker": 35.0,
    "spice":  40.0,  "condiment": 40.0, "sauce": 40.0,
}
DEFAULT_TOLERANCE = 25.0   # stricter for unknown / general products


def atwater_math_check(nutrients: dict, category: str = "unknown") -> dict:
    """
    Convenience wrapper used by tests and the unified pipeline.
    Differs from FakeDetector.validate() in two ways:
      1. Uses label_calories as the denominator (not calculated) for % diff.
      2. Checks sub-component hierarchy integrity before Atwater math.

    Returns {"is_valid": bool, "reason": str}
    """
    # ── 1. Hierarchy integrity: sub-components must not exceed parent totals ──
    carbs   = float(nutrients.get("carbs", 0) or nutrients.get("carbohydrate", 0) or 0)
    sugar   = float(nutrients.get("sugar",  0) or 0)
    fiber   = float(nutrients.get("fiber",  0) or nutrients.get("fibre", 0) or 0)
    protein = float(nutrients.get("protein", 0) or 0)
    fat     = float(nutrients.get("fat",    0) or nutrients.get("total_fat", 0) or 0)
    sat_fat = float(nutrients.get("saturated_fat", 0) or 0)

    if carbs > 0 and (sugar + fiber) > carbs * 1.05:   # 5 % rounding slack
        return {
            "is_valid": False,
            "reason": (
                f"Integrity Failure: Sugar ({sugar}g) + Fiber ({fiber}g) "
                f"exceed Carbs ({carbs}g). Likely double-counting."
            ),
        }
    if fat > 0 and sat_fat > fat * 1.05:
        return {
            "is_valid": False,
            "reason": (
                f"Integrity Failure: Saturated Fat ({sat_fat}g) > "
                f"Total Fat ({fat}g)."
            ),
        }
    
    # ── 1.1 Gross Weight Integrity: Sum of macros cannot exceed 100g ──────—
    # (Allowing 3g slack for rounding errors on the label)
    macro_sum = protein + carbs + fat
    if macro_sum > 103.0:
        return {
            "is_valid": False,
            "reason": (
                f"Gross Integrity Failure: Macros ({macro_sum:.1f}g) exceed 100g. "
                "This is physically impossible for a 100g portion."
            ),
        }

    # ── 2. Map to FakeDetector key names ──────────────────────────────────────
    fd = dict(nutrients)
    if "carbs" in fd and "carbohydrate" not in fd:
        fd["carbohydrate"] = fd.pop("carbs")

    label_calories = float(
        fd.get("calories") or fd.get("energy") or fd.get("kcal") or 0
    )

    if label_calories == 0:
        # Nothing to validate
        return {"is_valid": True, "reason": "No calorie data to validate."}

    # ── 3. Atwater calculation (sub-components excluded by FakeDetector) ───────
    detector = FakeDetector(tolerance_percent=100)   # we apply our own tolerance
    calculated = detector.calculate_expected_calories(fd)

    if calculated == 0:
        return {"is_valid": True, "reason": "No primary macros to validate against."}

    # ── 4. % diff relative to label (not calculated) ──────────────────────────
    diff_pct = abs(label_calories - calculated) / label_calories * 100
    tolerance = CATEGORY_TOLERANCES.get(category.lower().strip(), DEFAULT_TOLERANCE)

    if diff_pct <= tolerance:
        return {
            "is_valid": True,
            "reason": (
                f"✅ Math valid ({diff_pct:.1f}% within {tolerance:.0f}% tolerance)"
            ),
        }
    
    # HARDEN: If calculated > label, it's a major safety violation (undercounting calories)
    # If calculated < label, it might be due to missing components (ash, water, polyols)
    if calculated > label_calories * 1.5:
        return {
            "is_valid": False,
            "reason": (
                f"Safety Failure: Calculated energy ({calculated:.0f} kcal) is significantly HIGHER "
                f"than the label says ({label_calories:.0f} kcal). This product is likely being marketed as lower calorie than it is."
            ),
        }

    return {
        "is_valid": False,
        "reason": (
            f"Math Mismatch: Label={label_calories:.0f} kcal, "
            f"Calculated={calculated:.0f} kcal ({diff_pct:.1f}% off, "
            f"limit {tolerance:.0f}%)"
        ),
    }


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

    hidden_sugars = ["maltodextrin", "dextrose", "fructose", "glucose syrup", "corn syrup", "invert sugar", "date syrup", "cane juice", "jaggery", "honey", "sucrose", "malts"]
    hidden_colors = ["e102", "e110", "e122", "e127", "e129", "e133", "e150", "e171", "ins 102", "ins 110", "ins 122", "ins 127", "ins 129", "ins 133", "tartrazine", "sunset yellow", "carmine", "allura red"]

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
    """THE MASTER OVERRIDE FUNCTION

    BUG FIX: Previously called both FakeDetector.validate() AND atwater_math_check(),
    which caused double-validation and incorrectly blocked valid labels (e.g.
    single-ingredient products where calculated calories = 0 were flagged as MATH_MISMATCH).
    Now uses only atwater_math_check() which correctly handles:
      - per-category tolerance (noodles/snacks/spices get wider tolerance)
      - sub-component hierarchy (sugar+fiber must not exceed carbs)
      - zero-calorie pass-through
    """
    final_verdicts = []

    # 1. Atwater Math Check (single, correct implementation)
    math_ok = atwater_math_check(nutrients, category)
    if not math_ok["is_valid"]:
        return {
            "action": "BLOCK",
            "score": 0,
            "reason": f"❌ CANNOT SCORE: {math_ok['reason']}",
            "extra_flags": [],
        }

    # 2. Lie Detector (OVERRIDE level — Score 2)
    lie_check = detect_fake_claims(full_ocr_text, ingredients_raw, front_text=front_text)
    if lie_check["fake_claim_detected"]:
        hidden_str = ", ".join(lie_check["hidden_ingredients"])
        return {
            "action": "OVERRIDE",
            "score": 2,
            "reason": f"🚨 FAKE CLAIM: Brand claims healthy marketing, but contains {hidden_str}.",
            "extra_flags": [],
        }

    # 3. NOVA 4 (PASS level — Capped Score 3)
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
        "reason": math_ok.get("reason"),
        "extra_flags": final_verdicts,
    }
