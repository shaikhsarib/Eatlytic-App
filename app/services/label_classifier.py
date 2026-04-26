"""
app/services/label_classifier.py
─────────────────────────────────────────────────────────────────────────────
Eatlytic Label Classifier  (v1)

Answers ONE question BEFORE any LLM token is spent:
  "Is this image a food/beverage nutrition label — or something else entirely?"

Why this matters
────────────────
A Dove soap back label says "Ingredients: Aqua, Sodium Lauryl Sulfate, Glycerin…"
The universal_label_filter sees "Ingredients" + numbers → passes it as VALID.
The LLM then tries to score it as food → produces a nonsense "nutrition report".
This classifier blocks that before anything expensive runs.

Detection strategy: 3-layer cascade
────────────────────────────────────
Layer 1 — Hard non-food signals (INCI names, pharma, cleaning agents)
Layer 2 — Absent food signals (no kcal / per 100g / protein etc.)
Layer 3 — Category scoring (weighted keyword vote across 8 product categories)

Returns
───────
{
  "is_non_food": bool,
  "detected_type": str,   # e.g. "cosmetics", "pharmaceutical", "cleaning_product"
  "confidence": float,    # 0.0–1.0
  "signals_found": list,  # keywords that triggered the classification
}
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Layer 1: Hard non-food signals ───────────────────────────────────────
# Any ONE of these words/phrases is a strong signal that this is NOT food.
NON_FOOD_HARD_SIGNALS: dict[str, list[str]] = {
    "cosmetics": [
        "sodium lauryl sulfate", "sodium laureth sulfate", "sls", "sles",
        "cocamidopropyl betaine", "dimethicone", "silicone",
        "paraben", "methylparaben", "ethylparaben", "propylparaben",
        "phenoxyethanol", "benzyl alcohol",
        "fragrance", "parfum",
        "inci", "dermatologist tested", "ophthalmologist tested",
        "for external use only", "not for internal use",
        "apply to skin", "lather", "rinse thoroughly", "avoid contact with eyes",
        "shampoo", "conditioner", "moisturiser", "moisturizer", "serum", "toner",
        "face wash", "body wash", "hand wash", "sunscreen", "spf",
        "retinol", "hyaluronic acid", "niacinamide", "ceramide",
        "aqua (water)", "aqua, ", "ci 77891", "ci 19140",
    ],
    "pharmaceutical": [
        "active ingredient", "inactive ingredient", "drug facts",
        "dosage", "dose", "directions for use (medicine)",
        "warnings: keep out of reach", "do not exceed recommended dose",
        "consult a physician", "consult your doctor",
        "paracetamol", "ibuprofen", "aspirin", "antibiotic",
        "tablet", "capsule", "syrup", "suspension",
        "rx only", "prescription only",
        "store below 25°c", "store below 30°c",
        "batch no", "mfg date", "exp date",   # pharma-specific combo
    ],
    "cleaning_product": [
        "surfactant", "anionic surfactant", "non-ionic surfactant",
        "bleach", "sodium hypochlorite", "ammonia",
        "disinfectant", "antibacterial formula",
        "laundry detergent", "dishwashing", "floor cleaner",
        "kills 99.9%", "kills germs",
        "do not ingest", "harmful if swallowed",
        "wear gloves", "ventilate the area",
    ],
    "pet_food": [
        # Pet food HAS nutrition panels but needs separate persona advice
        # We allow it through — handled by persona/category
    ],
    "industrial": [
        "material safety data sheet", "msds", "sds",
        "flash point", "boiling point", "vapour pressure",
        "hazard statement", "precautionary statement",
        "ghs", "un number",
    ],
}

# ── Layer 2: Food MUST-HAVE signals ──────────────────────────────────────
# A genuine food nutrition label will have AT LEAST ONE of these.
FOOD_REQUIRED_SIGNALS = [
    r"\bkcal\b", r"\bkj\b", r"\bkilojoule", r"\bkilocalorie",
    r"per\s*100\s*g", r"per\s*100\s*ml",
    r"\bcalorie", r"\benergy\b",
    r"\bprotein\b", r"\bcarbohydrate", r"\bcarbs\b",
    r"\btotal\s*fat\b", r"\bsaturated",
    r"\bfibre\b", r"\bfiber\b",
    r"\bsodium\b",
    r"nutrition\s*facts", r"nutritional\s*information",
    r"nutrition\s*information",
    r"serving\s*size",
    r"daily\s*value",
    r"reference\s*intake",
]

# ── Layer 3: Category keyword scoring ────────────────────────────────────
# Weights per category. Score ≥ 3 → classify as that category.
CATEGORY_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "cosmetics": [
        ("sodium lauryl",     3.0), ("laureth sulfate",   3.0),
        ("dimethicone",       2.5), ("silicone",           2.0),
        ("paraben",           2.5), ("phenoxyethanol",    2.5),
        ("fragrance",         1.5), ("parfum",            1.5),
        ("inci",              2.0), ("external use",      2.0),
        ("skin",              0.5), ("hair",              0.5),
        ("rinse",             0.7), ("lather",            1.5),
    ],
    "pharmaceutical": [
        ("active ingredient", 3.0), ("drug facts",        3.0),
        ("dosage",            2.0), ("tablet",            1.5),
        ("capsule",           1.5), ("syrup",             1.5),
        ("paracetamol",       3.0), ("ibuprofen",         3.0),
        ("prescription",      2.5), ("rx only",           3.0),
    ],
    "cleaning_product": [
        ("surfactant",        2.5), ("hypochlorite",      2.5),
        ("disinfectant",      2.5), ("bleach",            2.5),
        ("do not ingest",     3.0), ("harmful if swallowed", 3.0),
        ("laundry",           1.5), ("detergent",         1.5),
    ],
    "industrial": [
        ("msds",              3.0), ("sds",               3.0),
        ("flash point",       3.0), ("hazard statement",  3.0),
        ("ghs",               2.5), ("un number",         2.5),
    ],
}

# ── Detected type → user-friendly message ────────────────────────────────
TYPE_MESSAGES: dict[str, str] = {
    "cosmetics":       "cosmetics / personal care",
    "pharmaceutical":  "medicine / pharmaceutical",
    "cleaning_product": "cleaning / household product",
    "industrial":      "industrial / chemical product",
    "non_food":        "non-food product",
}


def classify_label_type(ocr_text: str) -> dict:
    """
    Classify whether the OCR text came from a food label or something else.

    Parameters
    ----------
    ocr_text : str
        Raw text extracted from the label image (can be messy OCR output).

    Returns
    -------
    dict with keys:
        is_non_food   : bool
        detected_type : str
        confidence    : float
        signals_found : list[str]
    """
    if not ocr_text or len(ocr_text.strip()) < 15:
        # Practically no text — don't block, let the main pipeline handle it
        return {"is_non_food": False, "detected_type": "unknown",
                "confidence": 0.0, "signals_found": []}

    text_lower = ocr_text.lower()
    signals_found: list[str] = []
    detected_type = "food"
    best_score = 0.0

    # ── Layer 1: Hard non-food signal check ──────────────────────────────
    for category, keywords in NON_FOOD_HARD_SIGNALS.items():
        if not keywords:
            continue
        for kw in keywords:
            if kw in text_lower:
                signals_found.append(kw)
                # A single hard signal is enough to flag — but we keep scoring
                # to find the BEST category name for the error message.
                if category != detected_type:
                    detected_type = category

    # If hard signals found, immediately check Layer 2 to confirm
    if signals_found:
        has_food_signal = any(
            re.search(pat, text_lower) for pat in FOOD_REQUIRED_SIGNALS
        )
        if not has_food_signal:
            # Hard signal + no food signal = strong non-food
            confidence = min(0.95, 0.6 + 0.05 * len(signals_found))
            logger.info(
                "Label classified as NON-FOOD (%s) | signals: %s",
                detected_type, signals_found[:3]
            )
            return {
                "is_non_food": True,
                "detected_type": TYPE_MESSAGES.get(detected_type, detected_type),
                "confidence": round(confidence, 2),
                "signals_found": signals_found[:5],
            }
        else:
            # Hard non-food signal BUT also has food signals — ambiguous
            # (e.g. a "nutritional supplement" or protein powder with ingredient list)
            # Let it through with a lower confidence flag
            logger.info(
                "Ambiguous label — has both food and non-food signals. Passing through."
            )
            return {
                "is_non_food": False,
                "detected_type": "ambiguous",
                "confidence": 0.4,
                "signals_found": signals_found[:3],
            }

    # ── Layer 3: Category keyword scoring ────────────────────────────────
    for category, kw_list in CATEGORY_KEYWORDS.items():
        cat_score = 0.0
        cat_signals: list[str] = []
        for kw, weight in kw_list:
            if kw in text_lower:
                cat_score += weight
                cat_signals.append(kw)
        if cat_score > best_score:
            best_score = cat_score
            if cat_score >= 3.0:
                detected_type = category
                signals_found = cat_signals

    if best_score >= 3.0:
        # Weighted vote says this is non-food
        has_food_signal = any(
            re.search(pat, text_lower) for pat in FOOD_REQUIRED_SIGNALS
        )
        if not has_food_signal:
            confidence = min(0.90, 0.45 + best_score * 0.05)
            logger.info(
                "Label classified as NON-FOOD (scored) | type=%s score=%.1f",
                detected_type, best_score
            )
            return {
                "is_non_food": True,
                "detected_type": TYPE_MESSAGES.get(detected_type, detected_type),
                "confidence": round(confidence, 2),
                "signals_found": signals_found[:5],
            }

    # ── Default: passes as food label ─────────────────────────────────────
    return {
        "is_non_food": False,
        "detected_type": "food",
        "confidence": 1.0 - (best_score * 0.05),  # small penalty for borderline cases
        "signals_found": [],
    }


# ── Quick unit test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    soap = (
        "Ingredients: Aqua, Sodium Lauryl Sulfate, Cocamidopropyl Betaine, "
        "Glycerin, Fragrance, Methylparaben. For external use only."
    )
    food = (
        "Nutrition Information Per 100g\n"
        "Energy: 389 kcal\nProtein: 8.2g\nCarbohydrate: 59.6g\nFat: 13.5g"
    )
    pharma = "Drug Facts Active Ingredient: Ibuprofen 200mg Dosage: 1-2 tablets"

    for label, name in [(soap, "Soap"), (food, "Maggi"), (pharma, "Pharma")]:
        r = classify_label_type(label)
        print(f"{name}: is_non_food={r['is_non_food']} type={r['detected_type']} "
              f"conf={r['confidence']} signals={r['signals_found']}")
