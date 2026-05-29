"""
app/services/personalized_medicine.py
Personalized Genomic & Biomarker Intelligence Engine.
Deterministic parser mapping DNA SNPs and biomarker ranges to safety coefficient overrides.
"""

import logging
from typing import Dict, List
from app.database.connection import get_genomic_profile

logger = logging.getLogger(__name__)

def apply_genomic_overrides(device_key: str, nutrients: Dict, matched_ingredients: List[Dict], audit: Dict) -> Dict:
    """
    Applies genomic SNP and biomarker overrides to the baseline clinical audit statelessly.
    Returns the updated audit dictionary with modified scores, verdicts, and clinical reasons.
    """
    if not device_key:
        return audit

    profile = get_genomic_profile(device_key)
    if not profile:
        return audit

    snps = profile.get("genetic_snps", {})
    biomarkers = profile.get("biomarkers", {})

    verdict = audit["verdict"]
    safety_tier = audit["safety_tier"]
    score = audit["score"]
    reasons = list(audit["reasons"])
    gi_level = audit["gi_level"]

    # ── 1. GENOMIC OVERRIDE: TCF7L2 (rs7903146) - Diabetes Predisposition ──
    tcf7l2_genotype = snps.get("rs7903146", "").upper().strip()
    if tcf7l2_genotype in ["CT", "TT"]:
        high_gi_found = False
        high_gi_names = []
        for ing in matched_ingredients:
            if ing.get("gi_level") == "HIGH" or ing.get("typical_gi", 0) > 70:
                high_gi_found = True
                high_gi_names.append(ing["name"])
        
        # If sugar or carbs are high, double their score deduction
        sugar = float(nutrients.get("sugar") or nutrients.get("sugar_100g") or 0.0)
        carbs = float(nutrients.get("carbs") or nutrients.get("carbs_100g") or 0.0)
        
        tcf_deduction = 0
        if sugar > 5.0 or carbs > 25.0 or high_gi_found:
            tcf_deduction += 3
            if high_gi_found:
                verdict = "AVOID"
                safety_tier = "Avoid"
                gi_level = "HIGH"
                score = min(score, 3)
                reasons.append(
                    f"🧬 Genomic Risk (TCF7L2 - rs7903146 Variant '{tcf7l2_genotype}'): "
                    f"High Type-2 Diabetes predisposition detected. Highly aggressive insulin surges "
                    f"expected from high-GI ingredients present: {', '.join(high_gi_names)}."
                )
            else:
                reasons.append(
                    f"🧬 Genomic Warning (TCF7L2 - rs7903146 Variant '{tcf7l2_genotype}'): "
                    f"Moderate T2D risk. High glycemic sensitivity applied to carbohydrate load."
                )
            score = max(1, score - tcf_deduction)

    # ── 2. GENOMIC OVERRIDE: AGT (rs5068) - Sodium-Sensitive Hypertension ──
    agt_genotype = snps.get("rs5068", "").upper().strip()
    if agt_genotype in ["GG", "AG"]:
        sodium = float(nutrients.get("sodium") or nutrients.get("sodium_mg") or nutrients.get("sodium_100g") or 0.0)
        # Dynamic sodium threshold drop
        if sodium > 300.0:
            verdict = "AVOID"
            safety_tier = "Avoid"
            score = min(score, 3)
            reasons.append(
                f"🧬 Genomic Risk (AGT - rs5068 Variant '{agt_genotype}'): "
                f"High sodium-sensitive hypertension risk. Serving sodium ({sodium}mg) "
                f"wrecks your hyper-restricted 1000mg/day genetic ceiling."
            )
        elif sodium > 100.0:
            verdict = "CAUTION" if verdict != "AVOID" else verdict
            safety_tier = "Limit" if safety_tier == "Safe" else safety_tier
            score = min(score, 6)
            reasons.append(
                f"🧬 Genomic Warning (AGT - rs5068 Variant '{agt_genotype}'): "
                f"Restricted sodium sensitivity active. Watch salt portions closely."
            )

    # ── 3. GENOMIC OVERRIDE: LCT (rs4988235) - Lactase Non-Persistence ──
    lct_genotype = snps.get("rs4988235", "").upper().strip()
    if lct_genotype == "GG":
        # Lactose intolerant. Scan matched ingredients or raw ingredients text
        ingredients_raw_text = str(nutrients.get("ingredients_raw_text") or "").lower()
        lactose_indicators = ["milk", "whey", "lactose", "milk solids", "butter", "cheese", "cream", "curd", "milk powder", "yogurt"]
        lactose_found = any(ind in ingredients_raw_text for ind in lactose_indicators)
        
        # Also check names of matched ingredients
        for ing in matched_ingredients:
            if any(ind in ing["name"].lower() for ind in lactose_indicators):
                lactose_found = True
                break
                
        if lactose_found:
            verdict = "AVOID"
            safety_tier = "Avoid"
            score = 1
            reasons.append(
                f"🧬 Genomic Risk (LCT - rs4988235 Variant 'GG'): "
                f"Primary Lactase Non-Persistence (lactose intolerance) detected. "
                f"Contains lactose/dairy derivatives threatening severe gastric discomfort."
            )

    # ── 4. BIOMARKER OVERRIDES: HbA1c & Fasting Glucose ──
    hba1c = biomarkers.get("hba1c")
    fasting_glucose = biomarkers.get("fasting_glucose")

    if hba1c is not None:
        try:
            hba1c_val = float(hba1c)
            if hba1c_val >= 5.7:
                # Downgrade artificial/synthetic sweeteners
                sweetener_found = False
                sweetener_names = []
                for ing in matched_ingredients:
                    if ing.get("category") == "sweetener" and ing.get("diabetic_verdict") in ["SAFE", "CAUTION"]:
                        # Exclude pure stevia, monkfruit if desired, but downgrade synthetic ones
                        if any(x in ing["name"].lower() for x in ["aspartame", "acesulfame", "sucralose", "maltitol", "sorbitol"]):
                            sweetener_found = True
                            sweetener_names.append(ing["name"])
                
                if sweetener_found:
                    verdict = "AVOID"
                    safety_tier = "Avoid"
                    score = min(score, 3)
                    reasons.append(
                        f"🩸 Biomarker Risk (HbA1c: {hba1c_val}%): "
                        f"Prediabetic/diabetic glycemic profile active. "
                        f"Downgrading synthetic sweeteners ({', '.join(sweetener_names)}) "
                        f"due to cephalic-phase insulin release and gut microbiota disruption risks."
                    )
        except ValueError:
            pass

    if fasting_glucose is not None:
        try:
            fg_val = float(fasting_glucose)
            if fg_val >= 100.0:
                sugar = float(nutrients.get("sugar") or nutrients.get("sugar_100g") or 0.0)
                if sugar > 2.0:
                    verdict = "AVOID"
                    safety_tier = "Avoid"
                    score = min(score, 3)
                    reasons.append(
                        f"🩸 Biomarker Warning (Fasting Glucose: {fg_val} mg/dL): "
                        f"Impaired fasting glucose detected. Restricted sugar tolerance ceiling active."
                    )
        except ValueError:
            pass

    # Normalize score colors and safety tiers
    if score >= 7:
        color = "#22c55e"
        safety_tier = "Safe"
        safety_verdict = "Nutritious Choice"
    elif score >= 4:
        color = "#f59e0b"
        safety_tier = "Limit"
        safety_verdict = "Moderation Advised"
    else:
        color = "#ef4444"
        safety_tier = "Avoid"
        safety_verdict = "Glycemic Threat"

    return {
        "verdict": verdict,
        "safety_tier": safety_tier,
        "score": score,
        "score_color": color,
        "safety_verdict": safety_verdict,
        "reasons": reasons,
        "gi_level": gi_level,
        "teaspoons": audit["teaspoons"]
    }
