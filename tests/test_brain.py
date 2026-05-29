import pytest
from app.services.brain import EatlyticBrain, INDIAN_INGREDIENT_LEXICON

def test_lexicon_matching():
    brain = EatlyticBrain()
    
    # 1. Match by name
    text = "Ingredients: Refined wheat flour (maida), palm oil, sugar, maltodextrin."
    matches = brain.match_lexicon_ingredients(text)
    matched_names = {m["name"] for m in matches}
    
    assert "Maltodextrin" in matched_names
    assert "Maida (Refined Wheat Flour)" in matched_names or "Refined Wheat Flour (Maida)" in matched_names
    assert "Refined Sugar (Sucrose)" in matched_names
    assert "Refined Palm Oil" in matched_names

    # 2. Match by INS code / E-number
    text_ins = "Carbonated water, acidity regulator (INS 330), artificial sweetener (ins 951)."
    matches_ins = brain.match_lexicon_ingredients(text_ins)
    matched_ins_names = {m["name"] for m in matches_ins}
    assert "Aspartame" in matched_ins_names

def test_diabetic_care_audit_avoid():
    brain = EatlyticBrain()
    
    # High sugar biscuit scan
    nutrients = {
        "sugar": 21.0,
        "carbs": 70.0,
        "protein": 5.0,
        "fat": 15.0,
        "calories": 435.0,
    }
    ingredients = "Refined wheat flour, Sugar, Palm oil, Maltodextrin."
    
    report = brain.compile_local_report(
        product_name="Cookies",
        brand="Britannia",
        category="biscuit",
        nutrients=nutrients,
        ingredients_raw=ingredients,
        persona="diabetic"
    )
    
    assert report["safety_tier"] == "Avoid"
    assert report["score"] == 1 # heavy deductions
    assert report["safety_verdict"] == "Glycemic Threat"
    assert any("Maltodextrin" in reason for reason in report["cons"])
    assert report["sugar"] == 21.0
    # 21g / 4.2 = 5 teaspoons
    assert report["summary"] is not None
    assert "5" in report["eli5_explanation"] # 5 teaspoons check

def test_diabetic_care_audit_caution():
    brain = EatlyticBrain()
    
    # Sugar-free drink with Aspartame
    nutrients_diet = {
        "sugar": 0.0,
        "carbs": 0.2,
        "protein": 0.0,
        "fat": 0.0,
        "calories": 1.0,
        "sodium": 15.0,
    }
    ingredients_diet = "Carbonated water, color (INS 150d), sweeteners (INS 951, INS 950)."
    
    report = brain.compile_local_report(
        product_name="Diet Cola",
        brand="Brand X",
        category="beverage",
        nutrients=nutrients_diet,
        ingredients_raw=ingredients_diet,
        persona="diabetic"
    )
    
    # Additive DB now correctly identifies E150d (Caramel IV) + Aspartame + Acesulfame K
    # as CAUTION additives, increasing deductions. Score ≤6 and Limit/Avoid are both valid.
    assert report["safety_tier"] in ("Limit", "Avoid")  # Capped due to artificial sweeteners
    assert report["score"] <= 6  # base 10 - deductions for sweetener + caramel additives
    assert any("FSSAI Statutory Warning" in c for c in report["cons"])

def test_hypertension_sodium_audit():
    brain = EatlyticBrain()
    
    # High sodium chips
    nutrients = {
        "sugar": 2.0,
        "carbs": 50.0,
        "protein": 7.0,
        "fat": 35.0,
        "calories": 543.0,
        "sodium": 890.0, # High sodium
    }
    
    report = brain.compile_local_report(
        product_name="Masala Chips",
        brand="Brand Y",
        category="snack",
        nutrients=nutrients,
        ingredients_raw="Potatoes, Palm Oil, Spices, Iodised Salt",
        persona="adult"
    )
    
    assert any("sodium" in c.lower() for c in report["cons"])
    assert any("hypertension" in c.lower() for c in report["cons"])

def test_atwater_physics_audit():
    brain = EatlyticBrain()
    
    # Impossible macros/calories
    nutrients_fraud = {
        "sugar": 0.0,
        "carbs": 10.0, # 40 kcal
        "protein": 10.0, # 40 kcal
        "fat": 10.0, # 90 kcal -> expected total = 170 kcal
        "calories": 300.0, # Stated = 300 kcal (severe mismatch!)
    }
    
    report = brain.compile_local_report(
        product_name="Fake Bar",
        brand="FraudCorp",
        category="other",
        nutrients=nutrients_fraud,
        ingredients_raw="Protein isolate, glycerin",
        persona="adult"
    )
    
    assert report["extraction_confidence"]["atwater_valid"] is False
    assert "Atwater mismatch" in report["eli5_explanation"]


def test_clinical_audit_exposure():
    brain = EatlyticBrain()
    nutrients = {
        "sugar": 16.8,
        "carbs": 55.0,
        "protein": 6.0,
        "fat": 12.0,
        "calories": 352.0,
    }
    ingredients = "Refined wheat flour, Sugar, Palm oil, Maltodextrin."
    report = brain.compile_local_report(
        product_name="Sweet Biscuits",
        brand="CookieCorp",
        category="biscuit",
        nutrients=nutrients,
        ingredients_raw=ingredients,
        persona="diabetic"
    )
    
    assert "clinical_audit" in report
    assert "sugar_teaspoons" in report
    assert "gi_level" in report
    assert report["gi_level"] == "HIGH"
    # 16.8 / 4.2 = 4.0 teaspoons
    assert abs(report["sugar_teaspoons"] - 4.0) < 0.1
    assert report["clinical_audit"]["verdict"] == "AVOID"

