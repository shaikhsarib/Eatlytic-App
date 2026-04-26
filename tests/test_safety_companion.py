import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Import the logic we want to test
from app.services.llm import unified_analyze_flow

@pytest.mark.asyncio
async def test_safety_companion_diabetes_avoid():
    """Verify that high-sugar labels trigger 🔴 Avoid for Diabetics."""
    
    # Simulated high-sugar OCR text
    label_text = "Nutrition Facts\nPer 100g\nSugar: 45g\nCarbohydrates: 80g\nEnergy: 400kcal"
    
    # Mock LLM response including the new Safety Companion fields
    mock_llm_json = {
        "product_name": "Sugary Cereal",
        "product_category": "Cereal",
        "score": 2,
        "safety_tier": "Avoid",
        "safety_verdict": "High Risk",
        "safety_reason": "Contains 45g of sugar (nearly 2x daily limit) which will cause dangerous blood glucose spikes.",
        "verdict": "Sugar Bomb",
        "summary": "This cereal is primarily sugar and refined carbs.",
        "calories": 400, "protein": 5, "carbs": 80, "fat": 2, "sugar": 45,
        "nutrients": [
            {"name": "Energy", "value": 400, "unit": "kcal", "rating": "caution"},
            {"name": "Protein", "value": 5, "unit": "g", "rating": "moderate"},
            {"name": "Total Fat", "value": 2, "unit": "g", "rating": "good"},
            {"name": "Total Carbohydrate", "value": 80, "unit": "g", "rating": "bad"},
            {"name": "Sugar", "value": 45, "unit": "g", "rating": "bad", "impact": "Extremely high sugar."}
        ],
        "ingredients_raw": "Sugar, Corn, Maltodextrin",
        "pros": [],
        "cons": ["Extreme sugar content"],
        "age_warnings": [],
        "better_alternative": "Plain Oats"
    }

    with patch("app.services.llm.call_llm", return_value=json.dumps(mock_llm_json)):
        result = await unified_analyze_flow(
            extracted_text=label_text,
            persona="Diabetes Care",
            age_group="adult",
            product_category_hint="Cereal",
            language="en",
            web_context="",
            blur_info={"detected": False},
            label_confidence="high"
        )
        
        assert "error" not in result
        assert result["safety_tier"] == "Avoid"
        assert "sugar" in result["safety_reason"].lower()
        assert "glucose" in result["safety_reason"].lower()
        assert result["score"] <= 4

@pytest.mark.asyncio
async def test_safety_companion_hypertension_avoid():
    """Verify that high-sodium labels trigger 🔴 Avoid for Hypertension patients."""
    
    label_text = "Nutrition Facts\nPer 100g\nSodium: 1200mg\nSalt: 3g"
    
    mock_llm_json = {
        "product_name": "Salted Chips",
        "product_category": "Snack",
        "score": 3,
        "safety_tier": "Avoid",
        "safety_verdict": "High Sodium",
        "safety_reason": "Sodium content (1200mg) exceeds 50% of the daily recommended limit for hypertension patients.",
        "verdict": "Very Salty",
        "summary": "Processed snack with excessive salt.",
        "calories": 500, "protein": 2, "carbs": 50, "fat": 30, "sodium_mg": 1200,
        "nutrients": [
            {"name": "Energy", "value": 500, "unit": "kcal", "rating": "bad"},
            {"name": "Protein", "value": 2, "unit": "g", "rating": "moderate"},
            {"name": "Total Fat", "value": 30, "unit": "g", "rating": "bad"},
            {"name": "Total Carbohydrate", "value": 50, "unit": "g", "rating": "moderate"},
            {"name": "Sodium", "value": 1200, "unit": "mg", "rating": "bad", "impact": "Extremely high salt."}
        ],
        "ingredients_raw": "Potatoes, Oil, Salt",
        "pros": [],
        "cons": ["High sodium"],
        "age_warnings": [],
        "better_alternative": "Unsalted Nuts"
    }

    with patch("app.services.llm.call_llm", return_value=json.dumps(mock_llm_json)):
        result = await unified_analyze_flow(
            extracted_text=label_text,
            persona="Blood Pressure (Hypertension)",
            age_group="adult",
            product_category_hint="Snack",
            language="en",
            web_context="",
            blur_info={"detected": False},
            label_confidence="high"
        )
        
        assert "error" not in result
        assert result["safety_tier"] == "Avoid"
        assert "sodium" in result["safety_reason"].lower()
        assert result["safety_verdict"] == "High Sodium"

@pytest.mark.asyncio
async def test_safety_companion_general_health_safe():
    """Verify that a healthy product triggers 🟢 Safe for general users."""
    
    label_text = "Nutrition Facts\nPer 100g\nSugar: 0g\nProtein: 15g\nFiber: 8g"
    
    mock_llm_json = {
        "product_name": "Lentils",
        "product_category": "Pulse",
        "score": 10,
        "safety_tier": "Safe",
        "safety_verdict": "Excellent Choice",
        "safety_reason": "High in fiber and protein with zero added sugars. Excellent for blood sugar stability and satiation.",
        "verdict": "Superfood",
        "summary": "Clean, whole-food source of nutrition.",
        "calories": 300, "protein": 15, "carbs": 50, "fat": 1, "fiber": 8, "sugar": 0,
        "nutrients": [
             {"name": "Energy", "value": 300, "unit": "kcal", "rating": "good"},
            {"name": "Protein", "value": 15, "unit": "g", "rating": "good"},
            {"name": "Total Fat", "value": 1, "unit": "g", "rating": "good"},
            {"name": "Total Carbohydrate", "value": 50, "unit": "g", "rating": "moderate"},
            {"name": "Dietary Fiber", "value": 8, "unit": "g", "rating": "good"}
        ],
        "ingredients_raw": "Dried Lentils",
        "pros": ["High Fiber", "High Protein"],
        "cons": [],
        "age_warnings": [],
        "better_alternative": "None"
    }

    with patch("app.services.llm.call_llm", return_value=json.dumps(mock_llm_json)):
        result = await unified_analyze_flow(
            extracted_text=label_text,
            persona="General Health",
            age_group="adult",
            product_category_hint="Pulse",
            language="en",
            web_context="",
            blur_info={"detected": False},
            label_confidence="high"
        )
        
        assert "error" not in result
        assert result["safety_tier"] == "Safe"
        assert result["score"] >= 8
