"""
app/services/nutrition_parser.py
Hybrid extraction: Regex (Fast Path) vs LLM (Deep Path).
"""

import re
import logging

logger = logging.getLogger(__name__)

# Common regex patterns for Indian nutrition labels
NUTRIENT_PATTERNS = {
    "calories": [
        r"(?:energy|calories|kcal)\D*(\d+(?:\.\d+)?)",
    ],
    "protein": [
        r"(?:protein|protine)\D*(\d+(?:\.\d+)?)",
    ],
    "carbs": [
        r"(?:carbohydrate|total\s*carb|carbs)\D*(\d+(?:\.\d+)?)",
    ],
    "fat": [
        r"(?:total\s*fat|fat)\D*(\d+(?:\.\d+)?)",
    ],
    "sugar": [
        r"(?:total\s*sugar|sugars)\D*(\d+(?:\.\d+)?)",
    ],
    "sodium_mg": [
        r"(?:sodium|na)\D*(\d+(?:\.\d+)?)\s*(?:mg|g)",
    ],
    "fiber": [
        r"(?:fiber|fibre|dietary\s*fiber)\D*(\d+(?:\.\d+)?)",
    ],
}

def classify_label(text: str) -> str:
    """
    Classifies label as SIMPLE or COMPLEX based on numeric density.
    """
    lines = text.split('\n')
    numeric_lines = [l for l in lines if re.search(r'\d', l)]
    
    if len(numeric_lines) <= 5:
        return "SIMPLE"
    return "COMPLEX"

def regex_parse(text: str) -> dict:
    """
    Extracts nutrients using regex patterns.
    """
    results = {}
    text_lower = text.lower()
    
    for nutrient, patterns in NUTRIENT_PATTERNS.items():
        results[nutrient] = 0.0
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    results[nutrient] = float(match.group(1))
                    break
                except ValueError:
                    continue
                    
    return results

async def smart_parse(text: str, llm_callback) -> dict:
    """
    Decides between Regex or LLM extraction.
    """
    label_type = classify_label(text)
    logger.info(f"Label classified as: {label_type}")
    
    if label_type == "SIMPLE":
        parsed = regex_parse(text)
        # Verify if we got minimum critical data
        if parsed.get("calories", 0) > 0 and parsed.get("carbs", 0) > 0:
            logger.info("Fast Path (Regex) succeeded.")
            # Mock the other fields the LLM provides
            parsed["product_name"] = "Simple Product"
            parsed["product_category"] = "unknown"
            parsed["ingredients_raw"] = ""
            return parsed
            
    # Fallback to LLM for COMPLEX or if Regex missed critical data
    logger.info("Deep Path (LLM) triggered.")
    return await llm_callback(text)
