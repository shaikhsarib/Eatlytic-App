"""
Test script for Smart Label Detection & Explanation Engine.
Specifically tests the Math Mismatch fix and the Explanation Engine.
"""

import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.services.fake_detector import FakeDetector
from app.services.explanation_engine import get_explanation_report

def test_math_fix():
    print("Testing Math Mismatch fix...")
    detector = FakeDetector(tolerance_percent=35.0)
    
    # problematic Maggi-like label data
    label_data = {
        'calories': 389.0,
        'protein': 8.2,
        'carbohydrate': 59.6,
        'total_fat': 13.5,
        'total_sugars': 1.8,  # SUB
        'added_sugars': 1.1,   # SUB
        'fiber': 2.0,         # SUB
        'saturated_fat': 8.2,  # SUB
        'trans_fat': 0.13,     # SUB
    }
    
    result = detector.validate(label_data)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Label: {result['label_calories']}, Calculated: {result['calculated_calories']}")
    
    assert result['status'] == 'VALID', f"Expected VALID, got {result['status']}"
    print("✅ Math Mismatch fix verified.\n")

def test_explanation_engine():
    print("Testing Explanation Engine...")
    nutrients = {
        'calories': 389.0,
        'protein': 8.2,
        'carbs': 59.6,
        'fat': 13.5,
        'sugar': 1.8,
        'sodium_mg': 1028.3,
    }
    ingredients = "Wheat flour, palm oil, salt, sugar, maltodextrin, flavor enhancer (635), thickener (412), acidity regulator (501i)."
    
    report = get_explanation_report(nutrients, ingredients)
    print(f"NOVA Level: {report['nova_level']}")
    print(f"Verdict: {report['verdict']}")
    print("Insights:")
    for insight in report['humanized_insights']:
        print(f" - {insight}")
    
    assert report['nova_level'] >= 3, "Expected NOVA 3 or 4"
    assert "🔴 RED" in report['verdict'], "Expected RED verdict due to high sodium"
    print("✅ Explanation Engine verified.\n")

if __name__ == "__main__":
    try:
        test_math_fix()
        test_explanation_engine()
        print("All tests passed! 🚀")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
