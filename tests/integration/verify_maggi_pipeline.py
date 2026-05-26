# -*- coding: utf-8 -*-
"""Send the Maggi label data through the full analysis pipeline."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import requests, json, time

BASE = "http://127.0.0.1:8000"

# Full OCR text from the Maggi Masala Noodles label
maggi_label = """
NUTRITIONAL INFORMATION
If a serve is 75g
NUTRITION Per 100g Per serve Per serve
Energy (kcal) 384 288 14%
Protein (g) 8.2 6.2 14%
Carbohydrate (g) 59.6 44.7 17%
-Total Sugars (g) 1.8 1.4 2%
-Added Sugars (g) 0.1 0.1 0%
Total Fat (g) 12.5 9.4 13%
-Saturated Fat (g) (not more than) 8.2 6.2 31%
-Trans Fat (g) (not more than) 0 0 0%
Sodium (mg) 1000 750 38%
Iron (mg) 6.90 5.17

INGREDIENTS
Noodles: Refined wheat flour (Maida), Palm oil, Iodised salt, Wheat gluten, 
Thickeners (508 & 412), Acidity regulators (501(i) & 500(ii)) and Humectant (451(ii)).

Masala TASTEMAKER: Mixed Spices (26.2%) (Onion powder, Coriander powder, Red chilli powder, 
Turmeric powder, Garlic powder, Cumin powder, Aniseed powder, Ginger powder, Fenugreek powder, 
Black pepper powder, Toasted onion powder, Clove powder, Green cardamom powder, Nutmeg powder), 
Hydrolysed groundnut protein, Refined wheat flour (Maida), Sugar, Starch, Palm oil, Iodized salt, 
Thickener (508), Acidity regulator (330), Flavour enhancer (635), Colour (150d), 
Mineral and Wheat gluten.

Allergen Note: Contains Wheat and Nut. May contain Milk, Oats and Soy.

Product: Maggi Masala Noodles 75g
Brand: Nestle India Limited
Pack contains 1 serve. 1 serve = 75g
"""

print("=" * 60)
print("MAGGI MASALA NOODLES - Full Pipeline Scan")
print("=" * 60)

# Send as voice-meal (text-based scan)
t0 = time.time()
resp = requests.post(f"{BASE}/parse-voice-meal", data={
    "text": maggi_label,
    "persona": "General Adult",
    "language": "en",
}, timeout=360)
elapsed = time.time() - t0

print(f"Status: {resp.status_code} ({elapsed:.1f}s)")
if resp.status_code == 200:
    d = resp.json()
    print(f"\nProduct:   {d.get('product_name', d.get('meal_name', '?'))}")
    print(f"Score:     {d.get('score', '?')}/10")
    print(f"Safety:    {d.get('safety_tier', '?')} - {d.get('safety_verdict', '?')}")
    print(f"Verdict:   {d.get('verdict', '?')}")
    print(f"Summary:   {d.get('summary', '?')}")
    print(f"\nNutrients:")
    for n in d.get("nutrient_breakdown", d.get("nutrients", []))[:10]:
        print(f"  {n.get('name','?'):20s} {n.get('value','?'):>8} {n.get('unit',''):3s}  [{n.get('rating','')}] {n.get('impact','')}")
    print(f"\nIngredients Spotlight:")
    for ing in d.get("ingredients_spotlight", [])[:8]:
        print(f"  * {ing.get('name','?')} ({ing.get('type','?')}) [{ing.get('safety_rating','')}]")
        print(f"    {ing.get('what_it_is','')}")
    print(f"\nELI5:       {d.get('eli5_explanation', '')}")
    print(f"Molecular:  {d.get('molecular_insight', '')}")
    print(f"Pros:       {d.get('pros', [])}")
    print(f"Cons:       {d.get('cons', [])}")
    
    # Save full result for inspection
    with open("data/maggi_scan_result.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Full JSON -> data/maggi_scan_result.json")
else:
    print(f"Error: {resp.text[:500]}")
