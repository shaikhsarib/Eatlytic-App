import os
import sys

# Ensure parent directory is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import db_conn, init_db

def seed_foods():
    # Make sure tables are initialized
    init_db()
    
    foods = [
        {
            "name": "Pasteurised Butter",
            "brand": "Amul",
            "category": "Dairy",
            "barcode": "8901262010018",
            "calories_100g": 722.0,
            "protein_100g": 0.6,
            "carbs_100g": 0.0,
            "fat_100g": 80.0,
            "sodium_100g": 830.0,
            "fiber_100g": 0.0,
            "sugar_100g": 0.0,
            "sat_fat_100g": 51.0,
            "eatlytic_score": 4,
            "ingredients_raw": "Butter, Common Salt",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Maggi Masala Noodles",
            "brand": "Nestle",
            "category": "Instant Noodles",
            "barcode": "8901058895724",
            "calories_100g": 389.0,
            "protein_100g": 8.0,
            "carbs_100g": 59.6,
            "fat_100g": 13.5,
            "sodium_100g": 1240.0,
            "fiber_100g": 2.0,
            "sugar_100g": 1.2,
            "sat_fat_100g": 7.0,
            "eatlytic_score": 2,
            "ingredients_raw": "Refined wheat flour (Maida), Palm oil, Salt, Wheat gluten, Mineral (Calcium carbonate), Thickeners (508 & 412), Acidity regulators (501(i) & 500(i)) and Humectant (451(i)). Masala Tastemaker: Hydrolysed groundnut protein, Mixed spices, Noodle powder, Sugar, Starch, Flavor enhancers (635 & 621), Caramel color (150d)",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Good Day Cashew Cookies",
            "brand": "Britannia",
            "category": "Bakery",
            "barcode": "8901063142142",
            "calories_100g": 500.0,
            "protein_100g": 7.0,
            "carbs_100g": 67.0,
            "fat_100g": 23.0,
            "sodium_100g": 320.0,
            "fiber_100g": 1.5,
            "sugar_100g": 22.0,
            "sat_fat_100g": 11.0,
            "eatlytic_score": 3,
            "ingredients_raw": "Refined Wheat Flour (Maida), Sugar, Edible Vegetable Oil (Palm), Cashew Nuts, Invert Sugar Syrup, Raising Agents, Butter, Milk Solids, Iodised Salt, Emulsifiers",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Parle-G Gluco Biscuits",
            "brand": "Parle",
            "category": "Bakery",
            "barcode": "8901109101119",
            "calories_100g": 454.0,
            "protein_100g": 6.5,
            "carbs_100g": 78.2,
            "fat_100g": 12.5,
            "sodium_100g": 360.0,
            "fiber_100g": 1.0,
            "sugar_100g": 26.3,
            "sat_fat_100g": 6.0,
            "eatlytic_score": 3,
            "ingredients_raw": "Wheat Flour, Sugar, Partially Hydrogenated Vegetable Oil, Invert Sugar Syrup, Raising Agents, Salt, Milk Solids, Emulsifier, Dough Conditioner",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Bhujia Sev",
            "brand": "Haldiram's",
            "category": "Snacks",
            "barcode": "8904063200055",
            "calories_100g": 579.0,
            "protein_100g": 10.5,
            "carbs_100g": 42.4,
            "fat_100g": 41.8,
            "sodium_100g": 850.0,
            "fiber_100g": 3.8,
            "sugar_100g": 0.5,
            "sat_fat_100g": 14.5,
            "eatlytic_score": 2,
            "ingredients_raw": "Dew Bean Flour (Moth Flour), Gram Flour, Edible Vegetable Oil (Cotton Seed, Corn & Palmolein Oil), Salt, Mixed Spices (Black Pepper, Ginger, Clove, Cardamom, Nutmeg)",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Treat Jim Jam Biscuits",
            "brand": "Britannia",
            "category": "Biscuit",
            "barcode": "8901063029170",
            "calories_100g": 483.0,
            "protein_100g": 5.0,
            "carbs_100g": 73.0,
            "fat_100g": 19.0,
            "sodium_100g": 280.0,
            "fiber_100g": 1.0,
            "sugar_100g": 37.5,
            "sat_fat_100g": 9.5,
            "eatlytic_score": 3,
            "ingredients_raw": "Refined wheat flour (Maida), Sugar, Mixed fruit jam, Edible vegetable oil (Palm), Invert sugar syrup, Butter, Milk solids, Iodised salt, Raising agents [503(ii) & 500(ii)], Emulsifiers, Preservative (223), Acidity regulators, Synthetic food color (INS 122)",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Cheese Slices",
            "brand": "Amul",
            "category": "Dairy",
            "barcode": "8901262150318",
            "calories_100g": 310.0,
            "protein_100g": 20.0,
            "carbs_100g": 1.5,
            "fat_100g": 25.0,
            "sodium_100g": 1400.0,
            "fiber_100g": 0.0,
            "sugar_100g": 1.5,
            "sat_fat_100g": 16.0,
            "eatlytic_score": 5,
            "ingredients_raw": "Cheese, Milk Solids, Emulsifiers (INS 331, 339), Common Salt, Preservative (INS 200)",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Hide & Seek Chocolate Chip Cookies",
            "brand": "Parle",
            "category": "Bakery",
            "barcode": "8901109121605",
            "calories_100g": 480.0,
            "protein_100g": 5.5,
            "carbs_100g": 72.0,
            "fat_100g": 19.0,
            "sodium_100g": 340.0,
            "fiber_100g": 1.5,
            "sugar_100g": 32.0,
            "sat_fat_100g": 9.0,
            "eatlytic_score": 3,
            "ingredients_raw": "Refined wheat flour (Maida), Chocolate chips [Sugar, Cocoa mass, Cocoa butter, Dextrose, Emulsifier (INS 322)], Sugar, Edible vegetable oil (Palm), Invert sugar syrup, Raising agents, Iodised salt, Emulsifier",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Aloo Bhujia",
            "brand": "Haldiram's",
            "category": "Snacks",
            "barcode": "8904063200130",
            "calories_100g": 580.0,
            "protein_100g": 9.0,
            "carbs_100g": 45.0,
            "fat_100g": 40.0,
            "sodium_100g": 850.0,
            "fiber_100g": 3.0,
            "sugar_100g": 0.5,
            "sat_fat_100g": 14.0,
            "eatlytic_score": 2,
            "ingredients_raw": "Potatoes (44%), Tepary beans flour (Moth flour), Gram flour, Edible vegetable oil (Cotton seed, Corn & Palmolein), Spices & Condiments, Iodised salt, Citric acid",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Iodised Salt",
            "brand": "Tata",
            "category": "Condiment",
            "barcode": "8901058002313",
            "calories_100g": 0.0,
            "protein_100g": 0.0,
            "carbs_100g": 0.0,
            "fat_100g": 0.0,
            "sodium_100g": 38700.0,
            "fiber_100g": 0.0,
            "sugar_100g": 0.0,
            "sat_fat_100g": 0.0,
            "eatlytic_score": 6,
            "ingredients_raw": "Edible common salt, Potassium iodate, Anticaking agent (INS 536)",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "Dairy Milk Chocolate",
            "brand": "Cadbury",
            "category": "Confectionery",
            "barcode": "7622201768406",
            "calories_100g": 530.0,
            "protein_100g": 7.5,
            "carbs_100g": 60.0,
            "fat_100g": 29.0,
            "sodium_100g": 150.0,
            "fiber_100g": 2.0,
            "sugar_100g": 57.0,
            "sat_fat_100g": 17.5,
            "eatlytic_score": 3,
            "ingredients_raw": "Sugar, Milk solids (22%*), Cocoa butter, Cocoa solids, Emulsifiers (INS 442, 476), Flavours",
            "verified": 1,
            "source": "verified_seeder"
        },
        {
            "name": "India's Magic Masala Chips",
            "brand": "Lays",
            "category": "Snacks",
            "barcode": "8901491101837",
            "calories_100g": 555.0,
            "protein_100g": 7.0,
            "carbs_100g": 52.0,
            "fat_100g": 35.0,
            "sodium_100g": 780.0,
            "fiber_100g": 2.5,
            "sugar_100g": 2.0,
            "sat_fat_100g": 13.5,
            "eatlytic_score": 2,
            "ingredients_raw": "Potato, Edible vegetable oil (Palmolein, Rice bran), Spices & Condiments, Iodised salt, Sugar, Acidity regulators (INS 330, 296), Flavor enhancers (INS 627, 631)",
            "verified": 1,
            "source": "verified_seeder"
        }
    ]
    
    with db_conn() as conn:
        for f in foods:
            # Check if exists by barcode or brand/name
            existing = conn.execute(
                "SELECT id FROM food_products WHERE barcode = ? OR (LOWER(name) = LOWER(?) AND LOWER(brand) = LOWER(?))",
                (f["barcode"], f["name"], f["brand"])
            ).fetchone()
            
            if existing:
                conn.execute("""
                    UPDATE food_products 
                    SET category=?, calories_100g=?, protein_100g=?, carbs_100g=?, 
                        fat_100g=?, sodium_100g=?, fiber_100g=?, sugar_100g=?, 
                        eatlytic_score=?, verified=?, verified_by='admin', verified_at=datetime('now'), updated_at=datetime('now')
                    WHERE id=?
                """, (
                    f["category"], f["calories_100g"], f["protein_100g"], f["carbs_100g"],
                    f["fat_100g"], f["sodium_100g"], f["fiber_100g"], f["sugar_100g"],
                    f["eatlytic_score"], f["verified"], existing["id"]
                ))
                prod_id = existing["id"]
                print(f"Updated: {f['brand']} {f['name']}")
            else:
                cursor = conn.execute("""
                    INSERT INTO food_products (
                        name, brand, category, barcode, calories_100g,
                        protein_100g, carbs_100g, fat_100g, sodium_100g,
                        fiber_100g, sugar_100g, eatlytic_score, verified,
                        verified_by, verified_at, scan_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'admin', datetime('now'), 0)
                """, (
                    f["name"], f["brand"], f["category"], f["barcode"], f["calories_100g"],
                    f["protein_100g"], f["carbs_100g"], f["fat_100g"], f["sodium_100g"],
                    f["fiber_100g"], f["sugar_100g"], f["eatlytic_score"], f["verified"]
                ))
                prod_id = cursor.lastrowid
                print(f"Seeded: {f['brand']} {f['name']}")
                
            # Upsert extra metadata in secondary table
            conn.execute("""
                INSERT OR REPLACE INTO food_products_extra (id, sat_fat_100g, ingredients_raw, source)
                VALUES (?, ?, ?, ?)
            """, (prod_id, f["sat_fat_100g"], f["ingredients_raw"], f["source"]))
    # Seed Nestle Maggi Masala Noodles pHash image fingerprint
    maggi_phash = "ae3a51e295e55259"
    maggi_fingerprint_json = {
        "product_name": "Nestle Maggi Masala Noodles",
        "product_category": "instant_noodles",
        "serving_size": "100g",
        "score": 4,
        "score_color": "#f59e0b",
        "safety_tier": "Limit",
        "safety_verdict": "Moderation Advised",
        "safety_reason": "Seeded Maggi Masala Noodles fingerprint.",
        "verdict": "Verified offline match.",
        "summary": "Maggi Noodles is a processed food product.",
        "nutrient_breakdown": [
            {"name": "Energy", "value": 384.0, "unit": "kcal", "rating": "moderate", "impact": "neutral"},
            {"name": "Protein", "value": 8.0, "unit": "g", "rating": "good", "impact": "positive"},
            {"name": "Carbohydrates", "value": 59.6, "unit": "g", "rating": "moderate", "impact": "neutral"},
            {"name": "Fat", "value": 13.5, "unit": "g", "rating": "moderate", "impact": "neutral"},
            {"name": "Sodium", "value": 1000.0, "unit": "mg", "rating": "high", "impact": "negative"},
            {"name": "Fiber", "value": 2.0, "unit": "g", "rating": "moderate", "impact": "neutral"},
            {"name": "Sugar", "value": 1.2, "unit": "g", "rating": "low", "impact": "positive"},
            {"name": "Saturated Fat", "value": 7.0, "unit": "g", "rating": "moderate", "impact": "neutral"},
            {"name": "Trans Fat", "value": 0.0, "unit": "g", "rating": "low", "impact": "positive"},
            {"name": "Cholesterol", "value": 0.0, "unit": "mg", "rating": "low", "impact": "positive"}
        ],
        "pros": ["Good amount of protein"],
        "cons": ["High in sodium"],
        "age_warnings": [],
        "eli5_explanation": "Nestle Maggi Masala Noodles is a well-known snack.",
        "molecular_insight": "",
        "chart_data": [60, 10, 30],
        "ingredients_raw": "Refined wheat flour (Maida), Palm oil, Salt, Wheat gluten, Mineral (Calcium carbonate), Thickeners (508 & 412), Acidity regulators (501(i) & 500(i)) and Humectant (451(i)). Masala Tastemaker: Hydrolysed groundnut protein, Mixed spices, Noodle powder, Sugar, Starch, Flavor enhancers (635 & 621), Caramel color (150d)",
        "ingredients_spotlight": [
            {"name": "Wheat Flour", "type": "natural", "safety_rating": "safe", "what_it_is": "Grain flour.", "health_impact": "Energy source.", "curiosity_fact": "Common staple."}
        ],
        "extraction_confidence": {
            "tier": "HIGH",
            "score": 1.0,
            "message": "Seeded pHash fingerprint match.",
            "atwater_valid": True,
        },
        "better_alternative": "",
        "whatsapp_content": {},
        "disclaimer": "Consult a nutritionist for personalized medical advice."
    }
    
    with db_conn() as conn:
        import json
        conn.execute("""
            INSERT OR REPLACE INTO image_fingerprints (hash_key, result_json)
            VALUES (?, ?)
        """, (maggi_phash, json.dumps(maggi_fingerprint_json)))
        print(f"Seeded image fingerprint for pHash {maggi_phash}")
            
    print("Database seeding completed successfully.")

if __name__ == "__main__":
    seed_foods()
