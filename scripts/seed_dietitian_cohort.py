import os
import sys
import json
import hashlib
import datetime

# Ensure parent directory is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import db_conn, init_db

def seed_dietitian_cohort():
    init_db()
    
    # Standardize on a single dietitian and access key for the live demo
    dietitian_code = "COHORT1"
    dietitian_key = "diet_testkey123"
    dietitian_name = "Dr. Ananya Sharma"
    dietitian_email = "ananya@eatlytic.com"
    
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    three_days_ago = (datetime.date.today() - datetime.timedelta(days=3)).isoformat()
    
    patients = [
        {
            "key": "patient_key_diabetic_1",
            "persona": "diabetic",
            "streak": 14,
            "logs": [
                {"meal": "Whole Wheat Roti & Paneer Curry", "calories": 420.0, "protein": 22.0, "carbs": 35.0, "fat": 18.0},
                {"meal": "Boiled Eggs & Roasted Almonds", "calories": 250.0, "protein": 18.0, "carbs": 4.0, "fat": 16.0}
            ],
            "scans": [
                {
                    "product_name": "Treat Jim Jam Biscuits",
                    "brand": "Britannia",
                    "score": 3,
                    "verdict": "Glycemic Threat",
                    "calories": 483.0, "protein": 5.0, "carbs": 73.0, "fat": 19.0, "sodium": 280.0, "fiber": 1.0, "sugar": 37.5,
                    "category": "biscuit",
                    "scanned_at": yesterday,
                    "analysis": {
                        "sugar": 37.5, "sodium_mg": 280.0,
                        "cons": ["High in added sugar (37.5g) triggering severe postprandial glucose spike", "Refined flour (Maida) base with zero fiber buffer"]
                    }
                },
                {
                    "product_name": "Pasteurised Butter",
                    "brand": "Amul",
                    "score": 4,
                    "verdict": "Caution: High Saturated Fat",
                    "calories": 722.0, "protein": 0.6, "carbs": 0.0, "fat": 80.0, "sodium": 830.0, "fiber": 0.0, "sugar": 0.0,
                    "category": "Dairy",
                    "scanned_at": three_days_ago,
                    "analysis": {
                        "sugar": 0.0, "sodium_mg": 830.0,
                        "cons": ["High saturated fatty acids (51g per 100g)", "High sodium content increases fluid retention and diabetic renal strain"]
                    }
                }
            ]
        },
        {
            "key": "patient_key_hypertensive_2",
            "persona": "hypertensive",
            "streak": 9,
            "logs": [
                {"meal": "Oatmeal with Chia Seeds", "calories": 310.0, "protein": 11.0, "carbs": 48.0, "fat": 6.5},
                {"meal": "Grilled Tofu Salad", "calories": 290.0, "protein": 24.0, "carbs": 12.0, "fat": 14.0}
            ],
            "scans": [
                {
                    "product_name": "Maggi Masala Noodles",
                    "brand": "Nestle",
                    "score": 2,
                    "verdict": "Extreme Sodium Warning",
                    "calories": 389.0, "protein": 8.0, "carbs": 59.6, "fat": 13.5, "sodium": 1240.0, "fiber": 2.0, "sugar": 1.2,
                    "category": "Instant Noodles",
                    "scanned_at": today,
                    "analysis": {
                        "sugar": 1.2, "sodium_mg": 1240.0,
                        "cons": ["Sodium overload (1240mg) representing >60% of daily clinical hypertensive allowance", "Monosodium Glutamate and flavor enhancers INS 635 triggering vascular fluid pressure"]
                    }
                }
            ]
        },
        {
            "key": "patient_key_diabetic_3",
            "persona": "diabetic",
            "streak": 3,
            "logs": [
                {"meal": "Brown Rice & Lentil Tadka", "calories": 480.0, "protein": 16.0, "carbs": 72.0, "fat": 9.0}
            ],
            "scans": [
                {
                    "product_name": "Hide & Seek Chocolate Chip Cookies",
                    "brand": "Parle",
                    "score": 3,
                    "verdict": "Glycemic Threat",
                    "calories": 480.0, "protein": 5.5, "carbs": 72.0, "fat": 19.0, "sodium": 340.0, "fiber": 1.5, "sugar": 32.0,
                    "category": "Bakery",
                    "scanned_at": yesterday,
                    "analysis": {
                        "sugar": 32.0, "sodium_mg": 340.0,
                        "cons": ["High added sugar content (32g)", "Refined wheat flour (maida) triggers rapid insulin release"]
                    }
                }
            ]
        }
    ]
    
    with db_conn() as conn:
        # 1. Seed dietitian record
        conn.execute("INSERT OR REPLACE INTO dietitians (code, name, email, dietitian_key) VALUES (?, ?, ?, ?)",
                     (dietitian_code, dietitian_name, dietitian_email, dietitian_key))
        print(f"Seeded Dietitian: {dietitian_name} (Code: {dietitian_code}, Key: {dietitian_key})")
        
        for p in patients:
            dk = p["key"]
            # 2. Seed device metadata
            conn.execute("""
                INSERT OR REPLACE INTO devices (device_key, streak_days, last_scan_date, persona, month, scan_count)
                VALUES (?, ?, ?, ?, ?, 5)
            """, (dk, p["streak"], today, p["persona"], today[:7]))
            
            # 3. Seed patient-dietitian handshake link
            conn.execute("INSERT OR IGNORE INTO patient_cohorts (device_key, dietitian_code) VALUES (?, ?)",
                         (dk, dietitian_code))
            
            # 4. Seed daily logs
            conn.execute("DELETE FROM daily_logs WHERE device_key = ?", (dk,))
            for log in p["logs"]:
                conn.execute("""
                    INSERT INTO daily_logs (device_key, log_date, meal_name, calories, protein, carbs, fat)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (dk, today, log["meal"], log["calories"], log["protein"], log["carbs"], log["fat"]))
                
            # 5. Seed scan details
            for scan in p["scans"]:
                conn.execute("DELETE FROM scans WHERE device_key = ? AND product_name = ?", (dk, scan["product_name"]))
                conn.execute("""
                    INSERT INTO scans (
                        device_key, product_name, brand, score, verdict, calories, protein, carbs, fat, sodium, fiber, sugar,
                        persona, language, scanned_at, analysis_json, verified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'en', ?, ?, 1)
                """, (
                    dk, scan["product_name"], scan["brand"], scan["score"], scan["verdict"],
                    scan["calories"], scan["protein"], scan["carbs"], scan["fat"], scan["sodium"], scan["fiber"], scan["sugar"],
                    p["persona"], scan["scanned_at"], json.dumps(scan["analysis"])
                ))
            print(f"Seeded Patient Cohort data for: {dk} (Persona: {p['persona']})")
            
    print("\n[SUCCESS] Dietitian Cohort Seeding Completed Perfectly!")
    print(f"To view the dietitian dashboard: http://localhost:8000/dietitian/dashboard?key={dietitian_key}")
    print(f"To view the cohort clinical telemetry: http://localhost:8000/dietitian/cohort?cohort_code={dietitian_code}&key={dietitian_key}")

if __name__ == "__main__":
    seed_dietitian_cohort()
