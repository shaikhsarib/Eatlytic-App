import json
import requests
import sys

def run_simulated_scan():
    # Local API server URL
    url = "http://127.0.0.1:8000/api/v1/analyze"
    
    # 1. We mock a diabetic patient's device profile (diabetic, carrying the TCF7L2 genetic risk variant)
    device_key = "whatsapp_9876543210"
    
    # First, let's register their genomic/biomarker profile so the system knows about their condition
    profile_url = "http://127.0.0.1:8000/personalized/profile"
    profile_payload = {
        "genetic_snps": {"rs7903146": "TT"}, # TCF7L2 risk allele (severe glucose intolerance risk)
        "biomarkers": {"hba1c": 6.8}         # Prediabetic/Diabetic HbA1c
    }
    
    try:
        # Register genomic profile
        reg_resp = requests.post(profile_url, json=profile_payload, headers={"X-Device-Key": device_key})
        if reg_resp.status_code == 200:
            print("[1/2] Successfully registered patient's genomic profile (TCF7L2 rs7903146 + HbA1c 6.8).")
        else:
            print(f"[!] Profile registration failed: {reg_resp.text}")
            return
    except Exception as e:
        print("[!] Local server is not running. Please start it using: uvicorn main:app --reload")
        return

    # 2. Now, let's simulate a food scan containing High-GI starch (Maida / Refined wheat flour)
    scan_payload = {
        "product_name": "Instant Masala Noodles",
        "brand": "ActiveFoods",
        "category": "Instant Noodles",
        "nutrients": {
            "calories": 389.0,
            "protein": 8.0,
            "carbs": 59.6,
            "fat": 13.5,
            "sodium_mg": 1240.0,
            "sugar": 1.2
        },
        "ingredients_raw": "Refined wheat flour (Maida), Palm oil, Salt, starch, MSG, Flavor enhancers",
        "persona": "diabetic"
    }
    
    print("\n[2/2] Scanning product: 'Instant Masala Noodles'...")
    resp = requests.post("http://127.0.0.1:8000/personalized/scan", json=scan_payload, headers={"X-Device-Key": device_key})
    
    if resp.status_code == 200:
        result = resp.json()
        print("\n================== EATLYTIC CLINICAL VERDICT ==================")
        print(f"Product Name : {result.get('product_name')} ({result.get('brand')})")
        print(f"Safety Score : {result.get('score')}/10 ({result.get('safety_tier')} Safety Tier)")
        print(f"Clinical Verdict : {result.get('clinical_audit', {}).get('verdict')}")
        print("\nFlagged Threat Warnings:")
        for con in result.get("cons", []):
            print(f"  - {con}")
        print("\nHealthy Alternative Swap:")
        print(f"  {result.get('better_alternative')}")
        print("==============================================================")
    else:
        print(f"[!] Scan API failed: {resp.text}")

if __name__ == "__main__":
    run_simulated_scan()
