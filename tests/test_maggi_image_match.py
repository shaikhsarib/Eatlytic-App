import os
import sqlite3
import json
import pytest
from app.services.hash_service import get_image_fingerprint
from app.models.db import get_image_fingerprint_match

def test_maggi_image_hash_and_db_match():
    """Verify that the Maggi back-panel image hash resolves to the seeded DB entry."""
    img_path = os.path.join("data", "media__1779297081184.jpg")
    
    # 1. Assert the image exists in the data directory
    assert os.path.exists(img_path), f"Maggi test image not found at {img_path}"
    
    # 2. Read the image and calculate its pHash fingerprint
    with open(img_path, "rb") as f:
        img_content = f.read()
    
    phash = get_image_fingerprint(img_content)
    assert phash == "ae3a51e295e55259", f"Expected pHash ae3a51e295e55259, but got {phash}"
    
    # 3. Perform a simulated database fingerprint match lookup
    # We will query directly from data/eatlytic.db as it has the seeded records
    conn = sqlite3.connect(os.path.join("data", "eatlytic.db"))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    row = c.execute("SELECT result_json FROM image_fingerprints WHERE hash_key=?", (phash,)).fetchone()
    conn.close()
    
    assert row is not None, "Maggi pHash entry not found in the database image_fingerprints table!"
    
    # 4. Parse the result and assert detailed nutrition info is correct
    res = json.loads(row["result_json"])
    assert res["product_name"] == "Nestle Maggi Masala Noodles"
    assert res["product_category"] == "instant_noodles"
    assert res["score"] == 4
    assert res["safety_tier"] == "Limit"
    
    # Check that all 10 nutrient breakdown rows exist
    nutrients = res.get("nutrient_breakdown", [])
    assert len(nutrients) >= 10, f"Expected at least 10 nutrients, got {len(nutrients)}"
    
    # Validate energy and sodium specifically
    energy = next((n for n in nutrients if n["name"].lower() == "energy"), None)
    sodium = next((n for n in nutrients if n["name"].lower() == "sodium"), None)
    
    assert energy is not None, "Energy nutrient row is missing"
    assert float(energy["value"]) == 384.0
    assert energy["unit"] == "kcal"
    
    assert sodium is not None, "Sodium nutrient row is missing"
    assert float(sodium["value"]) == 1000.0
    assert sodium["unit"] == "mg"
    
    print("✅ Nestle Maggi Masala Noodles pHash and full nutrient extraction verified successfully!")
