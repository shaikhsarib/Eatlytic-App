import sqlite3
import os

db_path = "data/eatlytic.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tables to clean
    tables = ["ai_cache", "image_fingerprints"]
    
    for table in tables:
        # Delete entries that are clearly wrong (Processed Meat for chocolate, or score 1 with no nutrients)
        # We also clear entries with score 1 because they are usually failed scans.
        cursor.execute(f"DELETE FROM {table} WHERE result_json LIKE '%Meat%' OR result_json LIKE '%\"score\": 1%'")
        print(f"Purged {cursor.rowcount} entries from {table}")
    
    conn.commit()
    conn.close()
else:
    print(f"Database not found at {db_path}")
