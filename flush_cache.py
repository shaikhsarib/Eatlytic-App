import sqlite3
import os

db_path = "data/eatlytic.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Delete failed/unreliable entries from AI cache
    # (Checking result_json for 'UNRELIABLE' or 'error')
    cursor.execute("DELETE FROM ai_cache WHERE result_json LIKE '%UNRELIABLE%' OR result_json LIKE '%\"error\"%'")
    print(f"Purged {cursor.rowcount} poisoned entries from ai_cache")
    
    # Same for image fingerprints
    cursor.execute("DELETE FROM image_fingerprints WHERE result_json LIKE '%\"error\"%'")
    print(f"Purged {cursor.rowcount} poisoned entries from image_fingerprints")
    
    conn.commit()
    conn.close()
else:
    print("Database not found locally.")
