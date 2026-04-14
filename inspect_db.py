import sqlite3
import json

def inspect():
    conn = sqlite3.connect('data/eatlytic.db')
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            print(f"ID: {row['id']}")
            print(f"Product: {row['product_name']}")
            print(f"Calories: {row['calories']}")
            print(f"Protein: {row['protein']}")
            print(f"Fat: {row['fat']}")
            print(f"Carbs: {row['carbs']}")
            print(f"Analysis JSON: {row['analysis_json']}")
        else:
            print("No scans found.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    inspect()
