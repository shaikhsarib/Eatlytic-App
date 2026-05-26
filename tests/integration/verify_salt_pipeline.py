# -*- coding: utf-8 -*-
"""Scan test via the /parse-voice-meal and /analyze endpoints."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests, json, time

BASE = "http://127.0.0.1:8000"

# ── Test 1: Voice Meal (Tata Salt) ──
print("=" * 60)
print("TEST 1: Voice Meal Scan - Tata Salt")
print("=" * 60)
t0 = time.time()
resp = requests.post(f"{BASE}/parse-voice-meal", data={
    "text": "Tata Salt iodised salt per 100g Energy 0 Kcal Carbohydrate 0g Protein 0g Fat 0g Fatty Acids 0g Cholesterol 0mg Sodium 38740mg Iodine 15ppm Ingredients iodised salt potassium iodate anti-caking agent E-536",
    "persona": "General Adult",
    "language": "en",
}, timeout=360)
elapsed = time.time() - t0
print(f"  Status: {resp.status_code} ({elapsed:.1f}s)")
if resp.status_code == 200:
    d = resp.json()
    print(f"  Product:  {d.get('product_name', d.get('meal_name', '?'))}")
    print(f"  Score:    {d.get('score', '?')}/10")
    print(f"  Verdict:  {d.get('verdict', d.get('summary', '?'))}")
    for n in d.get("nutrient_breakdown", d.get("nutrients", []))[:6]:
        name = n.get("name", "?")
        val = n.get("value", "?")
        unit = n.get("unit", "")
        print(f"    - {name}: {val} {unit}")
else:
    print(f"  Error: {resp.text[:300]}")

# ── Test 2: Health endpoint ──
print()
print("=" * 60)
print("TEST 2: Health Check")
print("=" * 60)
h = requests.get(f"{BASE}/health").json()
print(f"  Status:  {h.get('status')}")
print(f"  Engine:  {h.get('engine')}")
print(f"  Version: {h.get('version')}")

# ── Test 3: Scan Quota ──
print()
print("=" * 60)
print("TEST 3: Scan Quota")
print("=" * 60)
q = requests.get(f"{BASE}/scan-quota").json()
print(f"  Used:      {q.get('scans_used')}")
print(f"  Remaining: {q.get('scans_remaining')}")
print(f"  Pro:       {q.get('is_pro')}")

# ── Test 4: History ──
print()
print("=" * 60)
print("TEST 4: Scan History")
print("=" * 60)
hist = requests.get(f"{BASE}/api/v1/history").json()
print(f"  Total scans in history: {len(hist)}")
for s in hist[:3]:
    print(f"    - [{s.get('score', '?')}/10] {s.get('product_name', '?')} ({s.get('scanned_at', '?')[:10]})")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
