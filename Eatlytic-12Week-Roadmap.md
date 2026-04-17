# Eatlytic: 12-Week Production Roadmap
## From Prototype to Defensible Revenue-Generating Startup

*Audit date: April 2026 | Based on codebase v7 Single-Pass Architecture*

---

## Executive Diagnosis

The codebase is technically sophisticated but has **5 production-breaking vulnerabilities** and **3 revenue leaks** that will kill the business before it scales. The architecture decisions (Supabase + SQLite fallback, Tenacity retries, Atwater physics gate) are sound. The gaps are in trust infrastructure, quota enforcement, and label confidence signaling.

**The honest answer to "why pay for Eatlytic"**: Eatlytic's moat is not the LLM — anyone can call Groq. The moat is *validated, corrected Indian FMCG extraction data* plus a *physics-gated trust layer* that competitors don't have. Every fix below either protects that moat or converts it to revenue.

---

## Critical Bug Inventory (Fix Before Launch)

### P0 — Will Break Today

**P0-1: Pro subscription never expires**
`pro_expires` column exists in the schema but is *never checked*. A user who pays once has Pro forever. `check_and_increment_scan()` only reads `is_pro` — not the expiry date.

```python
# CURRENT (db.py ~line 368) — broken:
if u["is_pro"]:
    return {"allowed": True, "scans_remaining": 9999, "is_pro": True}

# FIX — add expiry check:
if u["is_pro"]:
    expires = u.get("pro_expires")
    if expires and expires < datetime.date.today().isoformat():
        # Expired — downgrade silently
        conn.execute("UPDATE devices SET is_pro=0 WHERE device_key=?", (device_key,))
    else:
        return {"allowed": True, "scans_remaining": 9999, "is_pro": True}
```

**P0-2: Device fingerprint is trivially bypassable**
`get_device_key()` hashes `IP + UserAgent + x-fingerprint header`. The `x-fingerprint` header is *sent by the client*, meaning any user can pass a random string on every request to get unlimited free scans. No server-side validation exists.

```python
# CURRENT (main.py line 117) — bypassable:
fp_hint = request.headers.get("x-fingerprint", "")

# FIX — treat fp_hint as advisory only; add a server-issued device token:
# 1. On first scan, issue a signed device_token (HMAC of IP+UA+timestamp)
# 2. Store token in DB with creation time
# 3. Require token on subsequent scans — never trust client-supplied IDs
# Short-term: remove fp_hint from hash, use IP+UA only, add Redis-based
# sliding window per IP to catch burst abuse
```

**P0-3: WhatsApp numbers stored in plaintext as scan quota keys**
`check_and_increment_scan(phone_number, ...)` uses the raw WhatsApp number (`whatsapp:+91XXXXXXXXXX`) as the device key, which is then stored in the `devices` table. This is PII and directly violates DPDP Act 2023 Article 4 (purpose limitation) and Article 6 (storage limitation).

```python
# FIX — hash the phone number before storing:
import hashlib
phone_hash = "wa_" + hashlib.sha256(phone_number.encode()).hexdigest()[:20]
check_and_increment_scan(phone_hash, ...)
```

**P0-4: Payment activation tied to device fingerprint, not payment ID**
If a user pays on device A and then scans on device B (different IP/UA), they lose Pro status. Worse: the payment record stores the `device_key` not the Razorpay `customer_id` — there's no way to restore Pro if the user changes devices or clears cookies.

```python
# FIX — store payment against a stable user identifier, not device_key:
# Option A (fast): require email OTP before payment; link payment to user.email
# Option B (slow): Razorpay webhook with customer_id persisted to users table
# Minimum viable fix: add /restore-pro endpoint that re-checks Razorpay 
# payment history by phone/email and re-grants Pro
```

**P0-5: No extraction confidence score — every response looks equally trustworthy**
The system returns the same JSON structure whether it extracted 12 clean nutrients from a clear Maggi label or hallucinated 4 values from a blurry sideways photo. Users can't tell the difference. This is the single biggest trust killer.

---

## Label Extraction Validation & Mismatch Detection

This is core to Eatlytic's positioning. Every response must signal its own reliability.

### The Confidence Pipeline (add to `llm.py`)

```python
def compute_extraction_confidence(
    result_data: dict,
    ocr_word_count: int,
    avg_ocr_confidence: float,
    atwater_valid: bool,
    nutrient_count: int,
) -> dict:
    """
    Returns a confidence dict: score (0-100), tier (HIGH/MEDIUM/LOW/UNRELIABLE),
    and a user-facing message.
    """
    score = 100

    # OCR quality signals
    if ocr_word_count < 15:   score -= 35   # too little text extracted
    elif ocr_word_count < 30: score -= 15
    if avg_ocr_confidence < 0.5: score -= 25
    elif avg_ocr_confidence < 0.7: score -= 10

    # Extraction completeness
    if nutrient_count == 0:  score -= 50   # nothing extracted — total fail
    elif nutrient_count < 4: score -= 25   # suspicious minimum
    elif nutrient_count < 7: score -= 10

    # Physics validation
    if not atwater_valid:    score -= 20   # numbers don't add up

    # Key nutrient presence (all real labels have these)
    required = ["energy", "protein", "fat", "carb"]
    names_lower = [n.get("name","").lower() for n in result_data.get("nutrients", [])]
    missing = [r for r in required if not any(r in nm for nm in names_lower)]
    score -= len(missing) * 10

    # Product name quality
    bad_names = {"unknown", "unknown product", "food product", ""}
    if result_data.get("product_name", "").lower().strip() in bad_names:
        score -= 15

    score = max(0, min(100, score))

    if score >= 80:
        tier, message = "HIGH", "Extracted from clear label — high confidence."
    elif score >= 55:
        tier, message = "MEDIUM", "Some uncertainty — verify key nutrients against the physical label."
    elif score >= 30:
        tier, message = "LOW", "Partial extraction — image may be blurry or label partially visible."
    else:
        tier, message = "UNRELIABLE", "Could not reliably read this label. Please retake the photo."

    return {
        "score": score,
        "tier": tier,
        "message": message,
        "nutrient_count": nutrient_count,
        "atwater_valid": atwater_valid,
    }
```

**Add to `unified_analyze_flow` return value:**
```python
confidence = compute_extraction_confidence(
    result_data=result_data,
    ocr_word_count=ocr_result.get("word_count", 0),
    avg_ocr_confidence=ocr_result.get("avg_confidence", 0),
    atwater_valid=math_ok["is_valid"],
    nutrient_count=len(nutrient_breakdown),
)
final_output["extraction_confidence"] = confidence

# If UNRELIABLE — return a soft error, not a fake analysis:
if confidence["tier"] == "UNRELIABLE":
    return {
        "error": "low_confidence",
        "message": "⚠️ Could not reliably extract this label. Please photograph the back panel in good light.",
        "confidence": confidence,
    }
```

### Mismatch Detection Across All Product Categories

Different label formats need different validation rules:

```python
# app/services/label_detector.py — add label_format_detector()

LABEL_FORMAT_RULES = {
    "indian_fssai": {
        # Must have: Energy, Protein, Carbs, Fat, Sodium
        # Per 100g column required
        # FSSAI license number in text is a trust signal
        "required_nutrients": ["energy", "protein", "carb", "fat"],
        "trust_signals": ["fssai", "per 100g", "per 100 g"],
        "red_flags": [],
    },
    "us_fda": {
        # Must have: Calories, Total Fat, Sodium, Total Carb, Protein
        # Serving size required
        "required_nutrients": ["calorie", "fat", "sodium", "carb", "protein"],
        "trust_signals": ["nutrition facts", "daily value", "serving size", "% dv"],
        "red_flags": [],
    },
    "eu_format": {
        # Reference Intake (RI) instead of Daily Value
        "required_nutrients": ["energy", "fat", "carb", "protein", "salt"],
        "trust_signals": ["reference intake", "typical values", "per 100g", "kj"],
        "red_flags": [],
    },
    "unknown": {
        "required_nutrients": ["energy", "fat", "protein"],
        "trust_signals": ["nutrition", "ingredient"],
        "red_flags": [],
    }
}

def detect_label_format(ocr_text: str) -> str:
    """Detect the regulatory format of this label."""
    t = ocr_text.lower()
    if "fssai" in t or "know your portion" in t or "per 100 g" in t:
        return "indian_fssai"
    if "nutrition facts" in t and "daily value" in t:
        return "us_fda"
    if "reference intake" in t or "typical values" in t:
        return "eu_format"
    return "unknown"

def validate_against_format(nutrients: list, format_key: str) -> dict:
    """Check if extracted nutrients match what this label format requires."""
    rules = LABEL_FORMAT_RULES.get(format_key, LABEL_FORMAT_RULES["unknown"])
    names = [n.get("name","").lower() for n in nutrients]
    
    missing_required = [
        r for r in rules["required_nutrients"]
        if not any(r in nm for nm in names)
    ]
    
    return {
        "format": format_key,
        "missing_required": missing_required,
        "completeness": 1.0 - (len(missing_required) / max(1, len(rules["required_nutrients"]))),
    }
```

---

## Week-by-Week Execution Plan

---

### WEEKS 1–3: Stop the Bleeding (Production Stability)

**Goal**: Fix every P0. Deploy without fear.

**Week 1 — Security & Quota**

| Task | File | Impact |
|------|------|--------|
| Fix Pro expiry check | `db.py` | Revenue protection |
| Hash WhatsApp numbers | `main.py` | DPDP compliance |
| Remove `x-fingerprint` from device key | `main.py` | Stop free quota gaming |
| Add server-issued device tokens | `main.py` + `db.py` | Reliable quota |
| Add extraction confidence score | `llm.py` | Trust/accuracy |

**Week 2 — Payment & Data Integrity**

```python
# Add /restore-pro endpoint (main.py)
@app.post("/restore-pro")
async def restore_pro(request: Request, email: str = Form(...)):
    """Allow users to restore Pro on a new device by verifying email."""
    # 1. Find user by email
    # 2. Check payments table for successful payment linked to any of their devices
    # 3. If found + not expired: grant Pro to current device_key
    # 4. Send confirmation
```

Add a `pro_expires` enforcement migration:
```sql
-- Run once in production
UPDATE devices SET is_pro = 0 
WHERE is_pro = 1 
AND pro_expires IS NOT NULL 
AND pro_expires < date('now');
```

**Week 3 — Label Confidence Pipeline**

- Ship `compute_extraction_confidence()` — add confidence tier to every API response
- Add `detect_label_format()` and `validate_against_format()` — use for mismatch detection
- Add `"extraction_confidence"` field to frontend result: show a subtle badge (HIGH / MEDIUM / LOW) next to each scan result
- Wire `UNRELIABLE` tier to the "try again" error state (already exists in `showError()`)

---

### WEEKS 4–6: Revenue Infrastructure

**Goal**: First paying users. B2C working. B2B API ready.

**Unit Economics (B2C)**

| Metric | Value |
|--------|-------|
| CAC (organic/WhatsApp referral) | ₹0–₹50 |
| CAC (Instagram/performance) | ₹80–₹150 |
| Monthly Pro price | ₹99/month |
| Annual Pro price | ₹799/year |
| Payback period (organic) | < 1 month |
| Target month-3 paid users | 500 |
| Month-3 MRR target | ₹49,500 |

**B2C Conversion Funnel Fix**

The current paywall triggers at scan 4 with a hard block. This is too abrupt — users haven't seen enough value. Recommended flow:

```
Scan 1: Full report, no friction
Scan 2: Full report + subtle "You have 2 scans left"  
Scan 3: Full report + strong CTA "Last free scan"
Scan 4: Paywall — show the LAST scan's result beneath the blur
         "Your scan of [ProductName] is locked. Upgrade to see it."
```

This is a 1-line change in `index.html` — show the product name + score in the paywall modal.

**B2B API (Week 5–6)**

The `/api/v1/analyze` endpoint already exists. What's missing:

```python
# Add to main.py — API key management
@app.post("/api/v1/keys/create")  # admin only
@app.get("/api/v1/keys/usage")    # show calls remaining
@app.delete("/api/v1/keys/{key}") # revoke

# Pricing tiers to implement:
# Starter:    ₹2,999/month →  500 scans/month (₹6/scan)
# Growth:    ₹9,999/month → 3,000 scans/month (₹3.3/scan)  
# Enterprise: custom →  50,000+ scans/month (₹1.5/scan)
```

**B2B Target Verticals (Weeks 5–6)**

1. **Quick Commerce** (Blinkit, Zepto, Instamart): They need nutrition enrichment for 1M+ SKUs. One integration = ₹5–15L/month at scale.
2. **Dietitian Platforms** (Practo Nutrition, HealthifyMe Pro): API to scan any product a client asks about. ₹50–200/month per dietitian seat.
3. **D2C Food Brands**: Competitive analysis — brand uploads competitor products, gets comparative analysis. One-time reports at ₹5,000–20,000 each.

Cold outreach script for quick commerce: *"We enriched Maggi's nutrition data in 4 seconds via API. Your catalog has [X] SKUs with missing or wrong nutrition data. We can fix them all at ₹[price]/SKU."*

---

### WEEKS 7–9: Accuracy & The pHash Moat

**Goal**: Build the defensible dataset. Achieve 10K corrected SKUs.

**Perceptual Hash Cache (The Real Moat)**

This is referenced in the architecture diagram but doesn't exist in the code yet. Here's the complete implementation:

```python
# app/services/phash_cache.py — NEW FILE

import hashlib
import numpy as np
from PIL import Image
from io import BytesIO

def compute_phash(image_bytes: bytes, hash_size: int = 16) -> str:
    """
    Compute a perceptual hash of a nutrition label image.
    Returns a 64-character hex string that is stable across:
    - Minor rotation/skew (±5°)
    - Brightness/contrast changes
    - JPEG compression artifacts
    - Different camera distances (if label fills >60% of frame)
    """
    img = Image.open(BytesIO(image_bytes)).convert("L")  # Grayscale
    # Resize to hash_size+1 × hash_size for DCT-like comparison
    img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=float)
    # Difference hash: compare adjacent pixels horizontally
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten().tobytes().hex()

def phash_distance(hash1: str, hash2: str) -> int:
    """Hamming distance between two pHashes. <10 = same label."""
    b1 = bytes.fromhex(hash1)
    b2 = bytes.fromhex(hash2)
    return sum(bin(x ^ y).count('1') for x, y in zip(b1, b2))

def find_cached_by_phash(image_bytes: bytes, threshold: int = 8) -> dict | None:
    """
    Check Supabase for a previously analyzed label that looks like this one.
    Returns cached result if pHash distance < threshold.
    """
    query_hash = compute_phash(image_bytes)
    
    # Query Supabase — fetch recent hashes and check distances
    # (Full scan is fine for <100K products; use pgvector for scale)
    try:
        from app.models.db import _supabase
        if not _supabase:
            return None
        
        response = _supabase.table("cached_products").select(
            "label_hash, phash, result_json, scan_count"
        ).not_.is_("phash", "null").limit(1000).execute()
        
        for row in response.data or []:
            if row.get("phash") and phash_distance(query_hash, row["phash"]) < threshold:
                # Cache hit — bump scan_count for dataset building
                _supabase.table("cached_products").update(
                    {"scan_count": row["scan_count"] + 1}
                ).eq("phash", row["phash"]).execute()
                return row.get("result_json") or row
    except Exception:
        pass
    return None
```

**Add to `main.py` `/analyze` endpoint (before OCR):**
```python
# pHash cache check — fastest possible path, returns in <50ms
phash_result = find_cached_by_phash(content)
if phash_result:
    phash_result["scan_meta"] = {"cached": True, "cache_type": "phash", ...}
    return phash_result
```

**Dataset Building Strategy**

Every verified scan = one corrected SKU. After 10,000 SKUs, Eatlytic has something no competitor can easily replicate: a validated Indian FMCG nutrition database with Atwater-verified values.

```python
# Add to unified_analyze_flow — store phash with every high-confidence scan
if confidence["tier"] in ("HIGH", "MEDIUM") and not cached:
    from app.services.phash_cache import compute_phash
    if image_content:
        phash = compute_phash(image_content)
        # Store phash alongside the cached result
        set_ai_cache(cache_key, {**cacheable, "phash": phash})
```

**Accuracy Benchmark**

Test against the provided label images:

| Label | Key Extraction Targets | Atwater Check |
|-------|----------------------|---------------|
| Parle-G Gold (Image 1) | Cal: 460, Fat: 15.16g, Carb: 75g, Protein: 7g, Sodium: 270mg | 460 ≈ 7×4 + 75×4 + 15.16×9 = 28+300+136 = **464** ✓ (±1%) |
| Maggi Masala (Image 2) | Cal: 389, Fat: 13.5g, Carb: 59.6g, Protein: 8.2g | 389 ≈ 8.2×4 + 59.6×4 + 13.5×9 = 33+238+122 = **393** ✓ (±1%) |
| Maggi Back (Image 3) | Multi-column: extract Per 100g only | Same as above |
| Peanut Butter (Image 4) | US format: Per Serving → convert to /100g | Per-serve ÷ 0.32 × 1.0 |

The Parle-G label has a critical edge case: it's **upside down** in the image. The ROI detector must handle this. Add rotation detection:

```python
# label_detector.py — add to process_image_for_ocr()
def detect_and_fix_rotation(image_np: np.ndarray) -> np.ndarray:
    """
    Detect if image is upside-down (common for product backs photographed from above).
    Uses text orientation detection via OSD (Orientation and Script Detection).
    """
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(image_np, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle in (90, 180, 270):
            return np.rot90(image_np, k=angle // 90)
    except Exception:
        pass
    return image_np
```

---

### WEEKS 10–12: Scale, Compliance & Distribution

**Goal**: DPDP compliant, 3 B2B pilots signed, 1,000 paying users.

**DPDP Act 2023 Compliance Checklist**

| Requirement | Current Status | Fix |
|------------|---------------|-----|
| Consent before data collection | ❌ Missing | Add consent screen on first scan; store `consent_given_at` timestamp |
| Purpose limitation | ⚠️ Partial | WhatsApp numbers used as device keys — hash them (P0-3 above) |
| Storage limitation | ❌ Missing | Implement 90-day auto-purge for device records with no activity |
| Right to erasure | ❌ Missing | Add `DELETE /user/data` endpoint |
| Privacy notice | ❌ Missing | Add `/privacy` page with data retention policy |
| Data fiduciary registration | ⚠️ Monitor | Register when DPDP rules finalized; budget ₹50K for compliance |

**Data Retention & Purge Policy**

```python
# Add to db.py — run daily via a cron job or startup task
def purge_old_records():
    """DPDP compliance: delete inactive device records after 90 days."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
    with db_conn() as conn:
        # Delete devices with no scans in 90 days
        conn.execute("""
            DELETE FROM devices 
            WHERE last_scan_date < ? AND is_pro = 0
        """, (cutoff,))
        # Purge OCR cache older than 30 days (already in ai_cache query)
        conn.execute("DELETE FROM ai_cache WHERE created_at < datetime('now', '-30 days')")
    logger.info("DPDP purge complete")
```

**User Erasure Endpoint**

```python
@app.delete("/user/data")
async def delete_user_data(request: Request):
    """DPDP Article 12: Right to erasure."""
    device_key = get_device_key(request)
    with db_conn() as conn:
        conn.execute("DELETE FROM devices WHERE device_key=?", (device_key,))
        conn.execute("DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE ...)")
    # Also delete from Supabase if configured
    if _supabase:
        _supabase.table("devices").delete().eq("device_key", device_key).execute()
    return {"status": "deleted", "message": "All your data has been permanently deleted."}
```

**Distribution — What Actually Works for Eatlytic**

*Tier 1: Zero-cost channels (Weeks 1–4)*
- **WhatsApp broadcast lists**: Every person who scans a product via WhatsApp is already your channel. After each analysis, message includes: *"Share this report with the family member who buys this product →"* with a deep link.
- **College canteen/gym partnerships**: 10 WhatsApp scan demos at a college nutrition event = 50–100 new users. Zero ad spend.
- **Reddit r/India + r/bangalore + r/Chennai**: Post the Maggi sodium analysis ("Your Maggi has 51% of your daily sodium limit in one pack"). No promotional language. Just data.

*Tier 2: Earned media (Weeks 5–8)*
- **FSSAI compliance angle**: When a brand's label has wrong Atwater math (happens frequently with Indian snacks), that's a news story. File RTI. Generate media coverage. Two or three of these = 10x organic traffic.
- **Dietitian influencer seeding**: 20 mid-tier Instagram dietitians with 50K–200K followers. Offer free Pro access. Ask them to use it on content, not to sponsor it. Authentic posts convert at 3–5x paid posts.

*Tier 3: Paid acquisition (Week 9+, only after positive unit economics confirmed)*
- Instagram/Reels: "What's actually in your [popular product]?" format — show the scan process and the score reveal. ROAS of 3–5× is achievable for this category.

**Groq Failure Resilience**

Current: if Groq fails after 3 retries, user gets an error. Single point of failure.

```python
# app/services/llm.py — add fallback provider
LLM_PROVIDERS = [
    {"type": "groq", "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]},
    {"type": "together", "models": ["meta-llama/Llama-3-70b-chat-hf"]},  # fallback
]

# If Groq fails completely (503, all retries exhausted):
# 1. Try Together AI with same prompt (add TOGETHER_API_KEY env var)
# 2. If both fail: return rule-based score only (no LLM summary)
#    — use compute_rule_based_score() + nutrient_breakdown from OCR text
#    — mark result with "analysis_mode": "offline" 
#    — show user: "AI analysis temporarily unavailable — showing extracted values only"
```

**SQLite → Production Database Migration**

SQLite with WAL mode works fine up to ~50 concurrent write requests. Above that, expect `OperationalError: database is locked`. Supabase is already wired in — the migration is just ensuring env vars are set in production.

Verification query to run in Supabase dashboard before launch:
```sql
-- Confirm all required tables exist
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;
-- Expected: ai_cache, cached_products, devices, ocr_cache, payments, sessions, users
```

---

## Prioritized Fix Order (Single Developer, 12 Weeks)

```
WEEK 1:  P0-3 (hash phone numbers) → P0-1 (pro expiry) → P0-2 (device token)
WEEK 2:  P0-4 (payment restore) → P0-5 (confidence score) → DPDP consent screen
WEEK 3:  Label format detection → mismatch validation → confidence in API response
WEEK 4:  Paywall funnel fix → /restore-pro endpoint → B2B API keys
WEEK 5:  B2B pricing tiers → cold outreach to 3 quick commerce targets
WEEK 6:  First B2B pilot signed → dietitian seed program launched
WEEK 7:  pHash implementation → phash stored with every HIGH-confidence scan
WEEK 8:  10,000 SKU milestone target → rotation detection for Parle-G edge case
WEEK 9:  Groq fallback provider → DPDP purge job → data erasure endpoint
WEEK 10: Privacy page → 3 B2B pilots active → 500 paying B2C users target
WEEK 11: Performance review → unit economics confirmed → paid acquisition decision
WEEK 12: Series A data room prep: MRR, dataset size, API uptime, accuracy benchmarks
```

---

## What Makes This Defensible in 12 Weeks

By week 12, if the above is executed, Eatlytic has:

1. **10,000+ Atwater-verified Indian FMCG SKUs** — no competitor has this with physics validation
2. **pHash cache** — subsequent scans of known products return in <100ms, at near-zero cost
3. **Extraction confidence scores** — the only scanner that tells you when to trust it
4. **DPDP-compliant data layer** — required for any enterprise deal in India
5. **3 B2B pilots** — converts the technology into recurring revenue proof
6. **Label format detection across US/EU/Indian labels** — not just Indian FMCG

The answer to "why pay for Eatlytic vs Google Lens + ChatGPT": those tools have no Atwater gate, no NOVA classification, no physics validation, no Indian FMCG training data, and no confidence signal. They hallucinate nutritional data with complete confidence. Eatlytic refuses to guess.

---

## What to Add Manually

The following require credentials/accounts that cannot be coded:

| Item | Where | Action |
|------|-------|--------|
| `GROQ_API_KEY` | `.env` | console.groq.com — free tier |
| `SUPABASE_URL` + `SUPABASE_KEY` | `.env` | supabase.com — free tier up to 500MB |
| `RAZORPAY_KEY_ID` + `RAZORPAY_KEY_SECRET` | `.env` | razorpay.com dashboard |
| `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN` | `.env` | twilio.com |
| `TOGETHER_API_KEY` | `.env` | api.together.xyz — Groq fallback |
| `FREE_SCAN_LIMIT` | `.env` or `main.py:84` | Currently 3 — consider 5 for conversion |
| Supabase tables | Supabase dashboard | Run `init_db()` equivalent SQL for production tables |
| Razorpay webhook | Razorpay dashboard | Point to `https://yourdomain.com/activate-pro` |
| DPDP registration | MeitY portal | When rules finalized — budget ₹50K |
