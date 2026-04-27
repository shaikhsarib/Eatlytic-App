# 🔥 EATLYTIC — COMPLETE BLUEPRINT
## From Alpha to Investment-Ready in 90 Days

> **Version:** 1.0 | **Date:** April 27, 2026 | **Status:** Pre-Seed / Alpha
> **Author:** Sarib Shaikh | **Perspectives:** YC Partner, Health-Tech Investor, AI Specialist, CEO Mentor

---

## TABLE OF CONTENTS
1. [Product Architecture Blueprint](#1-product-architecture-blueprint)
2. [Security Hardening Blueprint](#2-security-hardening-blueprint)
3. [User Authentication Blueprint](#3-user-authentication-blueprint)
4. [Payment Flow Blueprint](#4-payment-flow-blueprint)
5. [Frontend Blueprint](#5-frontend-blueprint)
6. [API & Backend Blueprint](#6-api--backend-blueprint)
7. [Trust & Compliance Blueprint](#7-trust--compliance-blueprint)
8. [Growth & Distribution Blueprint](#8-growth--distribution-blueprint)
9. [Technical Debt Roadmap](#9-technical-debt-roadmap)
10. [90-Day Execution Timeline](#10-90-day-execution-timeline)
11. [Investment Readiness Checklist](#11-investment-readiness-checklist)

---

## 1. PRODUCT ARCHITECTURE BLUEPRINT

### 1.1 Core Value Proposition

**"Eatlytic reads ANY food label in the world — no barcode needed. We validate the physics of nutrition claims so you know what you're actually eating."**

Key differentiators:
- Universal coverage (works on any text-based label)
- Atwater physics validation (catches fake calorie claims)
- Multi-language OCR (18+ scripts)
- NOVA-4 ultra-processed food detection
- Fake marketing claim detection ("no added sugar" but contains maltodextrin)

### 1.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │
│  │   Web App   │  │ WhatsApp Bot│  │     Future: Mobile Apps     │ │
│  │  (Vanilla)  │  │  (Twilio)   │  │   (React Native/Flutter)    │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────────────┘ │
└─────────┼────────────────┼─────────────────────────────────────────┘
          └────────┬───────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                           │
│  Rate Limiter (slowapi) │ CORS Whitelist │ API Key Auth │ Admin Auth│
└──────────────────┬──────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────┐
│                    SERVICE LAYER (16 Modules)                        │
│  OCR Engine │ Image Repair │ Label Detector │ LLM Brain │           │
│  Fake Detector │ Duel Service │ Alternatives │ pHash Cache │        │
│  Research Engine │ Explain Engine │ Formatter │ Auth │ Payments     │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────┐
│                    PERSISTENCE LAYER                                │
│  SQLite (current) → Supabase (production)  │  File System (cache)  │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                                │
│  Groq (LLM) │ Together AI (fallback) │ Twilio (WhatsApp) │ Razorpay│
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow (Single Scan)

```
User Uploads Image
  │
  ▼
[1] Image Quality Check → Auto-Enhance if blurry
  │
  ▼
[2] pHash Cache Lookup → Cache Hit? Return instant result
  │
  ▼
[3] Label Detection (MSER ROI + Deskew + CLAHE)
  │
  ▼
[4] OCR Extraction (auto-script, 3-pass retry, line reconstruction)
  │
  ▼
[5] Label Validation → Invalid? → Vision AI Fallback (Groq/Together)
  │
  ▼
[6] LLM Analysis (super-prompt, single call, extract all nutrients)
  │
  ▼
[7] Atwater Physics Validation → FAIL? → AI Self-Correction retry
  │
  ▼
[8] DNA Override (fake claims → score=2, NOVA-4 → cap score at 3)
  │
  ▼
[9] Explanation Engine (RDA benchmarks, NOVA classification)
  │
  ▼
[10] Persistence (save scan, cache pHash, update food_products)
  │
  ▼
[11] Response to User
```

### 1.4 Cost Architecture

| Tier | Users | Scans/Day | API Cost/Month | Hosting | Total/Month |
|------|-------|-----------|----------------|---------|-------------|
| Current | 0 | 0 | $0 (free tier) | $0 (HF Spaces) | ~$1 |
| Seed | 1K | 3K | $270 (Groq) | $50 (VPS) | ~$320 |
| Growth | 10K | 30K | $2,700 (Groq) | $200 (dedicated) | ~$2,900 |
| Scale | 100K | 300K | $0 (Gemma self-hosted) | $500 (cluster) | ~$500 |

> **Critical inflection:** 10K DAU is when Groq costs become unsustainable. Gemma 4 self-hosting must be production-ready BEFORE you hit this.

---

## 2. SECURITY HARDENING BLUEPRINT

### 2.1 Secret Management

```
Development: .env (local only, never committed)
Staging:     GitHub Secrets (repo settings)
Production:  HuggingFace Spaces Secrets / Cloud provider secrets
Runtime:     os.environ.get() — NO hardcoded fallbacks
```

### 2.2 Security Checklist

| # | Control | Priority | Status |
|---|---------|----------|--------|
| 1 | No secrets in git | P0 | ✅ Fixed (BFG needed for history) |
| 2 | API key rotation quarterly | P0 | ⚠️ Rotate NOW |
| 3 | CORS exact origins only | P0 | ⚠️ Wildcard `*.hf.space` active |
| 4 | Rate limiting per-endpoint | P0 | ✅ Done (slowapi) |
| 5 | Admin auth env-var only | P0 | ✅ Fixed (no default) |
| 6 | Payment HMAC verification | P0 | ✅ Done |
| 7 | Input sanitization | P1 | ❌ Missing |
| 8 | Cookie-based device IDs | P1 | ❌ Using MD5(IP+UA) |
| 9 | DPDP compliance | P1 | 🟡 Partial |
| 10 | HTTPS enforcement | P0 | ✅ Via HF Spaces |
| 11 | Dependency scanning | P2 | ❌ No Dependabot |
| 12 | Container non-root user | P1 | ❌ Missing |

### 2.3 Device Authentication (Current → Target)

**Current:** `MD5(IP + UserAgent)[:16]` — trivially bypassable

**Target:**
```python
# First visit: generate secure device ID
device_id = secrets.token_hex(16)
# Set as HTTP-only, Secure, SameSite=Strict cookie
response.set_cookie("eatlytic_device", device_id, httponly=True, secure=True, samesite="strict", max_age=365*86400)

# Subsequent visits: read from cookie
device_key = request.cookies.get("eatlytic_device")
```

---

## 3. USER AUTHENTICATION BLUEPRINT

### Phase 1: Device-Key Auth (Now)
- No friction — scan immediately on first visit
- Device cookie tracks quota
- Sufficient for 0-500 users

### Phase 2: Supabase Auth (500+ Users)
- Email/password + OAuth (Google, Apple)
- Cross-device sync
- Required for real Pro subscriptions
- JWT-based, row-level security

### Phase 3: Phone OTP (India Scale)
- SMS OTP via Twilio/MSG91
- Most Indians prefer phone auth
- Required for WhatsApp cross-device

---

## 4. PAYMENT FLOW BLUEPRINT

### 4.1 Razorpay Flow

```
User clicks "Upgrade to Pro"
  → Frontend opens Razorpay Checkout (₹49/month or ₹299/year)
  → User completes payment
  → Razorpay returns: order_id + payment_id + signature
  → Frontend POSTs to /activate-pro
  → Backend verifies HMAC-SHA256 signature
  → Backend sets is_pro=1, pro_expires=+30 days
  → Frontend shows success, reloads
```

### 4.2 Pricing Tiers

| Tier | Price | Features |
|------|-------|----------|
| Free | ₹0 | 10 scans/month, basic analysis |
| Pro Monthly | ₹49 | Unlimited scans, PDF export, full history |
| Pro Yearly | ₹299 | Same + 49% savings |
| Family Yearly | ₹499 | 5 profiles, child safety alerts |

### 4.3 B2B Pricing

| Tier | Price | Quota |
|------|-------|-------|
| Business | ₹5,000/month | 1,000 API calls |
| Enterprise | Custom | Unlimited, SLA, on-prem |

---

## 5. FRONTEND BLUEPRINT

### 5.1 Current State
- `index.html` — single landing page
- `app.js` — 62KB vanilla JS (monolithic)
- `style.css` — 56KB (single file)

### 5.2 Immediate Fixes (No Rewrite)
1. **Error boundary** — show user-friendly error on server 500
2. **Loading states** — spinner + disabled button during scan
3. **Razorpay integration** — implement `activatePro()` function
4. **Scan status display** — "X scans remaining" badge
5. **Image preview** — show uploaded image before scan

### 5.3 Future: PWA (Month 3-6)
- Installable (add to home screen)
- Camera access (native-like)
- Offline queue (scan later when online)
- Push notifications (scan reminders)
- Recommended stack: React + Vite + PWA manifest

---

## 6. API & BACKEND BLUEPRINT

### 6.1 Route Structure (Target)

```
main.py (entry point only — imports routes)
├── app/routes/web.py          # /, /analyze, /check-image, /enhance-preview
├── app/routes/api_b2b.py      # /api/v1/analyze, /admin/create-api-key
├── app/routes/payments.py     # /activate-pro, /restore-pro, /scan-status
├── app/routes/whatsapp.py     # /whatsapp-webhook
├── app/routes/export.py       # /export-pdf, /generate-share-card
├── app/routes/admin.py        # /admin/unverified, /admin/correct
└── app/routes/compliance.py   # /api/v1/user/delete, /api/v1/history
```

### 6.2 Key Endpoints

| Method | Endpoint | Rate Limit | Auth |
|--------|----------|------------|------|
| POST | /analyze | 15/min | Device cookie |
| POST | /check-image | 30/min | None |
| POST | /enhance-preview | 20/min | None |
| POST | /api/v1/analyze | 60/min | API Key |
| POST | /activate-pro | 5/min | Device + Razorpay sig |
| GET | /scan-status | 30/min | Device cookie |
| DELETE | /api/v1/user/delete | 5/min | Device cookie |
| GET | /api/v1/history | 20/min | Device cookie |
| POST | /export-pdf | 10/min | Device cookie |
| POST | /whatsapp-webhook | N/A | Twilio signature |

### 6.3 LLM Module Split (Target)

```
llm.py (51KB) → Split into:
├── prompt_builder.py    # Super-prompt construction, persona templates
├── analyzer.py          # LLM call orchestration, retry logic, vision fallback
├── validator.py         # Atwater physics, DNA overrides, label filter
└── llm_client.py        # Groq/Together client wrappers, token counting
```

---

## 7. TRUST & COMPLIANCE BLUEPRINT

### 7.1 Trust Signals to Add

| Signal | Priority | Status |
|--------|----------|--------|
| Privacy Policy page (`/privacy`) | P0 | ❌ Missing |
| About / Team page (`/about`) | P0 | ❌ Missing |
| Terms of Service (`/terms`) | P0 | ❌ Missing |
| Medical disclaimer | P0 | ✅ Done |
| DPDP data deletion | P0 | ✅ Done |
| Accuracy confidence badge | P1 | ❌ Missing |
| FSSAI standards reference | P1 | ❌ Missing |
| Live scan counter | P1 | ❌ Missing |
| Open-source scoring methodology | P2 | ❌ Missing |

### 7.2 DPDP Compliance

| Requirement | Status |
|-------------|--------|
| Right to erasure | ✅ `DELETE /api/v1/user/delete` |
| Data minimization | ✅ Only device_key, no PII |
| Retention limits | ✅ 90-day purge |
| Purpose limitation | ✅ Nutrition analysis only |
| Consent mechanism | ❌ Need cookie consent banner |
| Grievance officer | ❌ Need contact on /privacy |
| Data breach notification | ❌ Need policy on /privacy |

---

## 8. GROWTH & DISTRIBUTION BLUEPRINT

### 8.1 Channels

| Channel | Status | Strategy |
|---------|--------|----------|
| Web app | ✅ Live | SEO for "food label scanner India" |
| WhatsApp | ✅ Built | Beta groups → viral sharing |
| Instagram/Reels | ❌ | "We caught [brand] lying" videos |
| YouTube | ❌ | Label accuracy expose series |
| Reddit/Quora | ❌ | Organic nutrition answers |
| Gym partnerships | ❌ | QR codes at gyms |
| Nutritionist referrals | ❌ | Referral program |

### 8.2 Viral Content Ideas

1. **"Label Accuracy Score"** — Scan popular Indian brands, show when calorie math is off
2. **"Hidden Sugar"** — Products claiming "no added sugar" but containing maltodextrin
3. **"NOVA-4 Detector"** — Ultra-processed ingredients in "healthy" products
4. **"Mom Test"** — Mothers scanning kids' snacks, reactions to hidden ingredients

### 8.3 WhatsApp Growth Loop

```
User scans via WhatsApp → Gets result + "Share with friends"
  → Forwards to family group → Family members try it → Viral loop
```

---

## 9. TECHNICAL DEBT ROADMAP

### 9.1 Refactoring Priority

| Task | Effort | Impact |
|------|--------|--------|
| Split `llm.py` (51KB → 4 files) | 2 days | Engineer onboarding |
| Split `main.py` (42KB → 7 route files) | 2 days | Maintainability |
| Add type hints everywhere | 1 day | IDE support |
| Add integration tests | 2 days | Regression prevention |
| Set up mypy + pylint | 2 hours | Static analysis |
| Add load testing (locust) | 1 day | Performance baseline |

### 9.2 Infrastructure Evolution

```
Month 0-3:  HuggingFace Spaces (free) → SQLite → Single worker
Month 3-6:  VPS ($50/mo) → Supabase (free tier) → 2-3 workers → CDN
Month 6-12: Kubernetes → Supabase Pro → Redis → S3 → Load balancer → Monitoring
```

---

## 10. 90-DAY EXECUTION TIMELINE

### Phase 1: SECURE & TRUST (Days 1-14)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Rotate ALL API keys + scrub git history | Clean repo |
| 2 | Fix CORS wildcard, disable /restore-pro | Secure endpoints |
| 3-4 | Create /privacy, /about, /terms pages | Trust signals live |
| 5-6 | Add footer links to all pages | Navigation complete |
| 7 | Set up GitHub Actions CI (pytest + lint) | Green badge |
| 8-10 | Complete Razorpay frontend flow | Working payments |
| 11-12 | Harden device fingerprint (cookie-based) | Secure identity |
| 13-14 | Bug fixes + stability pass | Clean build |

**✅ Success: No secrets in git, all P0 security closed, payments working, CI passing**

### Phase 2: PROVE DEMAND (Days 15-45)

| Day | Task | Deliverable |
|-----|------|-------------|
| 15-17 | Recruit 20 beta users | Active user list |
| 18-20 | Add analytics (Google Analytics / PostHog) | Tracking live |
| 21-25 | Record 10 user testing sessions | Bug list from real usage |
| 26-30 | Fix top UX blockers | Improved flow |
| 31-35 | Create 5 viral content pieces | Social media posts |
| 36-40 | WhatsApp beta group (20 users) | Active group |
| 41-45 | Measure retention + "Eatlytic Effect" | Data report |

**✅ Success: 100 users, 20 WAU, Day-7 retention >20%, 5 "Eatlytic Effect" stories**

### Phase 3: INVESTMENT READY (Days 46-90)

| Day | Task | Deliverable |
|-----|------|-------------|
| 46-55 | Refactor llm.py + main.py | Clean modular code |
| 56-60 | Analytics dashboard | Metrics visibility |
| 61-70 | India growth push (gyms, parents) | 500+ users |
| 71-75 | Polish UI + trust signals | Professional look |
| 76-80 | Prepare pitch deck (10 slides) | Investor-ready |
| 81-85 | Investor outreach (20 pitches) | Conversations booked |
| 86-90 | Hit 1,000 MAU milestone | Investment-ready |

**✅ Success: 1,000 MAU, >1% Pro conversion, pitch deck ready, 5 investor calls booked**

---

## 11. INVESTMENT READINESS CHECKLIST

### Metrics That Matter

| Metric | Current | Day 90 Target | Investor Threshold |
|--------|---------|---------------|-------------------|
| MAU | 0 | 1,000 | 500+ |
| WoW growth | N/A | 15%+ | 10%+ |
| Day-7 retention | N/A | >20% | >15% |
| Pro conversion | N/A | >1% | >0.5% |
| Scans/user/week | N/A | >2 | >1 |
| "Eatlytic Effect" | N/A | >30% | >20% |
| Uptime | Unknown | 99%+ | 99%+ |

### Pitch Deck Structure

1. **Problem:** Brands lie about nutrition. Consumers can't verify.
2. **Solution:** Universal label scanner with physics validation
3. **Market:** $10B+ Indian health-tech by 2033, 1.4B people
4. **Product:** Live demo (scan a Maggi packet)
5. **Traction:** MAU, retention, growth rate graphs
6. **Business Model:** Freemium + B2B API
7. **Competition:** No one validates label physics
8. **Moat:** Atwater engine, 18-language OCR, NOVA-4 detection
9. **Team:** Founder + future hires
10. **Ask:** $X for Y months to hit Z metrics

### Due Diligence Prep

| Document | Status |
|----------|--------|
| Incorporation (MCA) | ❌ File |
| Cap table | ❌ Create |
| Financial projections | 🟡 In blueprint |
| Technical architecture | ✅ This document |
| Security audit | 🟡 After fixes |
| User research videos | ❌ Record in Phase 2 |
| Trademark (Eatlytic) | ❌ File with IPO |
| Patent (Atwater method) | ❌ Consult IP lawyer |

---

## APPENDIX A: TARGET FILE STRUCTURE

```
Eatlytic-App/
├── .github/workflows/ci.yml
├── app/
│   ├── routes/ (web, api_b2b, payments, whatsapp, export, admin, compliance)
│   ├── services/ (16 modules — already modular)
│   └── models/db.py
├── static/ (index.html, about.html, privacy.html, app.js, style.css)
├── tests/ (test_critical, test_phash, test_poison_pill, test_safety_companion)
├── scripts/ (flush_cache, scrub_meat, inspect_db, benchmark)
├── main.py, Dockerfile, docker-compose.yml, requirements.txt
├── .env.example, .gitignore, README.md, BLUEPRINT.md
```

## APPENDIX B: ENVIRONMENT VARIABLES

```bash
# Required
GROQ_API_KEY=gsk_...                    # AI analysis
TOGETHER_API_KEY=tgp_v1_...             # Fallback LLM
ADMIN_TOKEN=...                         # Admin endpoints (64+ chars)

# Required for payments
RAZORPAY_KEY_ID=...
RAZORPAY_KEY_SECRET=...

# Optional
TWILIO_ACCOUNT_SID=AC...               # WhatsApp
TWILIO_AUTH_TOKEN=...                   # WhatsApp
FREE_SCAN_LIMIT=10                      # Default scan quota
ENV=production                          # Environment flag
```

---

> **This is a living document. Update weekly as you ship, learn, and iterate.**
>
> *The tech is built. The blueprint is drawn. Now go find the humans.*
