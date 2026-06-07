# The Complete Eatlytic Project Summary (Made Simple)

This guide summarizes **everything** about the Eatlytic codebase and system architecture. It is written in simple, plain English so you can explain it to your team tomorrow with ease.

---

## 1. What is Eatlytic? (The 1-Minute Pitch)
> *"Eatlytic is a **Personal Nutrition Intelligence Layer (PNIL)**. It is not just another calorie-tracking app. It is a system that scans packaged food labels, validates the nutritional claims using physical laws (thermodynamics), decodes chemical additives, and checks if the food is clinically safe for individual health conditions (like diabetes or hypertension) or genetic profiles. It runs local, offline rule engines first to avoid LLM cloud costs, falling back to AI and web search only when necessary."*

---

## 2. The Core Architecture (How the Modules Connect)

The codebase is organized into clean, modular layers (called Onion Architecture):

```
       [ Client / Frontend (App, WhatsApp, Dashboard) ]
                             │
                             ▼
     [ API Routes: app/api/v1/ (scan, dietitian, cgm, mcp) ]
                             │
                             ▼
  [ Services: app/services/ (brain, fake_detector, explanation) ]
                             │
                             ▼
     [ Database & Caching: app/database/ (SQLite / Supabase) ]
```

---

## 3. Explaining the 8 Core Modules to Your Team

Here is a simple breakdown of what each Python module in the codebase does:

### 1. The Local Brain (`brain.py`)
* **What it is:** The central decision-maker.
* **In Simple Words:** It takes the parsed nutrients and runs offline, rule-based medical logic. It grades the product on a scale of 1 to 10 and assigns a safety verdict (Safe, Limit, or Avoid) based on user health profiles (diabetic, hypertension) without making expensive AI calls.

### 2. The Claims Lie Detector (`fake_detector.py`)
* **What it is:** The audit engine.
* **In Simple Words:** It does two main things:
  1. **Atwater Physics Check:** Calorie numbers on packages are checked against thermodynamic laws. It calculates: `(protein * 4) + (carbs * 4) + (fat * 9)`. If this is different from the declared calories, it flags it as fraud.
  2. **Marketing Lie Detector:** If a product claims "No Added Sugar" but the ingredients list contains hidden sugars like *maltodextrin*, *dextrose*, or *date syrup*, it flags the brand for fake marketing.

### 3. The Scan Orchestrator (`scan_orchestrator.py`)
* **What it is:** The traffic cop.
* **In Simple Words:** It coordinates the entire scan pipeline: OCR text extraction -> checking the local database for existing products -> running web search context (RAG) -> LLM verification -> saving the scan history.

### 4. The Personalized Genomic Engine (`personalized_medicine.py`)
* **What it is:** The DNA matching layer.
* **In Simple Words:** If a user has genetic data (like SNP data), this module checks for:
  * **TCF7L2 gene** (diabetes risk).
  * **AGT gene** (sodium-sensitive high blood pressure).
  * **LCT gene** (lactose intolerance).
  It overrides the food score. For example, if a user is genetically lactose intolerant, dairy products are automatically flagged as "Avoid."

### 5. The CGM Spike Engine (`cgm.py`)
* **What it is:** Continuous Glucose Monitor (CGM) correlation.
* **In Simple Words:** It tracks glucose readings from biosensors. It calculates Time-in-Range (70–140 mg/dL) and estimated HbA1c. Most importantly, it matches a food scan with the user's blood sugar spike 2 hours later to prove if the food is a personal metabolic trigger.

### 6. The Additive Decoder (`additive_db.py`)
* **What it is:** The proprietary database lookup.
* **In Simple Words:** It loads a verified database (`additives.json`) containing chemical additives. It does instant, offline lookups for E-numbers (like E621 / MSG) or INS codes, and returns their FSSAI regulatory status and safety profiles.

### 7. The Dietitian Dashboard (`dietitian.py`)
* **What it is:** The B2B Portal.
* **In Simple Words:** A dashboard built for clinical dietitians to monitor all their patients' food scans, CGM spikes, and compliance reports in real-time, allowing them to send direct advice.

### 8. The MCP Server (`mcp.py`)
* **What it is:** Model Context Protocol.
* **In Simple Words:** It exposes Eatlytic's tools (like calculating food compatibility or looking up additives) so that other AI assistants (like ChatGPT or Claude) can query Eatlytic dynamically over standard interfaces.

---

## 4. How to Explain the Technical Flow to Your Team

When presenting, walk them through the lifecycle of a scan:

```
[ User takes a photo of a label ]
              │
              ▼
[ Extract text using OCR / Vision ]
              │
              ▼
[ Step 1: Database Check ] ─────────► (Product is verified in DB?) ──► YES ──► [Return Local Report (0ms, ₹0)]
              │ NO
              ▼
[ Step 2: Query formulation ] ──────► (Filter keywords + brand hints)
              │
              ▼
[ Step 3: Check L1/L2 Caches ] ─────► (Was search done recently?) ───► YES ──► [Inject cached context]
              │ NO
              ▼
[ Step 4: Web Search (RAG) ] ───────► (Quick DuckDuckGo text fetch)
              │
              ▼
[ Step 5: Atwater Physics Check ] ──► (Validate if protein/carbs/fat math adds up to calories)
              │
              ▼
[ Step 6: Persona & DNA check ] ────► (Calculate glycemic threat & check genetic warnings)
              │
              ▼
[ Save scan, cache result, output finished report to user ]
```

---

## 5. Our Current Challenges & Dev Roadmap

Be honest with your team about what needs to be built next:
1. **Grow the Database:** Currently, we only have 27 ingredients and 76 additives. We need to scrape the Codex Alimentarius and FSSAI public databases to reach 500+ items.
2. **Fix Scraper Fragility:** DuckDuckGo blocks scrapers. We must migrate from free DDG scraping to structured local vector search or paid search APIs.
3. **Launch the WhatsApp Channel:** The code is ready for WhatsApp integrations, but we need to connect Twilio and release it to get our first users.
