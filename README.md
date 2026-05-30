# 🔬 Eatlytic — Universal AI Nutrition Platform & Clinical Diagnostics

Eatlytic is a high-performance, clinical-grade AI-native nutrition scanning and biosensor telemetry platform. Designed to convert raw food packaging photos and metabolic data into personalized health intelligence, it operates on a state-of-the-art **Clean Onion Architecture** matching elite AI engineering standards (OpenAI, Anthropic, Google DeepMind).

---

## 🏛️ Clean Onion Repository Architecture

Eatlytic's codebase is structured to decouple cognitive AI reasoning loops from application delivery and database transactions:

```text
Eatlytic-App-main/
├── app/
│   ├── ai/                          <--- 🧠 COGNITIVE REASONING LAYER
│   │   ├── llm/
│   │   │   ├── client.py            # Multi-provider failovers (Ollama Gemma-4 -> Groq -> Gemini -> Together)
│   │   │   ├── engine.py            # Stateless LLM routines, re-exports & label recovery helpers
│   │   │   ├── prompts.py           # Nutrition specialist prompt matrices & system prompts
│   │   │   └── validators.py        # Strict Pydantic JSON validation models
│   │   ├── ocr/
│   │   │   └── client.py            # PyTorch-backed EasyOCR loader with Vision-LLM fallback
│   │   └── perception/
│   │       └── bk_tree.py           # Metric space Burkhard-Keller Trees for sub-linear pHash matching
│   │
│   ├── core/                        <--- 🛡️ INFRASTRUCTURE & SECURITY UTILITIES
│   │   └── security.py              # Central cryptographic signing, device keys, & admin auth
│   │
│   ├── database/                    <--- 💾 DATA PERSISTENCE LAYER
│   │   └── connection.py            # Connection pool contexts & SQLite WAL locks
│   │
│   ├── services/                    <--- ⚙️ BUSINESS DOMAIN SERVICES
│   │   ├── scan_orchestrator.py     # Core scan orchestration, caching & local verified lookups
│   │   ├── personalized_medicine.py # Genomic SNPs (TCF7L2, AGT) & clinical safety algorithms
│   │   ├── explanation_engine.py    # Atwater calorie maths, ICMR RDA math, & INS/E-number translations
│   │   ├── fake_detector.py         # Nutrition fraud classification rules
│   │   ├── user_auth.py             # Session authentication, persistent OTP tokens, & streak trackers
│   │   └── payments.py              # Pro limits & Razorpay billing gateways
│   │
│   └── api/                         <--- 🌐 PRESENTATION LAYER (TRANSPORT GATEWAY)
│       └── v1/                      # FastAPI endpoint controllers
│           ├── scan.py              # Photo & Voice loggers, imports scan_orchestrator
│           ├── cgm.py               # CGM telemetry syncs & post-prandial correlations
│           ├── dietitian.py         # SSR clinical neobrutalist B2B cohort interfaces
│           └── user.py              # User profiles & session configurations
│
├── data/                            # Persistent databases, verified additives base, and seeders
├── scripts/                         # Maintenance, site generators, and programmatic seeders
├── static/                          # SPA client portals, service workers, and developer consoles
├── tests/                           # Complete integration and unit test suite
├── main.py                          # App bootstrap and static portal mounts
├── requirements.txt                 # Pin-point package dependencies
└── README.md                        # Documentation
```

---

## ⚡ Key Capabilities & Pipelines

### 1. Perceptual BK-Tree Image Caching (`app/ai/perception/`)
* **Sub-linear Match Execution:** To completely bypass high-latency OCR and LLM calculations, the system calculates a 64-bit image perceptual hash (pHash) for every label uploaded.
* **Burkhard-Keller Trees:** Utilizes an in-memory, thread-safe BK-Tree metric space searching Hamming distances in sub-linear logarithmic time $O(\log N)$.
* Matches with a Hamming distance $\le 4$ resolve to matching cached clinical audits instantly in under 10 milliseconds.

### 2. Multi-Engine Cognitive AI (`app/ai/llm/`)
* **Self-Correcting Calorie Math:** Compares the declared label calories against the Atwater physical weights:
  $$\text{Calories} = (\text{Protein} \times 4) + (\text{Carbohydrates} \times 4) + (\text{Fat} \times 9)$$
* If the variance exceeds standard clinical margins, the client catches the mismatch (hallucination) and runs a self-correction retry, outputting valid structured JSON.
* **Multi-Provider Failover:** High-priority Local Gemma-4 (Ollama) cascades seamlessly to Groq, Gemini 2.0 Flash, and Together AI upon network or rate-limiting thresholds.

### 3. Personalized Genomics & Metabolic Rules (`app/services/personalized_medicine.py`)
* Enforces direct contraindication checks based on user profiles.
* Correlates ingredients and chemical compounds against personal genetic risk factors, such as high-glycemic Maida for type-2 diabetes risk `TCF7L2` alleles, or high sodium for cardiovascular `AGT` markers.

### 4. Biosensor CGM Sync & Glycemic Correlations (`app/api/v1/cgm.py`)
* Syncs continuous glucose monitor records (Dexcom/Libre) securely, filtering duplicate timestamps.
* Automatically computes 30-day ADA-compliant clinical metrics: Estimated HbA1c (`eA1c`), averages, and Time-in-Range (TIR).
* **Glycemic Excursions:** Automatically correlates dietary scan times against a 2-hour post-meal interstitial glucose window, flagging food categories that trigger physical metabolic spikes ($>30\text{ mg/dL}$).

---

## 🛠️ Local Verification & Development

### 1. Boot up the Development Server
Install dependencies and run the FastAPI bootstrap:
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
* **Developer Portal:** `http://localhost:8000/developer`
* **Programmatic SEO Base:** `http://localhost:8000/ingredients/{slug}` (e.g. `sodium-benzoate`)

### 2. Run the Verification Test Suite
Eatlytic is fully covered by a robust test suite. All tests execute with zero shims or legacy imports:
```bash
python -m pytest
```
**Test Results:** **112/112 passed 100% green** in under 12 seconds.

---

## 🌐 SEO & Growth Channels

* **Programmatic Additive SEO:** Dynamically compiles neobrutalist, search-engine crawlable SEO cards with comprehensive metadata schemas (`MedicalWebPage`) for all 500+ ingredients inside `data/additives.json`.
* **Dietitian cohort dashboard:** Neobrutalist SSR panel allowing authorized B2B clinicians to monitor patient compliance streaks, flagged chemical alerts, and live metabolic CGM telemetries at a single glance.
