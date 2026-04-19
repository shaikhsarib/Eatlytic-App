# Eatlytic — Universal Nutrition Intelligence Platform

Eatlytic is a high-performance, AI-native nutritional analysis engine designed to transform blurry packaging photos into verifiable health intelligence. It is a **Universal engine** capable of processing food labels from any country, language, or format using advanced computer vision and script-aware OCR.

---

## 🚀 The Eatlytic "Grand Tour" Architecture

The system is organized into a modular, horizontally scalable architecture:

### **1. The Intelligence Services (`app/services/`)**
The "Brain" of Eatlytic, where raw pixels become data:
- **`ocr.py`**: The Multi-Pass Global OCR Engine. Features auto-script detection and 3-pass retry (Denoise/Sharpen/Binary).
- **`llm.py`**: The Universal AI Brain. Merges OCR text with global knowledge to produce high-fidelity JSON (Molecular Insight, ELI5, Age Warnings).
- **`label_detector.py`**: CV ROI targeting using **MSER (Maximally Stable Extremal Regions)** text-density heatmaps.
- **`image.py`**: The **"Never Reject"** repair pipeline (Upscale, Wiener deconvolution, CLAHE).
- **`fake_detector.py`**: The Atwater Physics Validator (Universal 20% tolerance floor).
- **`duel_service.py`**: Head-to-head persona-weighted product comparison logic.
- **`alternatives.py`**: Global healthy swap matrix (Ingredient-Pivot logic).
- **`hash_service.py`**: Perceptual Hashing (pHash) for instant deduplication.
- **`research_engine.py`**: Live Web Research (DDG) for messy-label fallback.
- **`explanation_engine.py`**: Global/ICMR RDA benchmarking and INS/E-number scanner.
- **`formatter.py`**: Result post-processing and text-tiering for WhatsApp/Web.

### **2. The API Layer (`app/routes/`)**
Handles user interaction, security, and business logic:
- **`auth.py`**: User authentication, session management, and Supabase security integration.
- **`benchmarks.py`**: Internal performance tracking (Latency, accuracy, and ROI stats).
- **`food_db.py`**: Analytics and Scan History (The backbone of the "History" tab).
- **`payments.py`**: Quota management and **Razorpay** integration for Pro activation.

### **3. The Persistence Layer (`app/models/`)**
The source of truth for the platform:
- **`db.py`**: Hybrid persistence engine. Uses **Supabase** for production clusters and **SQLite (WAL mode)** for local development and offline caching.

### **4. Maintenance & CLI Tooling (Root)**
Scripts for system upkeep and data repair:
- **`flush_cache.py`**: Clears broken or 0-nutrient cache entries.
- **`scrub_meat.py`**: Repairs categorization errors across the database.
- **`inspect_db.py`**: Terminal-based dashboard for viewing live scans and quotas.
- **`deploy.sh`**: Manual deployment script for server environments.

---

## 📂 Exhaustive File Structure

```text
Eatlytic-App-main/
├── .github/
│   └── workflows/
│       └── sync_to_huggingface.yml    # CI/CD: HF Spaces sync
├── app/
│   ├── models/
│   │   └── db.py                    # Persistence: Supabase/SQLite hybrid
│   ├── routes/
│   │   ├── auth.py                  # API: User Auth & Tokens
│   │   ├── benchmarks.py            # API: Performance Monitoring
│   │   ├── food_db.py               # API: History & Analytics
│   │   └── payments.py              # API: Razorpay & Quotas
│   └── services/
│       ├── ocr.py                   # Logic: Global OCR (18+ Scripts)
│       ├── llm.py                   # Logic: Universal AI Brain
│       ├── label_detector.py        # Vision: MSER ROI Targeting
│       ├── image.py                 # Vision: "Never Reject" Repair
│       ├── fake_detector.py         # Physics: Atwater Validation
│       ├── duel_service.py          # Feature: Persona-Weighted Duels
│       ├── alternatives.py           # Feature: Ingredient Swaps
│       ├── hash_service.py          # Performance: pHash Deduplication
│       ├── research_engine.py       # Fallback: Live Web Research
│       ├── formatter.py             # UX: Post-processing & Formatting
│       ├── explanation_engine.py    # Science: RDA & INS Scanning
│       ├── auth.py                  # Logic: Backend Security
│       └── payments.py              # Logic: Quota Verification
├── data/
│   ├── eatlytic.db                  # Local Persistence (fallback)
│   ├── ai_cache.json                # Local AI result cache 
│   └── ocr_cache.json               # Local OCR heatmap cache
├── main.py                          # Application Core (FastAPI)
├── index.html                       # Frontend Entry Point
├── test_critical.py                 # Stability: Pipeline stress tests
├── test_phash.py                   # Logic: Deduplication verification
├── test_poison_pill.py             # Security: Input resilience tests
├── conftest.py                      # Testing: Framework config
├── flush_cache.py                   # Maintenance: Cache Repair
├── inspect_db.py                    # Maintenance: DB Explorer
├── scrub_meat.py                    # Maintenance: Data Repair
├── Dockerfile                       # Infrastructure: Docker Image
├── docker-compose.yml               # Infrastructure: Local orchestration
├── requirements.txt                 # Dependencies: System packages
├── .env                             # Configuration: API Keys/URLs
└── Eatlytic-12Week-Roadmap.md      # Strategy: Future Growth
```

---

## ⚡ Performance & Compliance
- **DPDP Compliant**: Built-in data erasure (`/api/v1/user/delete`) and retention management in `db.py`.
- **HuggingFace Ready**: Auto-deploy via `.github/workflows` with tailored memory management for C-based vision libraries.
- **Cache Safety Valve**: Automatically discards suspect entries to ensure 100% data integrity.

---

*Built for Global Nutrition Intelligence.*
