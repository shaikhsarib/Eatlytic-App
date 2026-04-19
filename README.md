# Eatlytic — Universal Nutrition Intelligence Platform

Eatlytic is a high-performance, AI-native nutritional analysis engine designed to transform blurry packaging photos into verifiable health intelligence. It is a **Universal engine** capable of processing food labels from any country, language, or format using advanced computer vision and script-aware OCR.

---

## 🚀 The Eatlytic Intelligence Pipeline

The system processes images through a sophisticated, multi-pass pipeline:

### **1. Visual Hardening & Detection**
- **Deduplication**: Uses Perceptual Hashing (pHash) in `hash_service.py` to identify returning images and serve cached results instantly, skipping expensive OCR and AI calls.
- **MSER ROI Targeting**: `label_detector.py` uses Maximally Stable Extremal Regions to detect text density heatmaps. This allows the system to find nutrition tables on any material (shiny, dark, or transparent) without relying on hardcoded color panels.

### **2. "Never Reject" Blur Repair (`image.py`)**
 Eatlytic is designed to read "real-world" photos, including blurry WhatsApp thumbnails:
- **Repair Pipeline**: Triggers a 4-stage repair sequence for blurry images: **Lanczos4 Upscale (1800px)** → **Motion Deconvolution** → **Local Contrast (CLAHE)** → **Unsharp Masking**.
- **Result**: Drastically improves success rates for compressed or low-light photos that standard OCR would reject.

### **3. Global script-aware OCR (`ocr.py`)**
- **Auto-Language Detection**: Identifies the script (Arabic, CJK, Hindi, European) from the image center before running full OCR.
- **Multilingual Support**: Expanded to support **18+ languages** including Thai, Korean, Japanese, Russian, and Arabic.
- **Multi-Pass Retry**: If confidence is low, the system automatically retries with dedicated Denoise, Sharpen, and Binary processing passes.

### **4. Universal Analysis Brain (`llm.py`)**
- **Global Extraction**: A universalized "Super-Prompt" instructs the AI to act as a global nutrition specialist, extracting 15+ rich data fields (Molecular Insight, ELI5, Age Warnings) regardless of regional naming conventions.
- **Research Fallback**: If OCR is messy, `research_engine.py` performs a targeted web search for verified manufacturer data.

### **5. Physics & Validation**
- **Universal Atwater Gate (`fake_detector.py`)**: Checks the physics of the label. If `(Protein * 4) + (Fat * 9) + (Carb * 4)` does not match the stated Calories (within a 20% universal tolerance), the system flags the data as unreliable.

---

## 📂 Exact File Structure

```text
Eatlytic-App-main/
├── app/
│   ├── models/
│   │   └── db.py                    # SQLite & Persistence core
│   └── services/
│       ├── ocr.py                   # Multi-Pass Global OCR Engine
│       ├── llm.py                   # Universal AI Brain
│       ├── fake_detector.py         # Universal Atwater Physics Validator
│       ├── label_detector.py        # MSER Vision & ROI Targeting
│       ├── image.py                 # "Never Reject" Blur Repair Pipeline
│       ├── duel_service.py          # Persona-Weighted Comparison
│       ├── alternatives.py           # Global Healthy Swap Matrix
│       ├── hash_service.py          # Perceptual Hashing (pHash)
│       ├── research_engine.py       # Live Web Research (DDG)
│       ├── formatter.py             # Post-processing & WhatsApp Tiers
│       ├── explanation_engine.py      # ICMR/Global RDA Benchmarking
│       ├── auth.py                  # API Authentication
│       └── payments.py              # Quota & Payment logic
├── main.py                          # FastAPI Production Endpoint
├── index.html                       # Web Front-end
├── test_critical.py                 # Core Stability Testing
├── test_phash.py                   # Deduplication Testing
├── test_poison_pill.py             # Security & Resilience Testing
├── flush_cache.py                   # Maintenance: Cache Clearing
├── inspect_db.py                    # Maintenance: DB Inspection
├── scrub_meat.py                    # Maintenance: Categorization Repair
├── Dockerfile                       # Production Containerization
├── docker-compose.yml               # Local Orchestration
├── requirements.txt                 # System Dependencies
├── .env                             # Environment Config
└── Eatlytic-12Week-Roadmap.md      # Strategic Growth Plan
```

---

## ⚡ Key Specialty Features

### **The Duel Engine**
Located in `duel_service.py`, this module allows head-to-head product comparisons. Unlike generic score comparison, it uses a **Persona-Weighted Matrix** (Muscle Mode, Diabetic Mode, Weight Loss Mode).

### **Universal Product Alternatives**
The `alternatives.py` module uses an **Ingredient-Pivot** matrix that recommends healthy swaps locally and globally (e.g., roasted Makhana for chips, or Poha for instant noodles).

---

## 🛠️ Performance & Scalability
- **Quota Logic**: Built-in scan tracking in `main.py` prevents API abuse.
- **Cache Safety Valve**: Automatically discards results with 0 macronutrients.
- **HuggingFace Ready**: Memory-optimized deployment for HF Spaces.

---

*Built for Global Nutrition Intelligence.*
