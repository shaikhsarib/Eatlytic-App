# Eatlytic — The Ultimate Food Label Intelligence Platform

Eatlytic is a high-performance, AI-native nutritional analysis engine designed to transform blurry packaging photos into verifiable health intelligence. It is specifically optimized for the Indian market, featuring **Dual-Pass OCR**, **Atwater Physics Validation**, and **Cultural Benchmarking**.

---

## 🚀 The Eatlytic Intelligence Pipeline

The system processes images through a sophisticated multi-stage pipeline:

### **1. Visual Hardening & Detection**
- **Deduplication**: Uses Perceptual Hashing (pHash) in `hash_service.py` to identify returning images and serve cached results instantly, skipping expensive OCR and AI calls.
- **ROI Targeting**: `label_detector.py` automatically crops the the nutrition table (Region of Interest) and performs contrast-limiting adaptive histogram equalization (CLAHE) to make text stand out.

### **2. Dual-Pass OCR Engine (`ocr.py`)**
To handle high-resolution photos and blurry WhatsApp thumbnails, the engine uses two scanning modes:
- **Pass 1 (Natural Enhancement)**: Preserves gradients and antialiasing, perfect for high-quality camera photos.
- **Pass 2 (Adaptive Bitonal)**: Converts the image to a Gaussian adaptive black-and-white mask. This "cuts through" background noise on low-resolution or dark-light packaging (e.g., light text on dark purple wrappers).

### **3. The Analysis Brain (`llm.py`)**
- **Categorization Guardrails (V4)**: Hardcoded logical gates ensure that common brands (Cadbury, Amul, Nestlé) are never misidentified. 
- **Live Search Fallback**: If the OCR is messy, `research_engine.py` performs a targeted web search for the specific brand's verified nutrition table.

### **4. Verification & Insights**
- **Atwater Physics Validator (`fake_detector.py`)**: Checks the physics of the label. If `(Protein * 4) + (Fat * 9) + (Carb * 4)` does not match the stated Calories, the system flags the label as unreliable.
- **ICMR 2020 RDA (`explanation_engine.py`)**: Benchmarks all data against Indian Adult RDA standards.
- **Cultural Equivalents**: Converts abstract calories into relatable Indian foods (e.g., "This biscuit is equivalent to 1.5 Samosas in calories").

---

## 📂 Exact File Structure

```text
Eatlytic-App/
├── main.py                     # API Entry Point (FastAPI)
├── app/
│   ├── models/
│   │   ├── db.py               # Cache & Persistence (SQLite/Supabase)
│   ├── routes/
│   │   ├── food_db.py          # Scan History & Analytics
│   │   ├── auth.py             # User Management & Key Rotation
│   │   └── payments.py         # Monetization & Pro Quotas
│   └── services/
│       ├── ocr.py              # Dual-Pass OCR Logic
│       ├── llm.py              # AI Extractor & Brand Guardrails
│       ├── fake_detector.py    # Atwater Math Physics Engine
│       ├── duel_service.py     # Product Comparison Logic (Persona-weighted)
│       ├── alternatives.py      # Healthy Swap Matrix (Indian Context)
│       ├── explanation_engine.py # ICMR RDA & Cultural Benchmarking
│       ├── label_detector.py   # Computer Vision Pre-processing
│       └── research_engine.py  # DuckDuckGo Targeted Research
├── maintenance/
│   ├── flush_cache.py          # Admin Tool: Clear failed scans
│   ├── scrub_meat.py           # Safety Tool: Purge categorization errors
│   └── inspect_db.py           # Debug Tool: View live cache entries
```

---

## ⚡ Key Specialty Features

### **The Duel Engine**
Located in `duel_service.py`, this module allows head-to-head product comparisons. Unlike generic score comparison, it uses a **Persona-Weighted Matrix**:
- **Muscle Mode**: Favors protein density.
- **Diabetic Mode**: Penalizes sugar and refined carbs (Maida) with a -5.0 multiplier.
- **Weight Loss Mode**: Strongly penalizes calorie-to-satiety ratios.

### **Product Alternatives**
The `alternatives.py` module uses an **Ingredient-Pivot** matrix specifically for the Indian diet, recommending local healthy swaps like roasted Makhana (fox nuts) for chips, or Poha/Sevai for instant noodles.

### **INS Additive Scanner**
`explanation_engine.py` contains a built-in database of INS/E numbers (e.g., 621, 150d) that scans the ingredients list to warn users about flavor enhancers, synthetic colors, and preservatives.

---

## 🛠️ Performance & Scalability
- **Quota Logic**: Built-in scan tracking in `main.py` prevents API abuse.
- **Cache Safety Valve**: Automatically discards any cached results with 0 macronutrients to ensure users never see "dead" data.
- **Docker-Ready**: Optimized `Dockerfile` for deployment on Hugging Face Spaces with specialized memory management for EasyOCR.

---

*Built with ❤️ for a Healthier India.*
