# Eatlytic — Ultra-Reliable Food Label Intelligence

Eatlytic is a high-performance, AI-powered nutritional analysis platform designed to transform food label photos into verifiable health intelligence. It uses a custom **Dual-Pass OCR Engine**, **Atwater Physics Validation**, and **LLM-driven Research** to provide accurate analysis even from low-resolution or damaged packaging.

---

## 🚀 Core Technology Stack
- **Backend**: FastAPI (Python 3.11)
- **OCR Engine**: EasyOCR + OpenCV (Adaptive Dual-Pass)
- **AI Brain**: Groq (Llama-3-70B)
- **Search Engine**: DuckDuckGo Research (Live Fallback)
- **Phyics Engine**: Custom Atwater Math Validator
- **Deployment**: Hugging Face Spaces (Docker)

---

## 🎨 System Architecture & Flow

### **1. The Analysis Pipeline**
1. **Entry Point** (`main.py`): Receives image buffer and device metadata.
2. **Duplicate Detection** (`hash_service.py`): Uses **Perceptual Hashing (pHash)** to check if the image has been analyzed before. Discards 0-nutrient "poisoned" caches automatically.
3. **Visual Hardening** (`label_detector.py`): Crops the Region of Interest (ROI), enhances contrast, and performs initial AI-based label detection.
4. **Dual-Pass OCR** (`ocr.py`):
    - **Pass 1 (Natural)**: Upscaling + CLAHE + Sharpening (Best for clear color labels).
    - **Pass 2 (Adaptive)**: Adaptive Gaussian Thresholding (Fallback for blurry or low-contrast text).
5. **Universal Label Filter** (`ocr.py`): Custom "Trash Compactor" that strips out Indian legal text (FSSAI, MRP, License Nos) and focuses purely on the nutrition table.
6. **AI Analysis** (`llm.py`):
    - **V4 Categorization Guardrails**: Brand-aware logic (e.g., Cadbury is always Sweet/Dairy, never Meat).
    - **Live Research Fallback** (`research_engine.py`): Queries the web for verified nutrition facts if the OCR is compromised.
7. **Atwater Physics Check** (`fake_detector.py`): A physics-based verification layer that checks if `Protein + Carb + Fat` math matches the `Total Calories` stated on the label.
8. **Final Rendering**: Returns a structured JSON displayed as a premium Health Card.

---

## 📂 Project Structure

```text
Eatlytic-App/
├── main.py                     # FastAPI Application Root & Routes
├── flush_cache.py              # Maintenance: Purge failure/poisoned caches
├── inspect_db.py              # Maintenance: Inspect local database content
├── scrub_meat.py               # Maintenance: Specific classification purge
├── index.html                  # Frontend: Web interface entry point
├── requirements.txt           # Dependencies
├── Eatlytic-12Week-Roadmap.md # Long-term project strategy
├── app/
│   ├── models/
│   │   ├── db.py               # Database connections (SQLite/Supabase)
│   │   └── __init__.py
│   ├── routes/
│   │   ├── auth.py             # User authentication routes
│   │   ├── benchmarks.py       # Performance testing endpoints
│   │   ├── food_db.py          # Analytics and scanning history
│   │   ├── payments.py         # Subscription & Billing routes
│   │   └── __init__.py
│   └── services/
│       ├── ocr.py              # Dual-Pass OCR Engine (Natural vs Adaptive)
│       ├── llm.py              # AI Brain & Categorization Guardrails
│       ├── fake_detector.py    # Atwater Physics Math Validator
│       ├── label_detector.py   # Visual Image ROI & Contrast Enhancement
│       ├── hash_service.py     # Perceptual Hashing (pHash) Deduplication
│       ├── research_engine.py  # DuckDuckGo Targeted Web Research
│       ├── explanation_engine.py # Health Card narrative generator
│       ├── formatter.py        # Post-processing nutrient formatting
│       ├── duel_service.py     # Competitive product comparison logic
│       ├── alternatives.py      # Healthier product recommendation engine
│       ├── image.py            # Image compression & handling
│       ├── auth.py             # Authentication backend logic
│       ├── payments.py         # Stripe/Razorpay integration service
│       └── __init__.py
├── tests/ (Scripts)
│   ├── test_critical.py        # Core stability tests
│   ├── test_phash.py           # Duplicate detection tests
│   └── test_poison_pill.py    # Resilience tests for bad inputs
```

---

## 🛠️ Local Setup & Maintenance

### **Running Locally**
1. Install requirements: `pip install -r requirements.txt`
2. Set `GROQ_API_KEY` in `.env`
3. Run the server: `uvicorn main:app --reload --port 8000`

### **Maintenance Utilities**
- **Fresh Start**: To wipe all local cache/history: `Remove-Item data/eatlytic.db`
- **Partial Purge**: To remove specific classification errors: `python scrub_meat.py`
- **Cache Clean**: To remove "failed" or "unreliable" entries: `python flush_cache.py`

---

## 🔒 Security & Reliability
- **Safety Valve**: The system automatically detects and bypasses "poisoned" cache entries (results with 0 macro-nutrients).
- **Tier-Based Confidence**: Results are tagged as **HIGH**, **MEDIUM**, **LOW**, or **UNRELIABLE** based on OCR word count and physics verification.
- **Brand Guardrails**: Hardcoded logic prevents hallucinations for common global brands (Cadbury, Amul, Nestle, Britannia, etc.).

---

Developed with ❤️ by the Vartistic Studio Team.
