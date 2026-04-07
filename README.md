---
title: Eatlytic
emoji: 🥗
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Eatlytic

Eatlytic is a high-precision **AI-powered food analysis engine** designed to decode complex Indian nutrition labels and marketing claims instantly via WhatsApp.

### 🌟 Core Mission
Traditional food labels are designed to confuse. Eatlytic uses advanced computer vision and a proprietary rule-based engine to strip away marketing "lies" and provide a clear, **1-10 health score** based on a product's actual ingredients.

### 🚀 Key Features
1. **Product DNA Classification:**
   * **NOVA 1-4 Engine:** Automatically flags "Ultra-Processed Foods" (UPF).
   * **Marketing Lie Detector:** Catches fake claims (e.g., flags "Sugar-Free" if it contains hidden Maltodextrin).
2. **Advanced Vision Pipeline:**
   * **AI Enhancement:** Auto-deblurs and denoises poor-quality photos using Wiener Deconvolution and CLAHE.
   * **Resolution Awareness:** Blocks tiny thumbnails to ensure 0% hallucinations.
3. **Indian-Language OCR:** Supports English, Hindi, and Tamil with automatic script fallback.
4. **Persona-Based Scoring:** Tailors advice for specific Indian health profiles (e.g., **Diabetic Care, Pregnancy Safe, Child Safety, Senior Citizens, Athletes**).

### 💻 Technical Stack
* **Engine:** Python (FastAPI)
* **AI & Vision:** Llama 3 via Groq, EasyOCR, OpenCV
* **Infrastructure:** Self-Hosted VPS (Optimized for zero-cold-boot, instant responses), Docker.

### 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shaikhsarib/Eatlytic-App.git
   cd Eatlytic-App
   ```

2. **Configure Environment:**
   Create a `.env` file with your `GROQ_API_KEY`.

3. **Run with Docker:**
   ```bash
   docker-compose up --build -d
   ```

## License
Proprietary. All rights reserved.
