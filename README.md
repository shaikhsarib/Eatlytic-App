# Eatlytic

Eatlytic is an AI-powered FastAPI application designed to analyze food nutritional labels. It uses Computer Vision and Large Language Models to check ingredients, determine nutritional value, and detect blurred out labels to provide accurate food quality assessments.

## Features

- **Food Label OCR**: Extracts text from food labels using EasyOCR with multi-language support (English, Hindi, Chinese, etc.).
- **Smart Image Deblurring**: Multi-method blur detection (Laplacian, Tenengrad, Brenner) combined with an enhancement pipeline (Wiener deconvolution, Unsharp masking, CLAHE) using OpenCV to recover text from poorly captured images.
- **AI Nutritional Analysis**: Leverages the Groq API (LLMs) to analyze ingredients, rate healthiness, and summarize nutritional facts.
- **Label Detection Validation**: Determines whether an image is a front-of-pack marketing image or the actual nutritional information table.
- **Rate Limiting & Authentication**: Enforced API tier limits with dynamic API keys and device fingerprinting using `slowapi`.

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- [Git](https://git-scm.com/)
- API Keys for **Groq** (`GROQ_API_KEY`)

### 2. Installation

Clone the repository and move into the application directory:
```bash
git clone https://github.com/shaikhsarib/Eatlytic-App.git
cd Eatlytic-App
```

Set up a virtual environment and install the required dependencies:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory based on `.env.example`:
```bash
cp .env.example .env
```
Inside `.env`, populate the `GROQ_API_KEY` corresponding to your Groq account.

### 4. Running Locally

You can run the FastAPI server directly via Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The service will be accessible at:
- Web Interface: `http://localhost:8000/`
- API Documentation (Swagger): `http://localhost:8000/docs`

### 5. Running with Docker

You can easily run the entire application using Docker Compose:
```bash
docker-compose up --build -d
```

## Repository Structure

- `main.py` - Core FastAPI application and logic (OCR, Image Enhancements, LLM integrations).
- `app/` - Modular project routes and services abstractions.
- `Dockerfile` / `docker-compose.yml` - Containerization parameters.

## License

This is a proprietary, closed-source project. All rights reserved. No license is granted for unauthorized use, distribution, or reproduction.
