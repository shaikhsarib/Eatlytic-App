FROM python:3.11-slim

WORKDIR /app

# System deps for EasyOCR + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download EasyOCR models so first request isn't slow
RUN python -c "import easyocr; r = easyocr.Reader(['en'], gpu=False, download_enabled=False)" 2>/dev/null || true

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]
