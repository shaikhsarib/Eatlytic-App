FROM python:3.11-slim

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/.cache/easyocr_models

RUN python -c "import easyocr; \
    easyocr.Reader(['en'], gpu=False, model_storage_directory='/app/.cache/easyocr_models'); \
    easyocr.Reader(['en', 'hi', 'ta'], gpu=False, model_storage_directory='/app/.cache/easyocr_models'); \
    easyocr.Reader(['en', 'ch_sim'], gpu=False, model_storage_directory='/app/.cache/easyocr_models')" 2>/dev/null || true

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
