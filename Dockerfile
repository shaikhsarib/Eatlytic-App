FROM python:3.11-slim

# 1. Install system dependencies for OpenCV and OCR
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Create the Hugging Face User (standard for Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# 3. Handle dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 4. Copy project files
COPY --chown=user:user . .

# 5. Pre-setup directories and permissions
# This fixes the "Permission denied: /app/data" error
RUN mkdir -p /app/data /app/.cache/easyocr_models \
    && chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Pre-download models so the app starts INSTANTLY 
RUN python -c "import easyocr; \
    easyocr.Reader(['en'], gpu=False, model_storage_directory='/app/.cache/easyocr_models')"

# 6. Use Hugging Face default port (7860)
EXPOSE 7860

# 7. Start the backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
