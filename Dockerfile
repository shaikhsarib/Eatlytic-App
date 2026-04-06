FROM python:3.11-slim

# Set up a new user 'user' with UID 1000
RUN useradd -m -u 1000 user

# Switch to the new user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# System deps for EasyOCR + OpenCV (need root to install)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Pre-download EasyOCR models so first request isn't slow
# Pre-download models for English, Hindi, and Tamil separately to prevent architecture clashing
RUN python -c "import easyocr; \
    easyocr.Reader(['en'], gpu=False); \
    easyocr.Reader(['en', 'hi'], gpu=False); \
    easyocr.Reader(['en', 'ta'], gpu=False)" 2>/dev/null || true

# Expose the default HF Spaces port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
