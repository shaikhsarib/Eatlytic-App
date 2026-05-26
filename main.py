"""
Eatlytic — AI Nutrition Scanner
Entry point: FastAPI application with modular routing.
"""
import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Compatibility patch for older PIL versions
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

load_dotenv()

from app.models.db import init_db
from app.routes import scan, user, admin, b2b, payments, benchmarks, dietitian
from app.routes import food_db


# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Eatlytic: AI Nutrition Scanner", version="3.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS ---
CORS_WHITELIST = [o.strip() for o in os.environ.get("CORS_WHITELIST", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_WHITELIST or [
        "https://eatlytic.com",
        "https://www.eatlytic.com",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"https://.*\.hf\.space",
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# --- Database ---
init_db()

# --- Routers ---
app.include_router(scan.router)
app.include_router(user.router)
app.include_router(admin.router)
app.include_router(b2b.router)
app.include_router(payments.router)
app.include_router(benchmarks.router)
app.include_router(food_db.router)
app.include_router(dietitian.router)


# --- Static ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    from app.services.llm.client import LOCAL_LLM_URL, LOCAL_LLM_MODEL
    return {
        "status": "ok",
        "version": "3.1.0",
        "engine": f"Local ({LOCAL_LLM_MODEL})" if LOCAL_LLM_URL else "Cloud (Groq/Gemini)",
    }

@app.post("/export-pdf")
async def export_pdf(request: Request):
    from app.services.audit_pdf import generate_audit_pdf
    data = await request.json()
    filename = f"static/audit_{data.get('scan_id', 'temp')}.pdf"
    generate_audit_pdf(data, filename)
    return FileResponse(filename, media_type='application/pdf', filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
