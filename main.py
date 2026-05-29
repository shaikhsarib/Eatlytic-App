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

from app.database.connection import init_db
from app.api.v1 import scan, user, admin, b2b, payments, benchmarks, dietitian, personalized, cgm
from app.api.v1 import food_db
from app.api.v1 import additive_db as additive_db_route



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
app.include_router(personalized.router)
app.include_router(additive_db_route.router)
app.include_router(cgm.router)


# --- Static ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.get("/developer")
async def developer_portal():
    return FileResponse("static/developer.html")

@app.get("/health")
async def health():
    from app.ai.llm.client import LOCAL_LLM_URL, LOCAL_LLM_MODEL
    return {
        "status": "ok",
        "version": "3.1.0",
        "engine": f"Local ({LOCAL_LLM_MODEL})" if LOCAL_LLM_URL else "Cloud (Groq/Gemini)",
    }

@app.get("/sitemap.xml")
async def sitemap():
    import os
    if os.path.exists("static/sitemap.xml"):
        return FileResponse("static/sitemap.xml", media_type="application/xml")
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Sitemap not generated yet.")

@app.post("/export-pdf")
async def export_pdf(request: Request):
    # BUG 5 FIX: require valid session before generating a PDF
    from app.services.user_auth import get_user_from_token
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
    if not token:
        token = request.cookies.get("session_token")
    user = get_user_from_token(token) if token else None
    if not user:
        from fastapi.responses import JSONResponse as _JR
        return _JR(status_code=401, content={"error": "Login required to export PDF."})
    from app.services.audit_pdf import generate_audit_pdf
    data = await request.json()
    filename = f"static/audit_{data.get('scan_id', 'temp')}.pdf"
    generate_audit_pdf(data, filename)
    return FileResponse(filename, media_type='application/pdf', filename=filename)


@app.get("/ingredients/{slug}")
async def ingredient_seo_page(slug: str):
    """
    Renders a premium, neobrutalist SEO-optimized page for a specific food additive.
    Enables powerful search engine indexing (Programmatic SEO) across all 500+ additives.
    """
    from app.services.additive_db import lookup
    
    # 1. Lookup the chemical
    record = lookup(slug.replace("-", " "))
    if not record:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Ingredient '{slug}' not found.")
        
    # 2. Compile neobrutalist dynamic HTML
    name = record.get("name", "Unknown Additive")
    ins_code = record.get("id", "")
    category = record.get("category", "additive").title()
    safety = record.get("safety_tier", "SAFE")
    fssai = record.get("fssai_status", "unknown").upper()
    fda = record.get("fda_status", "unknown").upper()
    eu = record.get("eu_status", "unknown").upper()
    fact = record.get("curiosity_fact", "")
    typical = record.get("typical_use", "")
    
    # Color mapping
    color = "#ef4444" if safety == "AVOID" else "#f59e0b" if safety == "CAUTION" else "#22c55e"
    bg_tint = "rgba(239, 68, 68, 0.05)" if safety == "AVOID" else "rgba(245, 158, 11, 0.05)" if safety == "CAUTION" else "rgba(34, 197, 94, 0.05)"
    badge_label = "⛔ AVOID" if safety == "AVOID" else "⚠️ CAUTION" if safety == "CAUTION" else "✓ SAFE"
    
    # Format conditions checklist
    conds = record.get("condition_flags", {})
    conds_html = ""
    if conds:
        for cond_name, verdict in conds.items():
            c_color = "#ef4444" if verdict == "AVOID" else "#f59e0b"
            conds_html += f"""
            <div style="border: 2px solid #000; border-radius: 8px; padding: 10px 14px; background: #fff; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 800; font-size: 13px; text-transform: uppercase;">{cond_name}</span>
                <span style="border: 2px solid #000; border-radius: 20px; padding: 2px 10px; font-size: 10px; font-weight: 800; color: #fff; background: {c_color};">{verdict}</span>
            </div>
            """
    else:
        conds_html = "<div style='font-style: italic; color: #666; font-size: 12px;'>No special contraindications reported for standard populations.</div>"
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Is {name} ({ins_code}) Safe? | Eatlytic Food Chemical Database</title>
        <meta name="description" content="Safety analysis of {name} ({ins_code}). FSSAI: {fssai}, FDA: {fda}. Read clinical facts, safety ratings, side effects, and health impact instantly.">
        
        <!-- OpenGraph -->
        <meta property="og:title" content="Is {name} ({ins_code}) Safe? | Eatlytic">
        <meta property="og:description" content="Verified chemical safety profile of {name}. Health risk audit, regulatory status, and clinical analysis.">
        <meta property="og:type" content="article">
        
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Fraunces:ital,opsz,wght@0,9..144,100..900;1,9..144,100..900&family=Nunito:wght@400;600;700;800;900&display=swap" rel="stylesheet">
        
        <style>
            :root {{
                --bg: #F5F3EE;
                --ink: #0A0A0A;
                --white: #FFFFFF;
                --yellow: #FFD600;
                --border: 2.5px solid var(--ink);
                --shadow: 4px 4px 0px var(--ink);
            }}
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body {{
                background-color: var(--bg);
                color: var(--ink);
                font-family: 'Nunito', sans-serif;
                padding: 20px;
                line-height: 1.6;
            }}
            .container {{
                max-width: 600px;
                margin: 40px auto;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: var(--border);
                padding-bottom: 12px;
                margin-bottom: 24px;
            }}
            .logo {{
                font-family: 'Fraunces', serif;
                font-weight: 900;
                font-size: 1.5rem;
                font-style: italic;
                letter-spacing: -0.5px;
            }}
            .logo em {{
                color: var(--yellow);
                -webkit-text-stroke: 0.5px var(--ink);
            }}
            .back-btn {{
                font-family: 'Space Mono', monospace;
                font-weight: 700;
                text-decoration: none;
                color: var(--ink);
                border: 2px solid var(--ink);
                padding: 4px 10px;
                border-radius: 8px;
                background: var(--white);
                font-size: 11px;
                box-shadow: 2px 2px 0px var(--ink);
                transition: transform 0.1s, box-shadow 0.1s;
            }}
            .back-btn:active {{
                transform: translate(2px, 2px);
                box-shadow: none;
            }}
            .card {{
                background: var(--white);
                border: var(--border);
                border-radius: 16px;
                padding: 24px;
                box-shadow: var(--shadow);
                margin-bottom: 24px;
            }}
            .title {{
                font-family: 'Fraunces', serif;
                font-weight: 900;
                font-size: 2.2rem;
                letter-spacing: -1px;
                line-height: 1.1;
                margin-bottom: 8px;
            }}
            .code-badge {{
                font-family: 'Space Mono', monospace;
                font-size: 10px;
                font-weight: 700;
                background: var(--ink);
                color: var(--yellow);
                padding: 2px 8px;
                border-radius: 4px;
                display: inline-block;
                margin-bottom: 16px;
                text-transform: uppercase;
            }}
            .safety-pill {{
                display: inline-block;
                border: var(--border);
                border-radius: 20px;
                padding: 4px 14px;
                font-size: 12px;
                font-weight: 900;
                color: var(--white);
                background: {color};
                margin-bottom: 16px;
                box-shadow: 2px 2px 0px var(--ink);
            }}
            .section-title {{
                font-family: 'Space Mono', monospace;
                font-weight: 700;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #666;
                margin-bottom: 6px;
            }}
            .section-content {{
                font-size: 14px;
                color: var(--ink);
                margin-bottom: 20px;
                font-weight: 600;
            }}
            .fact-box {{
                background: {bg_tint};
                border-left: 5px solid {color};
                padding: 14px;
                border-radius: 0 12px 12px 0;
                margin-bottom: 20px;
                font-size: 13px;
                font-weight: 700;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 10px;
                margin-bottom: 20px;
            }}
            .grid-item {{
                border: 2px solid var(--ink);
                border-radius: 10px;
                padding: 10px;
                text-align: center;
                background: #fdfdfd;
            }}
            .grid-val {{
                font-family: 'Space Mono', monospace;
                font-weight: 700;
                font-size: 12px;
                text-transform: uppercase;
            }}
            .cta-box {{
                background: var(--yellow);
                border: var(--border);
                border-radius: 14px;
                padding: 16px;
                text-align: center;
                box-shadow: var(--shadow);
            }}
            .cta-title {{
                font-family: 'Fraunces', serif;
                font-weight: 900;
                font-size: 1.2rem;
                margin-bottom: 6px;
                font-style: italic;
            }}
            .cta-desc {{
                font-size: 12px;
                color: #333;
                margin-bottom: 12px;
                font-weight: 700;
            }}
            .cta-btn {{
                display: inline-block;
                background: var(--ink);
                color: var(--yellow);
                border: var(--border);
                border-radius: 20px;
                padding: 10px 24px;
                text-decoration: none;
                font-weight: 900;
                font-family: 'Fraunces', serif;
                font-style: italic;
                font-size: 13px;
                box-shadow: 2px 2px 0px var(--ink);
                transition: transform 0.1s, box-shadow 0.1s;
            }}
            .cta-btn:active {{
                transform: translate(2px, 2px);
                box-shadow: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header class="header">
                <div class="logo">Eat<em>l</em>ytic</div>
                <a href="/" class="back-btn">← Back to Portal</a>
            </header>
            
            <main>
                <div class="card">
                    <span class="code-badge">{ins_code}</span>
                    <h1 class="title">{name}</h1>
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <span class="safety-pill">{badge_label}</span>
                        <span style="font-size:12px; font-weight:800; color:#666">{category}</span>
                    </div>
                    
                    <div class="section-title">What it is</div>
                    <p class="section-content">
                        {name} (INS/E-Number: {ins_code}) is a common food chemical categorized as a <strong>{category.lower()}</strong> in industrial food formulations.
                    </p>
                    
                    <div class="section-title">Health Impact & Side Effects</div>
                    <div class="fact-box">
                        {fact if fact else "No severe hazards reported in standard culinary doses. Always check clinical constraints below."}
                    </div>
                    
                    <div class="section-title">Typical Use Cases</div>
                    <p class="section-content">{typical if typical else "Used in processed food formulations as a texturizer, stabilizing agent, or sensory modifier."}</p>
                    
                    <div class="section-title">Global Regulatory Status</div>
                    <div class="grid">
                        <div class="grid-item">
                            <div class="section-title" style="font-size:9px; margin-bottom:2px">FSSAI (India)</div>
                            <div class="grid-val" style="color:{'#22c55e' if 'permitted' in fssai.lower() else '#ef4444'}">{fssai}</div>
                        </div>
                        <div class="grid-item">
                            <div class="section-title" style="font-size:9px; margin-bottom:2px">FDA (US)</div>
                            <div class="grid-val" style="color:{'#22c55e' if 'gras' in fda.lower() or 'approved' in fda.lower() else '#ef4444'}">{fda}</div>
                        </div>
                        <div class="grid-item">
                            <div class="section-title" style="font-size:9px; margin-bottom:2px">EFSA (EU)</div>
                            <div class="grid-val" style="color:{'#22c55e' if 'approved' in eu.lower() else '#ef4444'}">{eu}</div>
                        </div>
                    </div>

                    <div class="section-title">Metabolic Genomics Contraindications</div>
                    {conds_html}
                </div>
                
                <div class="cta-box">
                    <h2 class="cta-title">Want a Personalized Health Scan?</h2>
                    <p class="cta-desc">
                        Scan any food product in under 10 milliseconds to check for hidden chemicals, added sugars, and genetic compatibility.
                    </p>
                    <a href="/" class="cta-btn">Scan Food Now →</a>
                </div>
            </main>
        </div>
        
        <!-- JSON-LD Programmatic SEO Schema Markup -->
        <script type="application/ld+json">
        {{
            "@context": "https://schema.org",
            "@type": "MedicalWebPage",
            "name": "Is {name} ({ins_code}) Safe?",
            "description": "Clinical safety analysis and regulatory clearance status for food additive {name}.",
            "lastReviewed": "2026-05-27",
            "reviewedBy": {{
                "@type": "Organization",
                "name": "Eatlytic Clinical Research Team"
            }}
        }}
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
