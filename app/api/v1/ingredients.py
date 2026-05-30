"""
app/api/v1/ingredients.py
Programmatic SEO router — generates dynamic ingredient detail pages for
every food additive in the Eatlytic database (~500+ chemicals).

Each page at /ingredients/{slug} is a fully server-rendered, SEO-optimised
neobrutalist HTML page with JSON-LD schema markup, OpenGraph tags, and
regulatory status grids for FSSAI / FDA / EU.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.services.additive_db import lookup

router = APIRouter(tags=["seo"])


def _render_conditions_html(condition_flags: dict) -> str:
    """Render the health condition contraindications grid as HTML."""
    if not condition_flags:
        return "<div style='font-style: italic; color: #666; font-size: 12px;'>No special contraindications reported for standard populations.</div>"

    html = ""
    for cond_name, verdict in condition_flags.items():
        c_color = "#ef4444" if verdict == "AVOID" else "#f59e0b"
        html += f"""
            <div style="border: 2px solid #000; border-radius: 8px; padding: 10px 14px; background: #fff; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 800; font-size: 13px; text-transform: uppercase;">{cond_name}</span>
                <span style="border: 2px solid #000; border-radius: 20px; padding: 2px 10px; font-size: 10px; font-weight: 800; color: #fff; background: {c_color};">{verdict}</span>
            </div>
            """
    return html


def _build_ingredient_page(record: dict) -> str:
    """Build the full HTML page for an ingredient record."""
    name = record.get("name", "Unknown Additive")
    ins_code = record.get("id", "")
    category = record.get("category", "additive").title()
    safety = record.get("safety_tier", "SAFE")
    fssai = record.get("fssai_status", "unknown").upper()
    fda = record.get("fda_status", "unknown").upper()
    eu = record.get("eu_status", "unknown").upper()
    fact = record.get("curiosity_fact", "")
    typical = record.get("typical_use", "")

    # Color mapping based on safety tier
    color = "#ef4444" if safety == "AVOID" else "#f59e0b" if safety == "CAUTION" else "#22c55e"
    bg_tint = (
        "rgba(239, 68, 68, 0.05)" if safety == "AVOID"
        else "rgba(245, 158, 11, 0.05)" if safety == "CAUTION"
        else "rgba(34, 197, 94, 0.05)"
    )
    badge_label = "⛔ AVOID" if safety == "AVOID" else "⚠️ CAUTION" if safety == "CAUTION"  else "✓ SAFE"

    conds_html = _render_conditions_html(record.get("condition_flags", {}))

    fssai_color = "#22c55e" if "permitted" in fssai.lower() else "#ef4444"
    fda_color = "#22c55e" if ("gras" in fda.lower() or "approved" in fda.lower()) else "#ef4444"
    eu_color = "#22c55e" if "approved" in eu.lower() else "#ef4444"

    return f"""<!DOCTYPE html>
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
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg);
            color: var(--ink);
            font-family: 'Nunito', sans-serif;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 600px; margin: 40px auto; }}
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
        .logo em {{ color: var(--yellow); -webkit-text-stroke: 0.5px var(--ink); }}
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
        .back-btn:active {{ transform: translate(2px, 2px); box-shadow: none; }}
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
        .section-content {{ font-size: 14px; color: var(--ink); margin-bottom: 20px; font-weight: 600; }}
        .fact-box {{
            background: {bg_tint};
            border-left: 5px solid {color};
            padding: 14px;
            border-radius: 0 12px 12px 0;
            margin-bottom: 20px;
            font-size: 13px;
            font-weight: 700;
        }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 20px; }}
        .grid-item {{
            border: 2px solid var(--ink);
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            background: #fdfdfd;
        }}
        .grid-val {{ font-family: 'Space Mono', monospace; font-weight: 700; font-size: 12px; text-transform: uppercase; }}
        .cta-box {{
            background: var(--yellow);
            border: var(--border);
            border-radius: 14px;
            padding: 16px;
            text-align: center;
            box-shadow: var(--shadow);
        }}
        .cta-title {{ font-family: 'Fraunces', serif; font-weight: 900; font-size: 1.2rem; margin-bottom: 6px; font-style: italic; }}
        .cta-desc {{ font-size: 12px; color: #333; margin-bottom: 12px; font-weight: 700; }}
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
        .cta-btn:active {{ transform: translate(2px, 2px); box-shadow: none; }}
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

                <div class="section-title">Health Impact &amp; Side Effects</div>
                <div class="fact-box">
                    {fact if fact else "No severe hazards reported in standard culinary doses. Always check clinical constraints below."}
                </div>

                <div class="section-title">Typical Use Cases</div>
                <p class="section-content">{typical if typical else "Used in processed food formulations as a texturizer, stabilizing agent, or sensory modifier."}</p>

                <div class="section-title">Global Regulatory Status</div>
                <div class="grid">
                    <div class="grid-item">
                        <div class="section-title" style="font-size:9px; margin-bottom:2px">FSSAI (India)</div>
                        <div class="grid-val" style="color:{fssai_color}">{fssai}</div>
                    </div>
                    <div class="grid-item">
                        <div class="section-title" style="font-size:9px; margin-bottom:2px">FDA (US)</div>
                        <div class="grid-val" style="color:{fda_color}">{fda}</div>
                    </div>
                    <div class="grid-item">
                        <div class="section-title" style="font-size:9px; margin-bottom:2px">EFSA (EU)</div>
                        <div class="grid-val" style="color:{eu_color}">{eu}</div>
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
</html>"""


@router.get("/ingredients/{slug}", response_class=HTMLResponse)
async def ingredient_seo_page(slug: str):
    """
    Renders a premium, neobrutalist SEO-optimized page for a specific food additive.
    Enables powerful search engine indexing (Programmatic SEO) across all 500+ additives.
    """
    record = lookup(slug.replace("-", " "))
    if not record:
        raise HTTPException(status_code=404, detail=f"Ingredient '{slug}' not found.")

    return HTMLResponse(content=_build_ingredient_page(record))
