"""
scripts/generate_sitemap.py
Generates a standard-compliant XML sitemap for Eatlytic's Programmatic SEO engine.
Compiles all 500+ verified food chemicals from data/additives.json into search engine index seed URLs.
"""
import os
import json
import re
from xml.sax.saxutils import escape

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDITIVES_JSON_PATH = os.path.join(BASE_DIR, "data", "additives.json")
SITEMAP_XML_PATH = os.path.join(BASE_DIR, "static", "sitemap.xml")

def slugify(text: str) -> str:
    """Slugify text by replacing non-alphanumeric characters with hyphens."""
    # Lowercase
    text = text.lower()
    # Replace spaces and special characters with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    # Strip leading/trailing hyphens
    return text.strip("-")

def generate_sitemap(base_url: str = "https://eatlytic.com"):
    print(f"[*] Loading verified additives from: {ADDITIVES_JSON_PATH}")
    if not os.path.exists(ADDITIVES_JSON_PATH):
        print(f"[!] Error: verified additives database not found at {ADDITIVES_JSON_PATH}")
        return

    with open(ADDITIVES_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    additives = data.get("additives", [])
    print(f"[*] Found {len(additives)} verified food additives.")

    urls = []
    
    # 1. Add home page
    urls.append({
        "loc": f"{base_url}/",
        "changefreq": "daily",
        "priority": "1.0"
    })

    # 2. Add each additive's programmatic SEO pages
    for additive in additives:
        # Generate URL by ID/E-number (e.g. /ingredients/E621)
        add_id = additive.get("id")
        if add_id:
            urls.append({
                "loc": f"{base_url}/ingredients/{add_id.lower()}",
                "changefreq": "weekly",
                "priority": "0.8"
            })
            urls.append({
                "loc": f"{base_url}/ingredients/{add_id.upper()}",
                "changefreq": "weekly",
                "priority": "0.8"
            })

        # Generate URL by slugified name (e.g. /ingredients/monosodium-glutamate)
        name = additive.get("name")
        if name:
            slug = slugify(name)
            urls.append({
                "loc": f"{base_url}/ingredients/{slug}",
                "changefreq": "weekly",
                "priority": "0.7"
            })

    # Ensure static directory exists
    os.makedirs(os.path.dirname(SITEMAP_XML_PATH), exist_ok=True)

    print(f"[*] Generating sitemap with {len(urls)} URLs...")

    # Build XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    ]

    for url in urls:
        xml_lines.append("  <url>")
        xml_lines.append(f"    <loc>{escape(url['loc'])}</loc>")
        xml_lines.append(f"    <changefreq>{url['changefreq']}</changefreq>")
        xml_lines.append(f"    <priority>{url['priority']}</priority>")
        xml_lines.append("  </url>")

    xml_lines.append("</urlset>")

    with open(SITEMAP_XML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))

    print(f"[OK] Sitemap successfully compiled and written to: {SITEMAP_XML_PATH}")

if __name__ == "__main__":
    generate_sitemap()
