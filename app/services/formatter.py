"""
app/services/formatter.py
WhatsApp-ready delivery: Tier 1 (Summary) and Tier 2 (Deep Dive).
"""

def format_whatsapp_tier1(analysis: dict) -> str:
    """
    🔴 Verdict One-Liner ( < 80 words )
    """
    product = analysis.get("product_name", "Product")
    verdict = analysis.get("explanation", {}).get("verdict", "🟡 AMBER")
    nova = analysis.get("explanation", {}).get("nova_level", 3)
    primary_con = analysis.get("verdict", "") or "High calorie snack"
    
    emoji = "🔴" if "RED" in verdict else "🟡" if "AMBER" in verdict else "🟢"
    
    text = (
        f"{emoji} *{product}*: {verdict}.\n"
        f"Processing: NOVA {nova}.\n"
        f"Verdict: {primary_con}.\n"
        f"Avoid if you are monitoring sugar/salt intake."
    )
    return text

def format_whatsapp_tier2(analysis: dict) -> str:
    """
    Detailed breakdown for the "Deep Dive".
    """
    product = analysis.get("product_name", "Product")
    nutrients = analysis.get("nutrient_breakdown", [])
    insights = analysis.get("explanation", {}).get("humanized_insights", [])
    
    nut_str = "\n".join([f"- {n['name']}: {n['value']}{n['unit']}" for n in nutrients])
    insight_str = "\n".join(insights)
    
    text = (
        f"🔍 *DEEP DIVE: {product}*\n\n"
        f"*Nutrients per 100g:*\n{nut_str}\n\n"
        f"*Nutritionist Insights:*\n{insight_str}\n\n"
        f"⚖️ *Cultural Comparison:*\n"
        f"This snack is equivalent to ~2 medium Chapatis in calories."
    )
    return text

def get_whatsapp_tiered_content(analysis: dict) -> dict:
    """Returns both tiers for the final response."""
    return {
        "tier1": format_whatsapp_tier1(analysis),
        "tier2": format_whatsapp_tier2(analysis)
    }
