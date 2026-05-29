"""
app/services/additive_db.py
Eatlytic Proprietary Food Additive Intelligence Service.

Loads the verified additives.json database and provides fast O(1) lookups
by ingredient name, alias, INS code, or E-number.

This is the core data moat — deterministic, verified safety data
that does NOT rely on LLM inference.
"""

import json
import re
import os
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ── Database Load ─────────────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "additives.json")
_DB_PATH = os.path.normpath(_DB_PATH)

_ADDITIVE_BY_ALIAS: Dict[str, dict] = {}
_ADDITIVE_BY_INS: Dict[str, dict] = {}
_ADDITIVE_BY_ID: Dict[str, dict] = {}
_ALL_ADDITIVES: List[dict] = []


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _load_db() -> None:
    """Load additives.json into memory indexes. Called once on import."""
    global _ALL_ADDITIVES
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _ALL_ADDITIVES = data.get("additives", [])
    except FileNotFoundError:
        logger.warning("additives.json not found at %s — additive lookup disabled", _DB_PATH)
        return
    except json.JSONDecodeError as e:
        logger.error("additives.json is malformed: %s", e)
        return

    for entry in _ALL_ADDITIVES:
        # Index by canonical ID (e.g. "E621", "FSSAI-001")
        if entry.get("id"):
            _ADDITIVE_BY_ID[entry["id"].upper()] = entry

        # Index by INS code
        if entry.get("ins"):
            ins_key = f"ins {entry['ins']}"
            _ADDITIVE_BY_INS[ins_key] = entry
            _ADDITIVE_BY_INS[f"e{entry['ins']}"] = entry  # E621 → e621

        # Index by all aliases (normalized)
        for alias in entry.get("aliases", []):
            normalized = _normalize(alias)
            if normalized:
                _ADDITIVE_BY_ALIAS[normalized] = entry

    logger.info("Additive DB loaded: %d entries, %d alias keys", len(_ALL_ADDITIVES), len(_ADDITIVE_BY_ALIAS))


# Load on module import
_load_db()


# ── Public API ────────────────────────────────────────────────────────────────

def lookup(ingredient_text: str) -> Optional[dict]:
    """
    Look up an additive by any name, alias, INS code, or E-number.
    Returns the full additive record or None.

    Examples:
        lookup("MSG")           → Monosodium Glutamate record
        lookup("E621")          → Monosodium Glutamate record
        lookup("INS 621")       → Monosodium Glutamate record
        lookup("Sodium Benzoate") → Sodium Benzoate record
        lookup("Red 40")        → Allura Red AC record
    """
    if not ingredient_text:
        return None

    normalized = _normalize(ingredient_text)
    if not normalized:
        return None

    # 1. Direct alias match
    if normalized in _ADDITIVE_BY_ALIAS:
        return _ADDITIVE_BY_ALIAS[normalized]

    # 2. INS/E-number pattern match: "ins 621", "e621", "e-621"
    ins_match = re.search(r"\b(?:ins\s*|e[-\s]?)(\d{3,4}[a-z]?)\b", normalized)
    if ins_match:
        ins_key = f"ins {ins_match.group(1)}"
        if ins_key in _ADDITIVE_BY_INS:
            return _ADDITIVE_BY_INS[ins_key]
        e_key = f"e{ins_match.group(1)}"
        if e_key in _ADDITIVE_BY_INS:
            return _ADDITIVE_BY_INS[e_key]

    # 3. Partial substring match (for compound names like "sodium benzoate salt")
    for alias_key, record in _ADDITIVE_BY_ALIAS.items():
        if alias_key in normalized or normalized in alias_key:
            return record

    return None


def scan_ingredients(ingredients_text: str) -> List[dict]:
    """
    Scan a full ingredients list string and return all matched additives.
    Splits on commas, parentheses, and semicolons before matching.

    Returns a list of match records with the original text fragment included.
    """
    if not ingredients_text:
        return []

    # Split ingredients list into individual tokens
    tokens = re.split(r"[,;()\[\]]+", ingredients_text)
    matches = []
    seen_ids = set()

    for token in tokens:
        token = token.strip()
        if len(token) < 2:
            continue

        record = lookup(token)
        if record and record["id"] not in seen_ids:
            seen_ids.add(record["id"])
            matches.append({
                **record,
                "matched_text": token,
            })

    return matches


def get_ingredient_risk_summary(ingredients_text: str, persona: str = "general") -> dict:
    """
    High-level risk analysis of an ingredient list.
    Returns counts by safety tier and a list of flagged additives.

    Used by brain.py to enrich the local report with verified additive data.
    """
    matches = scan_ingredients(ingredients_text)

    safe = [m for m in matches if m["safety_tier"] == "SAFE"]
    caution = [m for m in matches if m["safety_tier"] == "CAUTION"]
    avoid = [m for m in matches if m["safety_tier"] == "AVOID"]

    # Persona-specific flags (e.g. diabetic should avoid maltitol)
    persona_lower = persona.lower()
    persona_flags = []
    for m in matches:
        cf = m.get("condition_flags", {})
        # Check persona against condition flags
        for flag_key, flag_val in cf.items():
            if flag_key in persona_lower or persona_lower in flag_key:
                if flag_val in ("AVOID", "CAUTION"):
                    persona_flags.append({
                        "name": m["name"],
                        "flag": flag_val,
                        "reason": f"Flagged for {flag_key}: {m.get('curiosity_fact', '')}",
                        "matched_text": m.get("matched_text", "")
                    })
                    break

    # Build red flags list (AVOID tier first, then CAUTION)
    red_flags = [
        {
            "name": m["name"],
            "safety_tier": m["safety_tier"],
            "fssai_status": m.get("fssai_status", "unknown"),
            "curiosity_fact": m.get("curiosity_fact", ""),
            "matched_text": m.get("matched_text", ""),
            "category": m.get("category", "")
        }
        for m in (avoid + caution)
    ]

    return {
        "total_additives_found": len(matches),
        "safe_count": len(safe),
        "caution_count": len(caution),
        "avoid_count": len(avoid),
        "red_flags": red_flags,
        "persona_flags": persona_flags,
        "matched_additives": [m["name"] for m in matches],
    }


def get_all_additives() -> List[dict]:
    """Return all loaded additives (for search/browse endpoints)."""
    return _ALL_ADDITIVES


def get_additive_by_id(additive_id: str) -> Optional[dict]:
    """Get a specific additive by its ID (e.g. 'E621', 'FSSAI-001')."""
    return _ADDITIVE_BY_ID.get(additive_id.upper())
