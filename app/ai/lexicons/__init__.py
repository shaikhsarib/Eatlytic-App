"""
app/ai/lexicons/__init__.py
Eatlytic AI Lexicon Package.

Contains curated reference dictionaries used by the clinical intelligence
engine to make deterministic, offline nutritional assessments.
"""
from app.ai.lexicons.indian_ingredients import INDIAN_INGREDIENT_LEXICON

__all__ = ["INDIAN_INGREDIENT_LEXICON"]
