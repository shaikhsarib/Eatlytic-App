"""
app/services/llm/client.py
Multi-provider LLM client with automatic failover.
Priority: Local Gemma 4 (Ollama) -> Google Gemini -> Groq -> Together AI.
"""
import os
import json
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# ── Provider Configuration ──
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "gemma4")
LOCAL_LLM_TIMEOUT = int(os.environ.get("LOCAL_LLM_TIMEOUT", "120"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

# Lazy-init clients
_groq_client = None
_gemini_model = None
_genai_module = None

def _init_groq():
    global _groq_client
    if GROQ_API_KEY and _groq_client is None:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_API_KEY)
        except ImportError:
            logger.warning("groq library not installed — skipping Groq provider")

def _init_gemini():
    global _gemini_model, _genai_module
    if GEMINI_API_KEY and _gemini_model is None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            _genai_module = genai
        except ImportError:
            logger.warning("google-generativeai library not installed — skipping Gemini provider")


# System prompt that trains Gemma 4 to behave as a nutrition specialist
LOCAL_SYSTEM_PROMPT = (
    "You are Eatlytic AI — the world's most precise Nutrition Label Analyst. "
    "Your ONLY job is to read food labels and return structured JSON. "
    "Rules: 1) Extract EVERY SINGLE nutrient row visible on the label. If there are 6 rows, extract 6; "
    "if there are 12 rows, extract all 12. Do not skip any vitamins, minerals, trace elements, sodium, saturated fats, sugars, or others. "
    "2) Use 'per 100g' column when available. "
    "3) Correct obvious OCR errors (l→1, O→0). "
    "4) NEVER return markdown, commentary, or explanation — ONLY valid JSON. "
    "5) If a field is missing from the label, use null. "
    "6) Score products honestly: junk food gets 1-3, healthy food gets 8-10. "
    "7) For the 'ingredients_spotlight', decode every additive (E-numbers, INS codes). "
    "8) Always identify the product name — NEVER say 'Unknown'. "
    "9) Ensure chart_data sums to exactly 100."
)

def _call_local(prompt: str, max_tokens: int) -> str | None:
    """Call local Ollama instance (Gemma 4) with system prompt & large context."""
    if not LOCAL_LLM_URL:
        return None
    try:
        logger.info("🏠 Calling Local %s at %s", LOCAL_LLM_MODEL, LOCAL_LLM_URL)
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
                "num_ctx": 16384,
            }
        }
        res = requests.post(LOCAL_LLM_URL, json=payload, timeout=LOCAL_LLM_TIMEOUT)
        if res.status_code == 200:
            content = res.json().get("message", {}).get("content", "")
            if content and len(content.strip()) > 10:
                return content
            logger.warning("Local LLM returned empty response")
        else:
            logger.warning("Local LLM error %d: %s", res.status_code, res.text[:200])
    except requests.exceptions.Timeout:
        logger.warning("Local LLM timed out after %ds", LOCAL_LLM_TIMEOUT)
    except Exception as e:
        logger.warning("Local LLM failed: %s", e)
    return None


def _call_gemini(prompt: str, max_tokens: int) -> str | None:
    """Call Google Gemini API."""
    _init_gemini()
    if not _gemini_model:
        return None
    try:
        logger.info("☁️ Calling Gemini 1.5 Flash...")
        response = _gemini_model.generate_content(
            prompt,
            generation_config=_genai_module.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        if response.text:
            return response.text
    except Exception as exc:
        logger.warning("Gemini failed: %s", exc)
    return None


def _call_groq(prompt: str, max_tokens: int) -> str | None:
    """Call Groq Cloud API with model fallback."""
    _init_groq()
    if not _groq_client:
        return None
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    for model in models:
        try:
            current_max = 2000 if "8b" in model else max_tokens
            comp = _groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=current_max,
                response_format={"type": "json_object"},
            )
            return comp.choices[0].message.content
        except Exception as exc:
            err_msg = str(exc).lower()
            if "status_code: 413" in err_msg:
                continue
            if "status_code: 429" in err_msg:
                logger.warning("Groq rate limit — trying next model")
                continue
            logger.warning("Groq %s failed: %s", model, exc)
    return None


def _call_together(prompt: str, max_tokens: int) -> str | None:
    """Call Together AI as final fallback."""
    if not TOGETHER_API_KEY:
        return None
    try:
        logger.info("☁️ Calling Together AI fallback...")
        res = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/Llama-3-70b-chat-hf",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            },
            timeout=25,
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        logger.error("Together AI error %d: %s", res.status_code, res.text[:200])
    except Exception as e:
        logger.error("Together AI failed: %s", e)
    return None


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def call_llm(prompt: str, max_tokens: int = 4000) -> str:
    """
    Multi-provider LLM caller with automatic failover.
    Priority: Local Gemma 4 -> Gemini -> Groq -> Together AI.
    """
    providers = [
        ("Local Gemma", _call_local),
        ("Gemini", _call_gemini),
        ("Groq", _call_groq),
        ("Together AI", _call_together),
    ]

    for name, fn in providers:
        result = fn(prompt, max_tokens)
        if result:
            return result

    raise RuntimeError(
        "All AI providers failed. Check: "
        "LOCAL_LLM_URL (Ollama running?), GEMINI_API_KEY, GROQ_API_KEY, or TOGETHER_API_KEY."
    )


def _repair_json(s: str) -> str:
    """Attempt to fix common LLM JSON errors."""
    import re as _re
    # Remove trailing commas before } or ]
    s = _re.sub(r',\s*([}\]])', r'\1', s)
    # Remove // line comments
    s = _re.sub(r'//[^\n]*', '', s)
    # Remove /* block comments */
    s = _re.sub(r'/\*.*?\*/', '', s, flags=_re.DOTALL)
    # Fix truncated JSON — close any open braces/brackets
    opens = s.count('{') - s.count('}')
    s += '}' * max(0, opens)
    opens = s.count('[') - s.count(']')
    s += ']' * max(0, opens)
    return s.strip()


def parse_llm_response(s: str) -> dict:
    """Parse LLM response string into a dict, handling markdown fences and auto-repairing."""
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1]) if len(lines) >= 3 else s.replace("```json", "").replace("```", "").strip()
    # First attempt: direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Second attempt: repair and retry
    try:
        repaired = _repair_json(s)
        result = json.loads(repaired)
        logger.info("JSON auto-repaired successfully")
        return result
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed even after repair: %s | Raw: %.200s", e, s)
        raise
