from app.ai.llm.engine import recover_label_with_ai
from app.ai.llm.client import call_llm, parse_llm_response

__all__ = ["unified_analyze_flow", "recover_label_with_ai", "call_llm", "parse_llm_response"]

def __getattr__(name: str):
    if name == "unified_analyze_flow":
        import app.services.scan_orchestrator as scan_orchestrator
        return scan_orchestrator.unified_analyze_flow
    raise AttributeError(f"module {__name__} has no attribute {name}")

