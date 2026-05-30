"""
app/api/v1/mcp.py
─────────────────────────────────────────────────────────────────────────────
Model Context Protocol (MCP) Server for Eatlytic.
Exposes tools for autonomous AI agents to query the Personal Nutrition
Intelligence Layer (PNIL) over both SSE (Web/FastAPI) and stdio (CLI).
"""

import json
import logging
import sys
import asyncio
import uuid
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["mcp"])

sse_queues = {}

TOOLS = [
    {
        "name": "get_metabolic_compatibility",
        "description": "Evaluate the metabolic safety and FSSAI compliance of an ingredient list or product based on health persona and optional genomics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "string",
                    "description": "Comma-separated list of ingredients to analyze."
                },
                "persona": {
                    "type": "string",
                    "description": "Target health persona: 'Diabetic Care', 'Hypertension', 'general'."
                },
                "device_key": {
                    "type": "string",
                    "description": "Optional device key to fetch custom genomic risk profile overrides."
                }
            },
            "required": ["ingredients", "persona"]
        }
    },
    {
        "name": "query_chemical_additive",
        "description": "Look up an INS code, E-number, or chemical additive name to return FSSAI regulatory status and toxicity facts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "additive": {
                    "type": "string",
                    "description": "Additive name, E-number or INS code (e.g. 'MSG', 'INS 621', 'E621')."
                }
            },
            "required": ["additive"]
        }
    }
]

async def handle_mcp_request(request_dict: dict) -> dict:
    method = request_dict.get("method")
    req_id = request_dict.get("id")
    params = request_dict.get("params", {})
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "Eatlytic-PNIL-MCP",
                    "version": "1.0.0"
                }
            }
        }
        
    elif method == "notifications/initialized":
        return None
        
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS
            }
        }
        
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "get_metabolic_compatibility":
            ingredients = args.get("ingredients", "")
            persona = args.get("persona", "general")
            device_key = args.get("device_key")
            
            from app.services.additive_db import get_ingredient_risk_summary
            from app.database.connection import get_genomic_profile
            
            risk = get_ingredient_risk_summary(ingredients, persona)
            
            genomic = None
            if device_key:
                try:
                    genomic = get_genomic_profile(device_key)
                except Exception:
                    pass
            
            verdict = "SAFE"
            if risk.get("avoid_count", 0) > 0:
                verdict = "AVOID"
            elif risk.get("caution_count", 0) > 0:
                verdict = "CAUTION"
                
            report = (
                f"🩺 Eatlytic PNIL Verdict: **{verdict}**\n\n"
                f"📊 **Additive Analysis:**\n"
                f"- Total Food Additives Found: {risk.get('total_additives_found', 0)}\n"
                f"- Avoid Tier: {risk.get('avoid_count', 0)}\n"
                f"- Caution Tier: {risk.get('caution_count', 0)}\n"
                f"- Safe Tier: {risk.get('safe_count', 0)}\n\n"
            )
            
            if risk.get("red_flags"):
                report += "**🚩 Flagged Additives:**\n"
                for rf in risk["red_flags"]:
                    report += f"- **{rf['name']}** ({rf['safety_tier']}): {rf['curiosity_fact']}\n"
                    
            if genomic:
                report += f"\n🧬 **Genomic Overlays Applied:** {list(genomic.get('genetic_snps', {}).keys())}\n"
                
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": report
                        }
                    ],
                    "isError": False
                }
            }
            
        elif tool_name == "query_chemical_additive":
            additive_query = args.get("additive", "")
            from app.services.additive_db import lookup
            
            match = lookup(additive_query)
            if not match:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"❌ Additive '{additive_query}' not found in local Eatlytic FSSAI registry."
                            }
                        ],
                        "isError": True
                    }
                }
                
            report = (
                f"🔬 **Chemical Additive Found:** {match['name']}\n"
                f"- **INS/E-Number:** {match.get('id', 'N/A')}\n"
                f"- **Safety Tier:** {match['safety_tier']}\n"
                f"- **FSSAI Clearance Status:** {match.get('fssai_status', 'CLEAR')}\n"
                f"- **Category:** {match.get('category', 'General')}\n"
                f"- **Toxicity & Curiosity Fact:** {match.get('curiosity_fact', '')}\n"
            )
            
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": report
                        }
                    ],
                    "isError": False
                }
            }
            
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {tool_name}"
                }
            }
            
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }

@router.get("/api/v1/mcp/sse")
async def mcp_sse(request: Request):
    client_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    sse_queues[client_id] = queue
    
    async def event_generator():
        try:
            # SSE Handshake: Expose the postback URI with token
            yield f"event: endpoint\ndata: /api/v1/mcp/message?client_id={client_id}\n\n"
            while True:
                response_payload = await queue.get()
                yield f"event: message\ndata: {json.dumps(response_payload)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            sse_queues.pop(client_id, None)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/api/v1/mcp/message")
async def mcp_message(client_id: str, request_payload: dict):
    if client_id not in sse_queues:
        raise HTTPException(status_code=400, detail="Unknown SSE client_id")
        
    response = await handle_mcp_request(request_payload)
    if response:
        await sse_queues[client_id].put(response)
    return {"status": "accepted"}

async def run_stdio_mcp_server():
    """Standard IO transport reader for local LLM tools integration.
    Uses cross-platform, thread-safe input reading to support Windows ProactorEventLoop."""
    logger.info("Starting Eatlytic stdio MCP Server...")
    
    def read_line():
        return sys.stdin.readline()
        
    while True:
        line = await asyncio.to_thread(read_line)
        if not line:
            break
        try:
            line_str = line.strip()
            if not line_str:
                continue
            request_data = json.loads(line_str)
            response = await handle_mcp_request(request_data)
            if response:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except Exception as e:
            logger.error("Stdio MCP Server error: %s", e)

