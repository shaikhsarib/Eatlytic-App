import re
import secrets
import json
import hashlib
from fastapi import APIRouter, Request, Form, HTTPException, Query
from app.models.db import db_conn

router = APIRouter(prefix="/dietitian", tags=["dietitian"])

@router.post("/register")
async def register_dietitian(
    name: str = Form(...),
    email: str = Form(...),
    code: str = Form(...)
):
    """
    Registers a clinical dietitian/nutritionist and provides them with a cohort 
    code and dietitian key to access aggregate patient insights.
    """
    cleaned_code = re.sub(r"[^\w]", "", code.strip().upper())
    if not cleaned_code:
        raise HTTPException(status_code=400, detail="Invalid cohort code.")
    
    # Generate secure admin access key
    key = "diet_" + secrets.token_urlsafe(32)
    
    with db_conn() as conn:
        existing = conn.execute("SELECT code FROM dietitians WHERE code=?", (cleaned_code,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Cohort code is already registered.")
        
        conn.execute(
            "INSERT INTO dietitians (code, name, email, dietitian_key) VALUES (?, ?, ?, ?)",
            (cleaned_code, name.strip(), email.strip().lower(), key)
        )
        
    return {
        "status": "registered",
        "cohort_code": cleaned_code,
        "dietitian_key": key,
        "dashboard_url": f"/dietitian/dashboard?key={key}"
    }

@router.post("/join")
async def join_cohort(
    device_key: str = Form(...),
    cohort_code: str = Form(...)
):
    """
    Ties a patient's device_key to a dietitian's tracking cohort code.
    All subsequent scans are aggregated in the dietitian dashboard.
    """
    cleaned_code = cohort_code.strip().upper()
    with db_conn() as conn:
        dietitian = conn.execute("SELECT code, name FROM dietitians WHERE code=?", (cleaned_code,)).fetchone()
        if not dietitian:
            raise HTTPException(status_code=404, detail="Cohort code not found.")
        
        conn.execute(
            "INSERT OR IGNORE INTO patient_cohorts (device_key, dietitian_code) VALUES (?, ?)",
            (device_key.strip(), cleaned_code)
        )
        
    return {
        "status": "joined",
        "cohort_code": cleaned_code,
        "dietitian_name": dietitian["name"],
        "message": f"Successfully joined {dietitian['name']}'s cohort!"
    }

@router.get("/dashboard")
async def get_dietitian_dashboard(
    key: str = Query(...)
):
    """
    Provides clinical nutritionist with aggregate cohort stats, patient-by-patient 
    telemetry, and the Glycemic Threat Feed for patient intervention.
    """
    with db_conn() as conn:
        dietitian = conn.execute("SELECT * FROM dietitians WHERE dietitian_key=?", (key.strip(),)).fetchone()
        if not dietitian:
            raise HTTPException(status_code=403, detail="Invalid dietitian key.")
        
        code = dietitian["code"]
        
        # Fetch connected device keys
        cohort_rows = conn.execute("SELECT device_key FROM patient_cohorts WHERE dietitian_code=?", (code,)).fetchall()
        device_keys = [r["device_key"] for r in cohort_rows]
        
        if not device_keys:
            return {
                "dietitian_name": dietitian["name"],
                "cohort_code": code,
                "total_patients": 0,
                "total_scans_30d": 0,
                "safety_ratios": {"Safe": 0.0, "Limit": 0.0, "Avoid": 0.0},
                "glycemic_threat_feed": [],
                "recent_scans": []
            }
            
        # Compile dynamically parameterized SQL
        placeholders = ",".join("?" for _ in device_keys)
        scans_query = f"""
            SELECT id, device_key, product_name, score, verdict, persona, scanned_at, brand, category, analysis_json
            FROM scans
            WHERE device_key IN ({placeholders}) AND scanned_at >= datetime('now', '-30 days')
            ORDER BY scanned_at DESC
        """
        scan_rows = conn.execute(scans_query, device_keys).fetchall()
        
    total_scans = len(scan_rows)
    
    # Safety distributions
    safe_count = 0
    limit_count = 0
    avoid_count = 0
    
    glycemic_threat_feed = []
    recent_scans = []
    
    for r in scan_rows:
        row_dict = dict(r)
        score = row_dict["score"]
        
        if score >= 7:
            safe_count += 1
            tier = "Safe"
        elif score >= 4:
            limit_count += 1
            tier = "Limit"
        else:
            avoid_count += 1
            tier = "Avoid"
            
        try:
            analysis = json.loads(row_dict["analysis_json"])
        except:
            analysis = {}
            
        # Determine if this triggers a Glycemic or Health threat for diabetic patients
        is_glycemic_threat = (
            score < 4 or
            row_dict["verdict"] == "Glycemic Threat" or
            tier == "Avoid" or
            any(x in row_dict["product_name"].lower() for x in ["maltodextrin", "maida", "jim jam"]) or
            any("glycemic" in str(c).lower() for c in analysis.get("cons", [])) or
            any(any(kw in str(c).lower() for kw in ["high in added sugar", "excessive sugar", "glycemic threat", "insulin spike"]) for c in analysis.get("cons", []))
        )
        
        # Short MD5 hash of device key for anonymized Patient ID
        patient_id = f"PATIENT_{hashlib.md5(row_dict['device_key'].encode()).hexdigest()[:6].upper()}"
        
        scan_summary = {
            "id": row_dict["id"],
            "patient_id": patient_id,
            "product_name": row_dict["product_name"],
            "brand": row_dict["brand"] or "Unknown",
            "score": score,
            "verdict": row_dict["verdict"],
            "scanned_at": row_dict["scanned_at"],
            "category": row_dict["category"] or "other",
            "tier": tier
        }
        
        recent_scans.append(scan_summary)
        
        if is_glycemic_threat:
            threat_triggers = []
            cons_list = analysis.get("cons", [])
            for c in cons_list:
                if any(kw in c.lower() for kw in ["sugar", "glycemic", "maida", "maltodextrin", "insulin", "starch", "palm oil", "sodium"]):
                    threat_triggers.append(c)
            
            glycemic_threat_feed.append({
                "scan_id": row_dict["id"],
                "patient_id": patient_id,
                "product_name": scan_summary["product_name"],
                "brand": scan_summary["brand"],
                "scanned_at": scan_summary["scanned_at"],
                "sugar_content": analysis.get("sugar", 0.0),
                "sodium_content": analysis.get("sodium_mg", analysis.get("sodium", 0.0)),
                "verdict": scan_summary["verdict"],
                "threat_triggers": threat_triggers or ["Glycemic threat trigger flagged by local rules engine."]
            })
            
    # Calculate ratios
    safety_ratios = {
        "Safe": round((safe_count / total_scans * 100), 1) if total_scans > 0 else 0.0,
        "Limit": round((limit_count / total_scans * 100), 1) if total_scans > 0 else 0.0,
        "Avoid": round((avoid_count / total_scans * 100), 1) if total_scans > 0 else 0.0
    }
    
    return {
        "dietitian_name": dietitian["name"],
        "cohort_code": code,
        "total_patients": len(device_keys),
        "total_scans_30d": total_scans,
        "safety_ratios": safety_ratios,
        "glycemic_threat_feed": glycemic_threat_feed[:20],
        "recent_scans": recent_scans[:50]
    }
