import json
import hashlib
from typing import List
from pydantic import BaseModel
from fastapi import APIRouter, Query, HTTPException, Request
from app.database.connection import db_conn

class CGMReadingModel(BaseModel):
    glucose_mgdl: float
    recorded_at: str
    sensor_state: str = "active"

class CGMSyncPayload(BaseModel):
    device_key: str
    readings: List[CGMReadingModel]

router = APIRouter(prefix="/cgm", tags=["cgm"])

@router.post("/sync")
async def sync_cgm_readings(payload: CGMSyncPayload):
    """
    Deduplicated ingestion endpoint for FreeStyle Libre or Dexcom CGM telemetry.
    Locks the database to guarantee transactional integrity during high-frequency uploads.
    """
    device_key = payload.device_key.strip()
    if not device_key:
        raise HTTPException(status_code=400, detail="Invalid device key.")
    
    inserted = 0
    with db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        
        # Check if the device exists
        device = conn.execute("SELECT device_key FROM devices WHERE device_key=?", (device_key,)).fetchone()
        if not device:
            # Create anonymous device slot locally to guarantee zero-friction onboarding
            conn.execute("INSERT INTO devices (device_key, streak_days, last_scan_date) VALUES (?, 0, '')", (device_key,))
        
        # Insert non-duplicative readings
        for r in payload.readings:
            dup = conn.execute(
                "SELECT id FROM cgm_readings WHERE device_key=? AND recorded_at=?",
                (device_key, r.recorded_at.strip())
            ).fetchone()
            
            if not dup:
                conn.execute(
                    "INSERT INTO cgm_readings (device_key, glucose_mgdl, recorded_at, sensor_state) VALUES (?, ?, ?, ?)",
                    (device_key, r.glucose_mgdl, r.recorded_at.strip(), r.sensor_state.strip())
                )
                inserted += 1
                
    return {
        "status": "success",
        "device_key": device_key,
        "readings_received": len(payload.readings),
        "new_readings_inserted": inserted
    }

@router.get("/stats")
async def get_cgm_stats(device_key: str = Query(...)):
    """
    Computes clinically validated 30-day glycemic statistics:
    - Average glucose (mg/dL)
    - eA1c (Estimated HbA1c): (AvgGlucose + 46.7) / 28.7
    - Time-in-Range (TIR) percentage: readings between 70 and 140 mg/dL
    - Hyperglycemia (>140 mg/dL)
    - Hypoglycemia (<70 mg/dL)
    """
    device_key = device_key.strip()
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT glucose_mgdl, recorded_at, sensor_state 
            FROM cgm_readings 
            WHERE device_key=? AND datetime(recorded_at) >= datetime('now', '-30 days')
            ORDER BY recorded_at DESC
            """,
            (device_key,)
        ).fetchall()
        
    if not rows:
        return {
            "device_key": device_key,
            "total_readings_30d": 0,
            "average_glucose_mgdl": 0.0,
            "estimated_hba1c": 0.0,
            "time_in_range_percent": 0.0,
            "hyperglycemia_percent": 0.0,
            "hypoglycemia_percent": 0.0
        }
        
    total = len(rows)
    glucose_sum = 0.0
    in_range_count = 0
    hyper_count = 0
    hypo_count = 0
    
    for r in rows:
        val = r["glucose_mgdl"]
        glucose_sum += val
        if 70 <= val <= 140:
            in_range_count += 1
        elif val > 140:
            hyper_count += 1
        else:
            hypo_count += 1
            
    avg_glucose = round(glucose_sum / total, 2)
    estimated_hba1c = round((avg_glucose + 46.7) / 28.7, 2)
    
    return {
        "device_key": device_key,
        "total_readings_30d": total,
        "average_glucose_mgdl": avg_glucose,
        "estimated_hba1c": estimated_hba1c,
        "time_in_range_percent": round((in_range_count / total) * 100, 1),
        "hyperglycemia_percent": round((hyper_count / total) * 100, 1),
        "hypoglycemia_percent": round((hypo_count / total) * 100, 1)
    }

@router.get("/correlations")
async def get_cgm_correlations(device_key: str = Query(...)):
    """
    Correlates patient product scans/meal logs with blood sugar excursions.
    Identifies severe post-prandial spikes (>30 mg/dL increase within a 2-hour window after scans).
    """
    device_key = device_key.strip()
    correlations = []
    
    with db_conn() as conn:
        scans = conn.execute(
            """
            SELECT id, product_name, brand, scanned_at, score, verdict 
            FROM scans 
            WHERE device_key=? AND datetime(scanned_at) >= datetime('now', '-14 days')
            ORDER BY scanned_at DESC
            """,
            (device_key,)
        ).fetchall()
        
        for scan in scans:
            scan_time = scan["scanned_at"]
            
            # Find glucose reading closest to scan_time
            baseline_row = conn.execute(
                """
                SELECT glucose_mgdl, recorded_at 
                FROM cgm_readings 
                WHERE device_key=? 
                  AND datetime(recorded_at) >= datetime(?, '-15 minutes') 
                  AND datetime(recorded_at) <= datetime(?, '+15 minutes')
                ORDER BY abs(strftime('%s', datetime(recorded_at)) - strftime('%s', datetime(?))) ASC
                LIMIT 1
                """,
                (device_key, scan_time, scan_time, scan_time)
            ).fetchone()
            
            if not baseline_row:
                baseline_row = conn.execute(
                    """
                    SELECT glucose_mgdl, recorded_at 
                    FROM cgm_readings 
                    WHERE device_key=? AND datetime(recorded_at) <= datetime(?)
                    ORDER BY datetime(recorded_at) DESC
                    LIMIT 1
                    """,
                    (device_key, scan_time)
                ).fetchone()
                
            if not baseline_row:
                continue
                
            baseline_glucose = baseline_row["glucose_mgdl"]
            
            # Query maximum reading in the 2-hour post-prandial window
            post_readings = conn.execute(
                """
                SELECT glucose_mgdl, recorded_at 
                FROM cgm_readings 
                WHERE device_key=? 
                  AND datetime(recorded_at) > datetime(?) 
                  AND datetime(recorded_at) <= datetime(?, '+2 hours')
                ORDER BY glucose_mgdl DESC
                LIMIT 1
                """,
                (device_key, scan_time, scan_time)
            ).fetchall()
            
            if not post_readings:
                continue
                
            max_post_glucose = post_readings[0]["glucose_mgdl"]
            spike = round(max_post_glucose - baseline_glucose, 1)
            is_trigger = spike >= 30.0
            
            correlations.append({
                "scan_id": scan["id"],
                "product_name": scan["product_name"],
                "brand": scan["brand"] or "Unknown",
                "scanned_at": scan_time,
                "score_assigned": scan["score"],
                "baseline_glucose": baseline_glucose,
                "max_post_glucose": max_post_glucose,
                "glucose_spike": spike,
                "is_biosensor_trigger": is_trigger,
                "clinical_recommendation": (
                    "SEVERE GLYCEMIC TRIGGER: High physical glycemic excursion verified by biosensor. "
                    "Consider swapping for approved low-glycemic alternatives."
                    if is_trigger else "Metabolic safe: Product generated normal post-prandial glycemic response."
                )
            })
            
    return {
        "device_key": device_key,
        "total_scans_analyzed_14d": len(scans),
        "correlations": correlations
    }
