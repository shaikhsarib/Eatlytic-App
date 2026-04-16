"""
app/routes/benchmarks.py
Accuracy benchmarking endpoints.
"""

import json
import logging
import asyncio
from fastapi import APIRouter, Request, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from app.models.db import db_conn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def _compute_field_accuracy(llm_output: dict, ground_truth: dict) -> dict:
    """
    Compare LLM-extracted nutrient values against hand-verified ground truth.
    Returns per-field accuracy within a tolerance band.
    """
    fields = ["calories", "protein", "carbs", "fat", "sodium", "fiber", "sugar"]
    results = {}
    exact_ct = 0

    llm_nutr = {}
    for n in llm_output.get("nutrient_breakdown", []):
        key = n.get("name", "").lower()
        val = n.get("value", 0)
        if isinstance(val, (int, float)):
            llm_nutr[key] = val

    gt_nutr = ground_truth.get("nutrients", {})

    for field in fields:
        llm_val = None
        for k, v in llm_nutr.items():
            if field in k or k in field:
                llm_val = v
                break

        gt_val = gt_nutr.get(field)
        if gt_val is None or llm_val is None:
            results[field] = {"status": "missing", "llm": llm_val, "truth": gt_val}
            continue

        tolerance = max(abs(gt_val) * 0.15, 2)
        correct = abs(llm_val - gt_val) <= tolerance
        if correct:
            exact_ct += 1
        results[field] = {
            "status": "correct" if correct else "wrong",
            "llm": llm_val,
            "truth": gt_val,
            "delta": round(llm_val - gt_val, 2),
            "pct_err": round(abs(llm_val - gt_val) / max(gt_val, 1) * 100, 1),
        }

    gt_score = ground_truth.get("score")
    llm_score = llm_output.get("score")
    if gt_score is not None and llm_score is not None:
        score_delta = abs(llm_score - gt_score)
        results["score"] = {
            "status": "correct" if score_delta <= 1 else "wrong",
            "llm": llm_score,
            "truth": gt_score,
            "delta": llm_score - gt_score,
        }

    accuracy_pct = round(exact_ct / len(fields) * 100, 1)
    return {"fields": results, "field_accuracy_pct": accuracy_pct}


def _word_f1(pred: str, truth: str) -> float:
    if not truth:
        return 0.0
    pw = set(pred.lower().split())
    tw = set(truth.lower().split())
    tp = len(pw & tw)
    pr = tp / len(pw) if pw else 0
    rc = tp / len(tw) if tw else 0
    return round(2 * pr * rc / (pr + rc), 3) if (pr + rc) else 0.0


def _compute_ocr_f1(extracted_text: str, ground_truth_text: str) -> float:
    return _word_f1(extracted_text, ground_truth_text)


def _get_ocr_service():
    from app.services.ocr import run_ocr

    return run_ocr


def _get_image_service():
    from app.services.image import (
        validate_image,
        assess_image_quality,
        deblur_and_enhance,
        ocr_quality_score,
    )

    return validate_image, assess_image_quality, deblur_and_enhance, ocr_quality_score


def _get_llm_service():
    from app.services.llm import unified_analyze_flow

    return unified_analyze_flow


@router.post("/submit-ground-truth")
async def submit_ground_truth(
    request: Request,
    product_name: str = Form(...),
    admin_token: str = Form(...),
    nutrients: str = Form(...),
    score: int = Form(...),
    ingredients: str = Form(""),
    barcode: str = Form(""),
):
    import os

    _expected = os.environ.get("ADMIN_TOKEN")
    if not _expected:
        raise HTTPException(
            status_code=500, detail="Server misconfiguration: ADMIN_TOKEN not set."
        )
    import hmac as _hmac_mod
    if not _hmac_mod.compare_digest(admin_token, _expected):
        raise HTTPException(status_code=403, detail="Invalid admin token")

    try:
        gt = json.loads(nutrients)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="nutrients must be valid JSON")

    ground_truth = {"nutrients": gt, "score": score, "ingredients": ingredients}

    with db_conn() as conn:
        conn.execute(
            """INSERT INTO benchmarks(product_name, ground_truth_json)
               VALUES(?,?)""",
            (product_name, json.dumps(ground_truth)),
        )
    return JSONResponse(
        {
            "registered": True,
            "product": product_name,
            "message": "Ground truth saved. Run /benchmarks/run to test.",
        }
    )


@router.post("/run/{benchmark_id}")
async def run_benchmark(
    request: Request,
    benchmark_id: int,
    image: UploadFile = File(...),
    admin_token: str = Form(...),
):
    import os

    _expected = os.environ.get("ADMIN_TOKEN")
    if not _expected:
        raise HTTPException(
            status_code=500, detail="Server misconfiguration: ADMIN_TOKEN not set."
        )
    import hmac as _hmac_mod
    if not _hmac_mod.compare_digest(admin_token, _expected):
        raise HTTPException(status_code=403, detail="Invalid admin token")

    with db_conn() as conn:
        bm_row = conn.execute(
            "SELECT * FROM benchmarks WHERE id=?", (benchmark_id,)
        ).fetchone()
    if not bm_row:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    ground_truth = json.loads(bm_row["ground_truth_json"])
    content = await image.read()
    validate_img, assess_quality, deblur, ocr_score_fn = _get_image_service()
    content = validate_img(content)
    quality = assess_quality(content)

    run_ocr = _get_ocr_service()
    working = content
    if quality["is_blurry"]:
        try:
            enhanced, _ = deblur(content, quality["blur_severity"])
            if (
                ocr_score_fn(run_ocr(enhanced, "en"))
                >= ocr_score_fn(run_ocr(content, "en")) * 0.85
            ):
                working = enhanced
        except Exception:
            pass

    ocr_result = run_ocr(working, "en")
    extracted_text = ocr_result["text"]
    ocr_f1 = _compute_ocr_f1(extracted_text, ground_truth.get("ingredients", ""))

    blur_info = {
        "detected": quality["is_blurry"],
        "severity": quality["blur_severity"],
        "score": quality["blur_score"],
        "deblurred": working != content,
    }

    analyse_label = _get_llm_service()
    llm_output = await analyse_label(
        extracted_text=extracted_text,
        persona="General Adult",
        age_group="adult",
        product_category_hint="general",
        language="en",
        web_context="",
        blur_info=blur_info,
        label_confidence="high",
        front_text="",
    )

    field_acc = _compute_field_accuracy(llm_output, ground_truth)

    with db_conn() as conn:
        conn.execute(
            """UPDATE benchmarks
               SET ocr_text=?, llm_output_json=?, f1_score=?,
                   score_delta=?, field_accuracy=?, tested_at=datetime('now'),
                   model_used='llama-3.3-70b'
               WHERE id=?""",
            (
                extracted_text,
                json.dumps(llm_output),
                ocr_f1,
                llm_output.get("score", 0) - ground_truth.get("score", 0),
                json.dumps(field_acc),
                benchmark_id,
            ),
        )

    return JSONResponse(
        {
            "benchmark_id": benchmark_id,
            "product_name": bm_row["product_name"],
            "ocr_f1": ocr_f1,
            "score_predicted": llm_output.get("score"),
            "score_truth": ground_truth.get("score"),
            "score_delta": llm_output.get("score", 0) - ground_truth.get("score", 0),
            "field_accuracy_pct": field_acc["field_accuracy_pct"],
            "fields": field_acc["fields"],
        }
    )


@router.get("/report")
async def accuracy_report(request: Request):
    with db_conn() as conn:
        rows = conn.execute(
            """SELECT product_name, f1_score, score_delta, field_accuracy, tested_at
               FROM benchmarks WHERE f1_score > 0 ORDER BY tested_at DESC"""
        ).fetchall()

    if not rows:
        return JSONResponse(
            {
                "message": "No benchmarks run yet.",
                "action": "POST /benchmarks/submit-ground-truth to register products, then POST /benchmarks/run/{id}",
                "products_tested": 0,
            }
        )

    f1_scores = [r["f1_score"] for r in rows if r["f1_score"]]
    score_deltas = [abs(r["score_delta"]) for r in rows if r["score_delta"] is not None]

    field_accs = []
    for r in rows:
        try:
            fa = json.loads(r["field_accuracy"] or "{}")
            pct = fa.get("field_accuracy_pct")
            if pct is not None:
                field_accs.append(pct)
        except Exception:
            pass

    return JSONResponse(
        {
            "products_tested": len(rows),
            "avg_ocr_f1": round(sum(f1_scores) / len(f1_scores), 3) if f1_scores else 0,
            "avg_score_delta": round(sum(score_deltas) / len(score_deltas), 2)
            if score_deltas
            else 0,
            "avg_field_accuracy": f"{round(sum(field_accs) / len(field_accs), 1)}%"
            if field_accs
            else "N/A",
            "results": [
                {
                    "product": r["product_name"],
                    "ocr_f1": r["f1_score"],
                    "score_delta": r["score_delta"],
                    "tested_at": r["tested_at"],
                }
                for r in rows
            ],
        }
    )
