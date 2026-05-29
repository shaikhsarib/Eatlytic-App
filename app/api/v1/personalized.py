"""
app/routes/personalized.py
Stateless API router managing personalized metabolic genomics and biomarkers.
"""

import logging
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional
from app.database.connection import save_genomic_profile, get_genomic_profile
from app.services.brain import EatlyticBrain
from app.core.security import get_device_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/personalized", tags=["personalized"])

class ProfileUploadRequest(BaseModel):
    genetic_snps: Dict[str, str] = Field(default_factory=dict, description="Map of rsIDs to alleles e.g. {'rs7903146': 'TT'}")
    biomarkers: Dict[str, float] = Field(default_factory=dict, description="Map of biomarker names to values e.g. {'hba1c': 5.8}")

class PersonalizedScanRequest(BaseModel):
    product_name: str = "Test Product"
    brand: str = "Test Brand"
    category: str = "general"
    nutrients: Dict[str, float] = Field(default_factory=dict)
    ingredients_raw: str = ""
    persona: str = "diabetic"

@router.post("/profile")
async def upload_personalized_profile(request: Request, response: Response, payload: ProfileUploadRequest):
    """
    Statelessly saves genomic SNPs and clinical biomarker profiles for the device context.
    Strictly private—no login or credentials required.
    """
    device_key = get_device_key(request, response)
    if not device_key:
        raise HTTPException(status_code=400, detail="Missing device_key Context")
    
    try:
        # Standardize allele values to uppercase
        snps = {k.strip().lower(): v.strip().upper() for k, v in payload.genetic_snps.items()}
        biomarkers = {k.strip().lower(): float(v) for k, v in payload.biomarkers.items()}
        
        save_genomic_profile(device_key, snps, biomarkers)
        logger.info(f"Genomic profile updated statelessly for device_key: {device_key}")
        
        return JSONResponse({
            "status": "success",
            "message": "DNA SNPs and biomarkers registered statelessly.",
            "device_key": device_key
        })
    except Exception as e:
        logger.error(f"Failed to save genomic profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record profile: {str(e)}")

@router.get("/profile")
async def retrieve_personalized_profile(request: Request, response: Response):
    """
    Retrieves the registered genomic SNPs and biomarker readings for the device context.
    """
    device_key = get_device_key(request, response)
    if not device_key:
        raise HTTPException(status_code=400, detail="Missing device_key Context")
    
    profile = get_genomic_profile(device_key)
    if not profile:
        return JSONResponse({
            "device_key": device_key,
            "genetic_snps": {},
            "biomarkers": {}
        })
        
    return JSONResponse(profile)

@router.post("/scan")
async def evaluate_personalized_scan(request: Request, response: Response, payload: PersonalizedScanRequest):
    """
    Processes a localized scan matching against the device's genomic/biomarker profile.
    """
    device_key = get_device_key(request, response)
    brain = EatlyticBrain()
    
    try:
        report = brain.compile_local_report(
            product_name=payload.product_name,
            brand=payload.brand,
            category=payload.category,
            nutrients=payload.nutrients,
            ingredients_raw=payload.ingredients_raw,
            persona=payload.persona,
            device_key=device_key
        )
        return JSONResponse(report)
    except Exception as e:
        logger.error(f"Personalized scan compilation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnostic compilation failed: {str(e)}")
