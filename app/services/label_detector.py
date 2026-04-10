"""
app/services/label_detector.py
OpenCV-based ROI detection, deskewing, and YOLO integration.
"""

import cv2
import numpy as np
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

def get_nutrition_table_roi(image_np: np.ndarray) -> np.ndarray:
    """
    Finds high-density text regions surrounded by table borders (ROI).
    Heuristic: Look for rectangular contours with a certain aspect ratio.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Adaptive thresholding to handle different lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Dilate to connect text lines into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        # Filter contours by size and aspect ratio
        roi = image_np
        max_area = 0
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect_ratio = w / float(h)
            
            # Nutrition tables are usually wider than 0.5 and height is significant
            if area > 800 and 0.2 < aspect_ratio < 5.0:
                if area > max_area:
                    max_area = area
                    roi = image_np[y:y+h, x:x+w]
        
        return roi
    except Exception as e:
        logger.warning(f"ROI detection failed: {e}. Falling back to full image.")
        return image_np

def deskew_image(image_np: np.ndarray) -> np.ndarray:
    """
    Automatically rotate the image to be horizontal.
    Uses text baseline detection via minAreaRect.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)
        
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Find all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle for OpenCV 4.5+ (angle range is 0-90)
        # or handle older formats by normalizing to +/- 45
        if angle > 45:
            angle = angle - 90
        
        # Final rotation angle
        rotation_angle = -angle
            
        (h, w) = image_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    except Exception as e:
        logger.warning(f"Deskewing failed: {e}")
        return image_np

def process_image_for_ocr(content: bytes) -> bytes:
    """
    Orchestrates the image pipeline: ROI -> Deskew.
    Returns bytes of the processed image.
    """
    try:
        # Load image
        img = Image.open(BytesIO(content)).convert("RGB")
        img_np = np.array(img)
        
        # 1. ROI Detection
        roi = get_nutrition_table_roi(img_np)
        
        # 2. Deskew
        processed = deskew_image(roi)
        
        # Convert back to bytes
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        if is_success:
            return buffer.tobytes()
        return content
    except Exception as e:
        logger.error(f"Image pipeline failed: {e}")
        return content

def yolov8_table_detector_stub(image_np: np.ndarray):
    """
    Placeholder for Ultralytics YOLOv8 integration.
    To use: 
    1. pip install ultralytics
    2. from ultralytics import YOLO
    3. model = YOLO('path/to/best.pt')
    4. results = model(image_np)
    """
    pass
