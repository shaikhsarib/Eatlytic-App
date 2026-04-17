"""
app/services/hash_service.py
Perceptual image hashing to identify visually similar product labels.
"""

import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import imagehash
    _HASH_AVAILABLE = True
except ImportError:
    _HASH_AVAILABLE = False
    logger.warning("ImageHash not installed — perceptual caching disabled.")


def get_image_fingerprint(content: bytes) -> str:
    """
    Generate a perceptual hash (pHash) for the image.
    If the library is missing, falls back to a dummy string (caching disabled).
    """
    if not _HASH_AVAILABLE:
        return ""
    
    try:
        img = Image.open(BytesIO(content)).convert("L")  # Grayscale for hashing
        # pHash is robust to resizing and small rotations
        phash = imagehash.phash(img)
        return str(phash)
    except Exception as e:
        logger.error("Failed to generate pHash: %s", e)
        return ""


def calculate_hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate the Hamming distance between two pHash strings.
    A distance <= 4 usually means it's the same product.
    """
    if not hash1 or not hash2:
        return 999
    
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except Exception:
        return 999
