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


class BKTreeNode:
    def __init__(self, hash_key: str, result_json: str):
        self.hash_key = hash_key
        self.result_json = result_json
        self.children = {}  # distance -> BKTreeNode


class BKTree:
    def __init__(self):
        self.root = None

    def insert(self, hash_key: str, result_json: str):
        if not hash_key:
            return
        if not self.root:
            self.root = BKTreeNode(hash_key, result_json)
            return
        curr = self.root
        while True:
            dist = calculate_hamming_distance(hash_key, curr.hash_key)
            if dist == 0:
                curr.result_json = result_json  # Update matching node
                return
            if dist not in curr.children:
                curr.children[dist] = BKTreeNode(hash_key, result_json)
                return
            curr = curr.children[dist]

    def search(self, query_hash: str, max_distance: int = 6) -> list:
        if not self.root or not query_hash:
            return []
        results = []
        candidates = [self.root]
        while candidates:
            curr = candidates.pop()
            dist = calculate_hamming_distance(query_hash, curr.hash_key)
            if dist <= max_distance:
                results.append((dist, curr.result_json))
            
            # Prune step: only traverse children with distance in [dist - max, dist + max]
            low = max(0, dist - max_distance)
            high = dist + max_distance
            for d in curr.children:
                if low <= d <= high:
                    candidates.append(curr.children[d])
        return results

