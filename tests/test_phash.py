import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.services.hash_service import get_image_fingerprint, calculate_hamming_distance
from PIL import Image, ImageDraw
import io

def create_test_image(text="Test", color=(255, 255, 255), pattern=""):
    img = Image.new('RGB', (200, 201), color=color) # Slightly different sizes
    d = ImageDraw.Draw(img)
    d.text((10,10), text, fill=(0,0,0))
    if pattern == "rect":
        d.rectangle([50, 50, 150, 150], outline="red", width=5)
    elif pattern == "line":
        d.line([0, 0, 200, 200], fill="blue", width=10)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_hashing():
    print("Testing pHash consistency...")
    
    # 1. Base image
    img1 = create_test_image("Product A", (255, 255, 255), "rect")
    hash1 = get_image_fingerprint(img1)
    print(f"Hash 1: {hash1}")
    
    # 2. Same image, slightly different (light gray)
    img2 = create_test_image("Product A", (240, 240, 240), "rect")
    hash2 = get_image_fingerprint(img2)
    print(f"Hash 2: {hash2}")
    
    # 3. Different product
    img3 = create_test_image("Product B", (100, 100, 255), "line")
    hash3 = get_image_fingerprint(img3)
    print(f"Hash 3: {hash3}")
    
    dist_12 = calculate_hamming_distance(hash1, hash2)
    dist_13 = calculate_hamming_distance(hash1, hash3)
    
    print(f"Distance between similar images: {dist_12}")
    print(f"Distance between different images: {dist_13}")
    
    assert dist_12 < 5, "Similar images should have low Hamming distance"
    assert dist_13 > 10, "Different images should have high Hamming distance"
    print("✅ pHash Verification Passed!")

if __name__ == "__main__":
    test_hashing()
