import sys
import os

# Ensure code/ directory is on path so `from app.xxx` works in tests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PIL compatibility patch (Pillow 10+ removed ANTIALIAS in favour of LANCZOS)
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

