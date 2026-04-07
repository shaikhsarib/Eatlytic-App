import sys
import os

# Ensure code/ directory is on path so `from app.xxx` works in tests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
