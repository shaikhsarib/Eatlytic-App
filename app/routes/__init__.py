# Legacy compatibility redirect for routes module
import sys
from app.api import v1
sys.modules['app.routes'] = v1
