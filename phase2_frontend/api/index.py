
"""Vercel entry point for the DataGuard AI FastAPI backend."""
import sys
from pathlib import Path

# Allow imports from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase2_backend.main import app
