"""
ASGI entrypoint for Uvicorn.

Run from project root:
  uvicorn RestAPI:app --reload --host 0.0.0.0 --port 8000

Or use settings from .env (HOST/PORT) via:
  python -m uvicorn RestAPI:app --reload
"""

from app.main import app

__all__ = ["app"]
