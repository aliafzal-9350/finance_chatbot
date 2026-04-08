"""
Configuration for AI Finance Control Center
Secure environment-based configuration
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root / parent fallback
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


@dataclass(frozen=True)
class Config:
    # -------------------------
    # MongoDB
    # -------------------------
    MONGO_URI: str = os.getenv("MONGODB_URI", "").strip()
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "Zatca").strip()

    # Default collections (used as fallback only)
    INVOICE_COLLECTION: str = os.getenv("INVOICE_COLLECTION", "ZATCA_API_INVOICES_ALL").strip()
    LINE_ITEMS_COLLECTION: str = os.getenv("LINE_ITEMS_COLLECTION", "ZATCA_API_INVOICES_LINES_ALL").strip()

    # -------------------------
    # App / Display
    # -------------------------
    CURRENCY: str = os.getenv("CURRENCY", "SAR").strip()

    # -------------------------
    # Validation
    # -------------------------
    @staticmethod
    def validate() -> bool:
        if not Config.MONGO_URI:
            raise ValueError("MONGODB_URI is missing. Set it in your .env file.")
        if not Config.DATABASE_NAME:
            raise ValueError("DATABASE_NAME is missing.")
        if not Config.INVOICE_COLLECTION:
            raise ValueError("INVOICE_COLLECTION is missing.")
        if not Config.LINE_ITEMS_COLLECTION:
            raise ValueError("LINE_ITEMS_COLLECTION is missing.")
        return True