"""
Application Configuration
===========================
Centralized configuration using environment variables with sensible defaults.
Uses pydantic-settings for validation and type coercion.

Environment variables can be set in a .env file at the project root.
"""

import os
from pathlib import Path
from typing import List


class Settings:
    """Application settings loaded from environment variables."""

    # ── App ──
    APP_NAME: str = "Vehicle Detection & Tracking API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # ── Security ──
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-789xyz")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # ── Database ──
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./vehicle_tracking.db"  # SQLite for development
    )

    # ── Redis ──
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── CORS ──
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    # ── File Storage ──
    BASE_DIR: str = str(Path(__file__).parent.parent.parent.resolve())
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", os.path.join(BASE_DIR, "results"))
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(100 * 1024 * 1024)))  # 100MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/avi", "video/x-msvideo",
                                       "video/quicktime", "video/webm"]

    # ── Model ──
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/vehicle_detector.pt")
    DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    DETECTION_IOU: float = float(os.getenv("DETECTION_IOU", "0.45"))
    DETECTION_IMG_SIZE: int = int(os.getenv("DETECTION_IMG_SIZE", "640"))
    PRELOAD_MODEL: bool = os.getenv("PRELOAD_MODEL", "false").lower() == "true"

    # ── Rate Limiting ──
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))


settings = Settings()
