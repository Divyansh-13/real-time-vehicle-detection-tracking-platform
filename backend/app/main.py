"""
FastAPI Backend — Application Entry Point
============================================
Production-grade FastAPI application for the Vehicle Detection &
Tracking Platform. Serves as the REST API layer between the React
frontend and the ML inference engine.

Features:
    - JWT authentication with role-based access
    - File upload (image/video) with detection
    - Real-time WebSocket streaming
    - Detection history & analytics
    - Model management
    - Health checks & metrics
    - CORS middleware for frontend access

Run:
    uvicorn backend.app.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import engine, Base
from .routers import auth, upload, predict, analytics, health, models as models_router
from .routers import websocket as ws_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global reference to the detection pipeline (loaded once)
detection_pipeline = None


def get_pipeline():
    """Get or create the detection pipeline singleton."""
    global detection_pipeline
    if detection_pipeline is None:
        from src.inference.pipeline import DetectionPipeline
        model_path = settings.MODEL_PATH
        detection_pipeline = DetectionPipeline(
            model_path=model_path,
            confidence=settings.DETECTION_CONFIDENCE,
            iou_threshold=settings.DETECTION_IOU,
            img_size=settings.DETECTION_IMG_SIZE,
        )
        logger.info(f"Detection pipeline loaded with model: {model_path}")
    return detection_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    logger.info("🚗 Vehicle Tracking Platform API starting up...")

    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")

    # Create upload directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)

    # Pre-load model (optional — lazy load on first request)
    if settings.PRELOAD_MODEL:
        get_pipeline()

    # Inject pipeline into WebSocket router
    ws_router.set_pipeline(get_pipeline())

    logger.info("✅ Application ready!")
    yield

    # Shutdown
    logger.info("Application shutting down...")


# ── Create FastAPI Application ────────────────────────────────────

app = FastAPI(
    title="Vehicle Detection & Tracking API",
    description=(
        "Production-grade REST API for real-time vehicle detection, "
        "multi-object tracking, and traffic analytics. "
        "Built with YOLOv8, ByteTrack, FastAPI, and PostgreSQL."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["Upload"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(models_router.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(ws_router.router)  # WebSocket — no prefix (already set in router)

# ── Static Files ─────────────────────────────────────────────────

# Serve uploaded and result files
uploads_dir = Path(settings.UPLOAD_DIR)
results_dir = Path(settings.RESULTS_DIR)
uploads_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")
app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")


# ── Root Endpoint ────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Vehicle Detection & Tracking API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
