"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check — returns 200 if the API is running."""
    return {
        "status": "healthy",
        "service": "Vehicle Detection & Tracking API",
        "version": "1.0.0",
    }


@router.get("/health/ready")
async def readiness_check():
    """Readiness check — verifies all dependencies are available."""
    checks = {"api": True, "model": False, "database": False}

    # Check model
    try:
        from ..main import get_pipeline
        pipeline = get_pipeline()
        checks["model"] = pipeline is not None
    except Exception:
        pass

    # Check database
    try:
        from ..database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1" if hasattr(db, 'execute') else None)
        checks["database"] = True
        db.close()
    except Exception:
        pass

    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }
