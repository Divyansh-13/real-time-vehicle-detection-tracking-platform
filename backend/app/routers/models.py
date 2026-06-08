"""
Models Router — Model management endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ModelRecord
from ..schemas import ModelResponse

router = APIRouter()


@router.get("/", response_model=List[ModelResponse])
async def list_models(db: Session = Depends(get_db)):
    """List all registered models."""
    models = db.query(ModelRecord).order_by(ModelRecord.created_at.desc()).all()
    return models


@router.get("/active", response_model=ModelResponse)
async def get_active_model(db: Session = Depends(get_db)):
    """Get the currently active model."""
    model = db.query(ModelRecord).filter(ModelRecord.is_active).first()
    if not model:
        raise HTTPException(404, "No active model found")
    return model


@router.post("/activate/{model_id}")
async def activate_model(model_id: int, db: Session = Depends(get_db)):
    """Set a model as the active detection model."""
    model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found")

    # Deactivate all models
    db.query(ModelRecord).update({ModelRecord.is_active: False})

    # Activate the selected model
    model.is_active = True
    db.commit()

    # Reload the detection pipeline with the new model
    try:
        import backend.app.main as main_module
        main_module.detection_pipeline = None  # Force reload
    except Exception:
        pass

    return {"message": f"Model '{model.name}' v{model.version} activated", "model_id": model.id}


@router.get("/info")
async def get_model_info():
    """Get info about the currently loaded model."""
    try:
        from ..main import get_pipeline
        pipeline = get_pipeline()
        return pipeline._detector.get_model_info()
    except Exception as e:
        return {
            "error": str(e),
            "message": "Model not loaded. Upload a model or train one first.",
        }
