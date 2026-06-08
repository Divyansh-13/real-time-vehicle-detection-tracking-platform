"""
Upload Router — File upload handling for images and videos.
"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from ..config import settings

router = APIRouter()


def _validate_file(file: UploadFile, allowed_types: list, max_size: int = None):
    """Validate file type and size."""
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file.content_type}' not allowed. "
                   f"Allowed: {', '.join(allowed_types)}"
        )


def _save_upload(file: UploadFile, subdir: str = "") -> str:
    """Save uploaded file and return the saved filename."""
    ext = Path(file.filename).suffix or ".bin"
    unique_name = f"{uuid.uuid4().hex}{ext}"

    save_dir = Path(settings.UPLOAD_DIR) / subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / unique_name
    with open(save_path, "wb") as f:
        content = file.file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB"
            )
        f.write(content)

    return unique_name


@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file for detection."""
    _validate_file(file, settings.ALLOWED_IMAGE_TYPES)
    saved_name = _save_upload(file, "images")

    return {
        "filename": saved_name,
        "original_name": file.filename,
        "content_type": file.content_type,
        "url": f"/uploads/images/{saved_name}",
        "message": "Image uploaded successfully. Use /api/v1/predict/image to run detection.",
    }


@router.post("/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for tracking."""
    _validate_file(file, settings.ALLOWED_VIDEO_TYPES)
    saved_name = _save_upload(file, "videos")

    return {
        "filename": saved_name,
        "original_name": file.filename,
        "content_type": file.content_type,
        "url": f"/uploads/videos/{saved_name}",
        "message": "Video uploaded successfully. Use /api/v1/predict/video to run tracking.",
    }
