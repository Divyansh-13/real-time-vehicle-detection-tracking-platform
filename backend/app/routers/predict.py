"""
Prediction Router — Core detection and tracking endpoints.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import DetectionRecord, VideoRecord
from ..schemas import DetectionResponse, VideoTrackingResponse, BBoxResponse, TrackResponse

router = APIRouter()


def _get_pipeline():
    """Get the detection pipeline."""
    from ..main import get_pipeline
    return get_pipeline()


@router.post("/image", response_model=DetectionResponse)
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=1.0),
    db: Session = Depends(get_db),
):
    """
    Upload an image and run vehicle detection.

    Returns annotated image URL and detection details.
    """
    # Validate file type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Invalid image type: {file.content_type}")

    # Read image bytes
    image_bytes = await file.read()

    # Run detection
    pipeline = _get_pipeline()
    result, annotated_bytes = pipeline.process_image_bytes(image_bytes, draw=True)

    # Save results
    result_filename = f"{uuid.uuid4().hex}_result.jpg"
    result_dir = Path(settings.RESULTS_DIR) / "images"
    result_dir.mkdir(parents=True, exist_ok=True)

    if annotated_bytes:
        with open(result_dir / result_filename, "wb") as f:
            f.write(annotated_bytes)

    # Save source image
    source_filename = f"{uuid.uuid4().hex}_{file.filename}"
    source_dir = Path(settings.UPLOAD_DIR) / "images"
    source_dir.mkdir(parents=True, exist_ok=True)
    with open(source_dir / source_filename, "wb") as f:
        f.write(image_bytes)

    # Save to database
    class_counts = result.class_counts
    record = DetectionRecord(
        source_type="image",
        source_filename=source_filename,
        result_filename=result_filename,
        total_objects=result.count,
        car_count=class_counts.get("car", 0),
        minivan_count=class_counts.get("minivan", 0),
        avg_confidence=sum(d.confidence for d in result.detections) / max(len(result.detections), 1),
        inference_time_ms=result.inference_time * 1000,
        detections_json=result.to_dict(),
        status="completed",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return DetectionResponse(
        id=record.id,
        source_type="image",
        source_filename=file.filename,
        result_filename=result_filename,
        result_url=f"/results/images/{result_filename}",
        detections=[BBoxResponse(**d.to_dict()) for d in result.detections],
        count=result.count,
        class_counts=result.class_counts,
        inference_time_ms=round(result.inference_time * 1000, 2),
        fps=round(result.fps, 1),
        image_shape=list(result.image_shape),
        created_at=record.created_at,
    )


@router.post("/video", response_model=VideoTrackingResponse)
async def predict_video(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=1.0),
    max_frames: Optional[int] = Query(None, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    """
    Upload a video and run vehicle detection + tracking.

    Returns annotated video URL and tracking details.
    """
    if file.content_type not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Invalid video type: {file.content_type}")

    # Save uploaded video
    video_filename = f"{uuid.uuid4().hex}_{file.filename}"
    video_dir = Path(settings.UPLOAD_DIR) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / video_filename

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Process video
    pipeline = _get_pipeline()
    result_filename = f"{uuid.uuid4().hex}_tracked.mp4"
    result_dir = Path(settings.RESULTS_DIR) / "videos"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = str(result_dir / result_filename)

    all_results, summary, _ = pipeline.process_video(
        str(video_path), result_path, max_frames=max_frames
    )

    # Save to database
    record = VideoRecord(
        filename=video_filename,
        result_filename=result_filename,
        frame_count=summary.total_frames,
        total_detections=summary.total_detections,
        unique_vehicles=summary.unique_tracks,
        avg_fps=summary.avg_fps,
        tracking_summary=summary.to_dict(),
        status="completed",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Build tracks response
    tracks = [
        TrackResponse(**t.to_dict())
        for t in summary.tracks.values()
    ]

    return VideoTrackingResponse(
        id=record.id,
        filename=file.filename,
        result_filename=result_filename,
        result_url=f"/results/videos/{result_filename}",
        total_frames=summary.total_frames,
        total_detections=summary.total_detections,
        unique_vehicles=summary.unique_tracks,
        class_counts=summary.class_counts,
        avg_fps=round(summary.avg_fps, 1),
        total_time_seconds=round(summary.total_time, 2),
        tracks=tracks,
        status="completed",
        created_at=record.created_at,
    )


@router.get("/{detection_id}", response_model=DetectionResponse)
async def get_detection(detection_id: int, db: Session = Depends(get_db)):
    """Get a specific detection result by ID."""
    record = db.query(DetectionRecord).filter(DetectionRecord.id == detection_id).first()
    if not record:
        raise HTTPException(404, "Detection not found")

    # Reconstruct response from stored JSON
    detections_data = record.detections_json or {}
    det_list = detections_data.get("detections", [])

    return DetectionResponse(
        id=record.id,
        source_type=record.source_type,
        source_filename=record.source_filename,
        result_filename=record.result_filename,
        result_url=f"/results/images/{record.result_filename}" if record.result_filename else None,
        detections=[BBoxResponse(**d) for d in det_list],
        count=record.total_objects,
        class_counts={"car": record.car_count, "minivan": record.minivan_count},
        inference_time_ms=record.inference_time_ms,
        fps=1000 / record.inference_time_ms if record.inference_time_ms > 0 else 0,
        image_shape=detections_data.get("image_shape", [0, 0]),
        model_version=record.model_version,
        created_at=record.created_at,
    )


@router.get("/", response_model=dict)
async def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """List all detection records with pagination."""
    query = db.query(DetectionRecord)
    if source_type:
        query = query.filter(DetectionRecord.source_type == source_type)

    total = query.count()
    records = query.order_by(DetectionRecord.created_at.desc()) \
                   .offset((page - 1) * page_size) \
                   .limit(page_size) \
                   .all()

    items = []
    for r in records:
        items.append({
            "id": r.id,
            "source_type": r.source_type,
            "source_filename": r.source_filename,
            "total_objects": r.total_objects,
            "car_count": r.car_count,
            "minivan_count": r.minivan_count,
            "avg_confidence": r.avg_confidence,
            "inference_time_ms": r.inference_time_ms,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }
