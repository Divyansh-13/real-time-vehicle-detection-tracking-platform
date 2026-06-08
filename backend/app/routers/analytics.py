"""
Analytics Router — Dashboard statistics and historical data.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import DetectionRecord, VideoRecord
from ..schemas import AnalyticsSummary

router = APIRouter()


@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(db: Session = Depends(get_db)):
    """Get overall analytics summary for the dashboard."""
    # Image detection stats
    image_stats = db.query(
        func.count(DetectionRecord.id).label("total"),
        func.sum(DetectionRecord.total_objects).label("total_objects"),
        func.sum(DetectionRecord.car_count).label("cars"),
        func.sum(DetectionRecord.minivan_count).label("minivans"),
        func.avg(DetectionRecord.avg_confidence).label("avg_conf"),
        func.avg(DetectionRecord.inference_time_ms).label("avg_time"),
    ).first()

    # Video stats
    video_stats = db.query(
        func.count(VideoRecord.id).label("total"),
        func.sum(VideoRecord.unique_vehicles).label("unique_vehicles"),
    ).first()

    return AnalyticsSummary(
        total_images_processed=image_stats.total or 0,
        total_videos_processed=video_stats.total or 0,
        total_detections=int(image_stats.total_objects or 0),
        total_unique_vehicles=int(video_stats.unique_vehicles or 0),
        car_count=int(image_stats.cars or 0),
        minivan_count=int(image_stats.minivans or 0),
        avg_confidence=round(float(image_stats.avg_conf or 0), 4),
        avg_inference_time_ms=round(float(image_stats.avg_time or 0), 2),
    )


@router.get("/timeline")
async def get_timeline(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get detection timeline grouped by date."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    records = db.query(
        func.date(DetectionRecord.created_at).label("date"),
        func.sum(DetectionRecord.total_objects).label("detections"),
        func.sum(DetectionRecord.car_count).label("cars"),
        func.sum(DetectionRecord.minivan_count).label("minivans"),
        func.count(DetectionRecord.id).label("num_images"),
    ).filter(
        DetectionRecord.created_at >= cutoff
    ).group_by(
        func.date(DetectionRecord.created_at)
    ).order_by("date").all()

    entries = []
    for r in records:
        entries.append({
            "date": str(r.date),
            "detections": int(r.detections or 0),
            "cars": int(r.cars or 0),
            "minivans": int(r.minivans or 0),
            "images_processed": int(r.num_images or 0),
        })

    return {
        "entries": entries,
        "period": f"last_{days}_days",
    }


@router.get("/classes")
async def get_class_distribution(db: Session = Depends(get_db)):
    """Get class distribution across all detections."""
    stats = db.query(
        func.sum(DetectionRecord.car_count).label("cars"),
        func.sum(DetectionRecord.minivan_count).label("minivans"),
    ).first()

    cars = int(stats.cars or 0)
    minivans = int(stats.minivans or 0)
    total = cars + minivans

    return {
        "classes": [
            {"name": "car", "count": cars, "percentage": round(cars / max(total, 1) * 100, 1)},
            {"name": "minivan", "count": minivans, "percentage": round(minivans / max(total, 1) * 100, 1)},
        ],
        "total": total,
    }


@router.get("/recent")
async def get_recent_detections(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get most recent detection records."""
    records = db.query(DetectionRecord) \
        .order_by(DetectionRecord.created_at.desc()) \
        .limit(limit) \
        .all()

    return [{
        "id": r.id,
        "source_type": r.source_type,
        "source_filename": r.source_filename,
        "total_objects": r.total_objects,
        "car_count": r.car_count,
        "minivan_count": r.minivan_count,
        "avg_confidence": round(r.avg_confidence, 3),
        "inference_time_ms": round(r.inference_time_ms, 1),
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in records]
