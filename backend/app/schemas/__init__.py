"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field


# ── Auth Schemas ─────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(default="", max_length=255)

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str


# ── Detection Schemas ────────────────────────────────────────────

class BBoxResponse(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    width: float
    height: float
    area: float
    center: List[float]

class DetectionResponse(BaseModel):
    id: Optional[int] = None
    source_type: str
    source_filename: str
    result_filename: Optional[str] = None
    result_url: Optional[str] = None
    detections: List[BBoxResponse]
    count: int
    class_counts: Dict[str, int]
    inference_time_ms: float
    fps: float
    image_shape: List[int]
    model_version: str = ""
    created_at: Optional[datetime] = None

class DetectionListResponse(BaseModel):
    items: List[DetectionResponse]
    total: int
    page: int
    page_size: int


# ── Video/Tracking Schemas ───────────────────────────────────────

class TrackResponse(BaseModel):
    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    total_frames: int
    avg_confidence: float
    travel_distance: float

class VideoTrackingResponse(BaseModel):
    id: Optional[int] = None
    filename: str
    result_filename: Optional[str] = None
    result_url: Optional[str] = None
    total_frames: int
    total_detections: int
    unique_vehicles: int
    class_counts: Dict[str, int]
    avg_fps: float
    total_time_seconds: float
    tracks: List[TrackResponse]
    status: str = "completed"
    created_at: Optional[datetime] = None


# ── Analytics Schemas ────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_images_processed: int
    total_videos_processed: int
    total_detections: int
    total_unique_vehicles: int
    car_count: int
    minivan_count: int
    avg_confidence: float
    avg_inference_time_ms: float

class TimelineEntry(BaseModel):
    date: str
    detections: int
    cars: int
    minivans: int

class AnalyticsTimeline(BaseModel):
    entries: List[TimelineEntry]
    period: str


# ── Model Schemas ────────────────────────────────────────────────

class ModelResponse(BaseModel):
    id: int
    name: str
    version: str
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    num_classes: int
    class_names: List[str]
    is_active: bool
    description: str
    created_at: datetime
    class Config:
        from_attributes = True
