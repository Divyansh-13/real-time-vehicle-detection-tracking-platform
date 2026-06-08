"""SQLAlchemy ORM models for the vehicle tracking platform."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from ..database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), default="")
    role = Column(String(50), default="user")  # admin, user, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    detections = relationship("DetectionRecord", back_populates="user")


class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    source_type = Column(String(20), nullable=False)  # image, video, stream
    source_filename = Column(String(500), default="")
    result_filename = Column(String(500), default="")
    total_objects = Column(Integer, default=0)
    car_count = Column(Integer, default=0)
    minivan_count = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    inference_time_ms = Column(Float, default=0.0)
    model_version = Column(String(100), default="")
    detections_json = Column(JSON, nullable=True)  # Full detection details
    status = Column(String(20), default="completed")  # processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="detections")


class VideoRecord(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    filename = Column(String(500), nullable=False)
    result_filename = Column(String(500), default="")
    duration_seconds = Column(Float, default=0.0)
    frame_count = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    unique_vehicles = Column(Integer, default=0)
    avg_fps = Column(Float, default=0.0)
    tracking_summary = Column(JSON, nullable=True)
    status = Column(String(20), default="processing")
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(100), default="1.0")
    file_path = Column(String(500), nullable=False)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    num_classes = Column(Integer, default=2)
    class_names = Column(JSON, default=["car", "minivan"])
    is_active = Column(Boolean, default=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalyticsRecord(Base):
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    total_detections = Column(Integer, default=0)
    car_count = Column(Integer, default=0)
    minivan_count = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    total_images_processed = Column(Integer, default=0)
    total_videos_processed = Column(Integer, default=0)
