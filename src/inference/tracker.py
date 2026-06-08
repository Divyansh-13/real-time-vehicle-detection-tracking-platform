"""
Vehicle Tracker — ByteTrack Integration
==========================================
Multi-object tracker that assigns persistent IDs to detected vehicles
across video frames using Ultralytics' built-in ByteTrack implementation.

How ByteTrack works (simplified):
    1. High-confidence detections → match with existing tracks via IoU
    2. Unmatched tracks → try matching with low-confidence detections
    3. Still unmatched detections → initialize as new tracks
    4. Tracks without matches for N frames → mark as lost/dead

Why ByteTrack over DeepSORT:
    - No separate ReID model needed (faster, simpler)
    - Uses detection confidence as a signal (two-stage association)
    - Better at handling crowded scenes
    - Native Ultralytics integration

Usage:
    tracker = VehicleTracker("models/vehicle_detector.pt")
    for frame in video_frames:
        result = tracker.track(frame)
        for det in result.detections:
            print(f"Vehicle {det.track_id}: {det.class_name} at {det.bbox}")
    summary = tracker.get_summary()
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .detector import Detection, DetectionResult

logger = logging.getLogger(__name__)


class TrackInfo:
    """Metadata about a single tracked object."""

    def __init__(self, track_id: int, class_name: str):
        self.track_id = track_id
        self.class_name = class_name
        self.first_frame: int = 0
        self.last_frame: int = 0
        self.total_frames: int = 0
        self.positions: List[tuple] = []  # List of (x_center, y_center)
        self.confidences: List[float] = []

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def travel_distance(self) -> float:
        """Total pixel distance traveled by this track."""
        if len(self.positions) < 2:
            return 0.0
        dist = 0.0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            dist += (dx ** 2 + dy ** 2) ** 0.5
        return dist

    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'class_name': self.class_name,
            'first_frame': self.first_frame,
            'last_frame': self.last_frame,
            'total_frames': self.total_frames,
            'avg_confidence': round(self.avg_confidence, 4),
            'travel_distance': round(self.travel_distance, 1),
        }


class TrackingSummary:
    """Summary of all tracking results across a video."""

    def __init__(self):
        self.total_frames: int = 0
        self.total_detections: int = 0
        self.unique_tracks: int = 0
        self.tracks: Dict[int, TrackInfo] = {}
        self.class_counts: Dict[str, int] = {}
        self.total_time: float = 0.0

    @property
    def avg_fps(self) -> float:
        return self.total_frames / self.total_time if self.total_time > 0 else 0.0

    @property
    def avg_detections_per_frame(self) -> float:
        return self.total_detections / self.total_frames if self.total_frames > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'unique_vehicles': self.unique_tracks,
            'class_counts': self.class_counts,
            'avg_fps': round(self.avg_fps, 1),
            'avg_detections_per_frame': round(self.avg_detections_per_frame, 1),
            'total_time_seconds': round(self.total_time, 2),
            'tracks': {tid: t.to_dict() for tid, t in self.tracks.items()},
        }


class VehicleTracker:
    """
    Multi-object vehicle tracker using YOLOv8 + ByteTrack.

    Uses Ultralytics' built-in tracking which combines YOLOv8 detection
    with ByteTrack for multi-object tracking in a single call.

    State management:
    - Maintains a TrackingSummary across calls to track()
    - Each call updates track histories
    - Call get_summary() to retrieve aggregated tracking statistics
    - Call reset() between different videos/streams
    """

    CLASS_NAMES = {0: 'car', 1: 'minivan'}

    def __init__(
        self,
        model_path: str = 'models/vehicle_detector.pt',
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: Optional[str] = None,
        tracker_type: str = 'bytetrack',
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        track_buffer: int = 30,
    ):
        """
        Args:
            model_path: Path to YOLOv8 model.
            confidence: Detection confidence threshold.
            iou_threshold: NMS IoU threshold.
            img_size: Model input size.
            device: Compute device.
            tracker_type: 'bytetrack' or 'botsort'.
            track_high_thresh: High confidence threshold for first association.
            track_low_thresh: Low confidence threshold for second association.
            track_buffer: Frames to keep lost tracks alive.
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.tracker_type = tracker_type
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_buffer = track_buffer

        self._model = None
        self._frame_count = 0
        self._summary = TrackingSummary()

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        from ultralytics import YOLO

        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}, using pretrained yolov8n.pt")
            self._model = YOLO('yolov8n.pt')
        else:
            self._model = YOLO(str(self.model_path))
        logger.info(f"Tracker model loaded: {self.model_path}")

    def track(
        self,
        frame: Union[str, np.ndarray],
        persist: bool = True,
    ) -> DetectionResult:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: Image path or numpy array (BGR).
            persist: Persist tracks across calls (True for video, False for single images).

        Returns:
            DetectionResult with track_id assigned to each detection.
        """
        start_time = time.time()

        results = self.model.track(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            persist=persist,
            tracker=f"{self.tracker_type}.yaml",
            verbose=False,
        )

        inference_time = time.time() - start_time
        self._frame_count += 1

        # Parse results
        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf_score = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Get track ID
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                # Map class name
                if hasattr(result, 'names') and result.names:
                    class_name = result.names.get(cls_id, self.CLASS_NAMES.get(cls_id, f'class_{cls_id}'))
                else:
                    class_name = self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')

                det = Detection(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf_score,
                    class_id=cls_id,
                    class_name=class_name,
                    track_id=track_id,
                )
                detections.append(det)

                # Update tracking summary
                if track_id is not None:
                    self._update_track(track_id, class_name, det, self._frame_count)

        img_shape = result.orig_shape if hasattr(result, 'orig_shape') else (0, 0)

        det_result = DetectionResult(
            detections=detections,
            inference_time=inference_time,
            image_shape=img_shape,
            frame_id=self._frame_count,
        )

        # Update summary
        self._summary.total_frames = self._frame_count
        self._summary.total_detections += len(detections)
        self._summary.total_time += inference_time

        return det_result

    def _update_track(self, track_id: int, class_name: str,
                      detection: Detection, frame_num: int):
        """Update tracking metadata for a specific track."""
        if track_id not in self._summary.tracks:
            self._summary.tracks[track_id] = TrackInfo(track_id, class_name)
            self._summary.tracks[track_id].first_frame = frame_num
            self._summary.unique_tracks += 1
            self._summary.class_counts[class_name] = \
                self._summary.class_counts.get(class_name, 0) + 1

        track = self._summary.tracks[track_id]
        track.last_frame = frame_num
        track.total_frames += 1
        track.positions.append(detection.center)
        track.confidences.append(detection.confidence)

    def get_summary(self) -> TrackingSummary:
        """Get the accumulated tracking summary."""
        return self._summary

    def reset(self):
        """Reset tracking state for a new video/stream."""
        self._frame_count = 0
        self._summary = TrackingSummary()
        # Reset the model's internal tracker state
        if self._model is not None:
            self._model.predictor = None
        logger.info("Tracker state reset")
