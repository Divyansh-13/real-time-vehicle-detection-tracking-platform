"""
Vehicle Detector — YOLOv8 Inference Wrapper
=============================================
Production-grade wrapper around the Ultralytics YOLOv8 model for
vehicle detection. Handles model loading, input preprocessing,
inference, and result formatting.

Supports:
    - PyTorch (.pt) models
    - ONNX (.onnx) models
    - Configurable confidence and IoU thresholds
    - Batch inference
    - GPU/CPU automatic device selection

Usage:
    detector = VehicleDetector("models/vehicle_detector.pt")
    detections = detector.detect("path/to/image.jpg")
    detections = detector.detect_batch(["img1.jpg", "img2.jpg"])
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class Detection:
    """Single detection result."""

    __slots__ = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'class_name', 'track_id']

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 confidence: float, class_id: int, class_name: str,
                 track_id: Optional[int] = None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = track_id

    @property
    def bbox(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox,
            'confidence': round(self.confidence, 4),
            'class_id': self.class_id,
            'class_name': self.class_name,
            'track_id': self.track_id,
            'width': round(self.width, 1),
            'height': round(self.height, 1),
            'area': round(self.area, 1),
            'center': [round(c, 1) for c in self.center],
        }


class DetectionResult:
    """Container for all detections from a single frame/image."""

    def __init__(self, detections: List[Detection], inference_time: float,
                 image_shape: tuple, frame_id: Optional[int] = None):
        self.detections = detections
        self.inference_time = inference_time
        self.image_shape = image_shape
        self.frame_id = frame_id

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    @property
    def fps(self) -> float:
        return 1.0 / self.inference_time if self.inference_time > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'detections': [d.to_dict() for d in self.detections],
            'count': self.count,
            'class_counts': self.class_counts,
            'inference_time_ms': round(self.inference_time * 1000, 2),
            'fps': round(self.fps, 1),
            'image_shape': list(self.image_shape),
            'frame_id': self.frame_id,
        }

    def filter_by_class(self, class_name: str) -> List[Detection]:
        return [d for d in self.detections if d.class_name == class_name]

    def filter_by_confidence(self, min_conf: float) -> List[Detection]:
        return [d for d in self.detections if d.confidence >= min_conf]


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Wraps the Ultralytics YOLO model with a clean API for production use.
    Handles model loading, device selection, and result parsing.

    Design decisions:
    - Lazy model loading: Model is loaded on first inference call
    - Thread-safe: Each detector instance owns its model
    - Configurable: All parameters can be tuned per-deployment
    """

    CLASS_NAMES = {0: 'car', 1: 'minivan'}

    def __init__(
        self,
        model_path: str = 'models/vehicle_detector.pt',
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: Optional[str] = None,
        half: bool = False,
    ):
        """
        Args:
            model_path: Path to YOLOv8 model (.pt or .onnx).
            confidence: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for NMS (Non-Maximum Suppression).
            img_size: Input image size for the model.
            device: Device to run on ('cuda', 'cpu', or None for auto).
            half: Use FP16 half-precision (faster on GPU, slight accuracy loss).
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.half = half
        self._model = None

    @property
    def model(self):
        """Lazy-load the YOLO model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the YOLOv8 model."""
        from ultralytics import YOLO

        if not self.model_path.exists():
            # Try to fall back to a pretrained model
            logger.warning(f"Model not found at {self.model_path}, using pretrained yolov8n.pt")
            self._model = YOLO('yolov8n.pt')
        else:
            logger.info(f"Loading model from {self.model_path}")
            self._model = YOLO(str(self.model_path))

        # Warm up the model with a dummy inference
        import torch
        dummy = torch.zeros(1, 3, self.img_size, self.img_size)
        if self.device:
            dummy = dummy.to(self.device)
        try:
            self._model.predict(dummy, verbose=False)
            logger.info(f"Model loaded and warmed up on {self.device or 'auto'}")
        except Exception:
            logger.info("Model loaded (warmup skipped)")

    def detect(
        self,
        source: Union[str, np.ndarray],
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        Run detection on a single image.

        Args:
            source: Image path (str) or numpy array (BGR, HWC format).
            confidence: Override default confidence threshold.
            iou_threshold: Override default IoU threshold.

        Returns:
            DetectionResult with all detections.
        """
        conf = confidence or self.confidence
        iou = iou_threshold or self.iou_threshold

        start_time = time.time()

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=self.img_size,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        inference_time = time.time() - start_time

        # Parse results
        detections = []
        result = results[0]  # Single image

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf_score = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Map class name
                if hasattr(result, 'names') and result.names:
                    class_name = result.names.get(cls_id, self.CLASS_NAMES.get(cls_id, f'class_{cls_id}'))
                else:
                    class_name = self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')

                detections.append(Detection(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf_score,
                    class_id=cls_id,
                    class_name=class_name,
                ))

        # Get image shape
        img_shape = result.orig_shape if hasattr(result, 'orig_shape') else (0, 0)

        return DetectionResult(
            detections=detections,
            inference_time=inference_time,
            image_shape=img_shape,
        )

    def detect_batch(
        self,
        sources: List[Union[str, np.ndarray]],
    ) -> List[DetectionResult]:
        """Run detection on a batch of images."""
        results_list = []
        for source in sources:
            result = self.detect(source)
            results_list.append(result)
        return results_list

    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            'model_path': str(self.model_path),
            'confidence': self.confidence,
            'iou_threshold': self.iou_threshold,
            'img_size': self.img_size,
            'device': str(self.device),
            'class_names': self.CLASS_NAMES,
            'num_classes': len(self.CLASS_NAMES),
        }
