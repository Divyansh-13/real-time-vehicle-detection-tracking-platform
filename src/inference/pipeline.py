"""
Detection Pipeline — Unified Inference Interface
===================================================
High-level pipeline that handles all input types (image, video, webcam, RTSP)
and returns annotated results with optional visualization.

This is the main interface used by the FastAPI backend to process
user-uploaded content.

Usage:
    pipeline = DetectionPipeline("models/vehicle_detector.pt")

    # Image
    result = pipeline.process_image("input.jpg", "output.jpg")

    # Video
    results, output_path = pipeline.process_video("input.mp4", "output.mp4")

    # Stream (generator)
    for frame_result in pipeline.process_stream(0):  # webcam
        ...
"""

import logging
import os
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from .detector import DetectionResult, VehicleDetector
from .tracker import TrackingSummary, VehicleTracker

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Unified pipeline for detection and tracking across all input types.

    Architecture:
        Input → Decode → Detect/Track → Annotate → Encode → Output

    For images: Single-frame detection (no tracking)
    For videos/streams: Multi-frame detection + ByteTrack tracking
    """

    # Visualization colors (BGR) — distinct per class
    COLORS = {
        'car': (255, 178, 0),       # Cyan-blue
        'minivan': (147, 37, 183),   # Purple
        'default': (0, 255, 128),    # Green
    }

    # Track ID color palette (for unique track coloring)
    TRACK_COLORS = [
        (255, 178, 0), (0, 255, 128), (147, 37, 183), (0, 165, 255),
        (255, 0, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (0, 255, 255), (255, 0, 128), (64, 255, 64),
    ]

    def __init__(
        self,
        model_path: str = 'models/vehicle_detector.pt',
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device

        self._detector = VehicleDetector(
            model_path=model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            img_size=img_size,
            device=device,
        )
        self._tracker = VehicleTracker(
            model_path=model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            img_size=img_size,
            device=device,
        )

    # ─── Image Processing ────────────────────────────────────────────

    def process_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        draw: bool = True,
    ) -> Tuple[DetectionResult, Optional[np.ndarray]]:
        """
        Run detection on a single image.

        Args:
            input_path: Path to input image.
            output_path: Path to save annotated image (None to skip).
            draw: Whether to draw annotations on the image.

        Returns:
            Tuple of (DetectionResult, annotated_image_or_None).
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")

        result = self._detector.detect(image)

        annotated = None
        if draw:
            annotated = self._draw_detections(image, result)
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                cv2.imwrite(output_path, annotated)
                logger.info(f"Saved annotated image: {output_path}")

        return result, annotated

    def process_image_bytes(
        self,
        image_bytes: bytes,
        draw: bool = True,
    ) -> Tuple[DetectionResult, Optional[bytes]]:
        """
        Run detection on image bytes (from file upload).

        Args:
            image_bytes: Raw image bytes.
            draw: Whether to produce annotated image bytes.

        Returns:
            Tuple of (DetectionResult, annotated_image_bytes_or_None).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image from bytes")

        result = self._detector.detect(image)

        annotated_bytes = None
        if draw:
            annotated = self._draw_detections(image, result)
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            annotated_bytes = buffer.tobytes()

        return result, annotated_bytes

    # ─── Video Processing ────────────────────────────────────────────

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> Tuple[List[DetectionResult], TrackingSummary, Optional[str]]:
        """
        Run detection + tracking on a video file.

        Args:
            input_path: Path to input video.
            output_path: Path to save annotated video.
            max_frames: Maximum frames to process (None for all).
            progress_callback: Callable(frame_num, total_frames) for progress.

        Returns:
            Tuple of (list of per-frame results, tracking summary, output path).
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        logger.info(f"Processing video: {input_path} "
                     f"({width}x{height}, {fps}fps, {total_frames} frames)")

        # Setup video writer
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset tracker for new video
        self._tracker.reset()

        all_results = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_num >= max_frames):
                break

            # Run tracking
            result = self._tracker.track(frame)
            all_results.append(result)

            # Draw and write
            if writer:
                annotated = self._draw_detections(frame, result, show_tracks=True)
                writer.write(annotated)

            frame_num += 1

            if progress_callback and frame_num % 10 == 0:
                progress_callback(frame_num, total_frames)

        cap.release()
        if writer:
            writer.release()
            logger.info(f"Saved annotated video: {output_path}")

        summary = self._tracker.get_summary()
        logger.info(f"Video processing complete: {frame_num} frames, "
                     f"{summary.unique_tracks} unique vehicles, "
                     f"{summary.avg_fps:.1f} FPS")

        return all_results, summary, output_path

    # ─── Stream Processing ───────────────────────────────────────────

    def process_stream(
        self,
        source: Union[int, str] = 0,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[DetectionResult, np.ndarray], None, None]:
        """
        Process a live stream (webcam or RTSP) as a generator.

        Args:
            source: Webcam index (int) or RTSP URL (str).
            max_frames: Maximum frames to process.

        Yields:
            Tuple of (DetectionResult, annotated_frame).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open stream: {source}")

        self._tracker.reset()
        frame_num = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self._tracker.track(frame)
                annotated = self._draw_detections(frame, result, show_tracks=True)

                yield result, annotated

                frame_num += 1
                if max_frames and frame_num >= max_frames:
                    break
        finally:
            cap.release()

    # ─── Visualization ───────────────────────────────────────────────

    def _draw_detections(
        self,
        image: np.ndarray,
        result: DetectionResult,
        show_tracks: bool = False,
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on an image.

        Visual design:
        - Rounded rectangle backgrounds for labels
        - Color-coded by class (or by track ID when tracking)
        - Confidence percentage shown
        - Track ID shown when available
        """
        annotated = image.copy()

        for det in result.detections:
            # Choose color
            if show_tracks and det.track_id is not None:
                color = self.TRACK_COLORS[det.track_id % len(self.TRACK_COLORS)]
            else:
                color = self.COLORS.get(det.class_name, self.COLORS['default'])

            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Build label text
            label_parts = []
            if det.track_id is not None:
                label_parts.append(f"ID:{det.track_id}")
            label_parts.append(det.class_name)
            label_parts.append(f"{det.confidence:.0%}")
            label = ' '.join(label_parts)

            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_y1 = max(0, y1 - th - 10)
            label_y2 = y1
            cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 8, label_y2), color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x1 + 4, label_y2 - 4),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Draw frame info overlay
        info_text = f"Objects: {result.count} | {result.fps:.0f} FPS"
        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 128), 2, cv2.LINE_AA)

        return annotated

    def frame_to_jpeg_bytes(self, frame: np.ndarray, quality: int = 80) -> bytes:
        """Convert a frame to JPEG bytes for streaming."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()
