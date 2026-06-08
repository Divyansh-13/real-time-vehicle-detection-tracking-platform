"""
Speed Estimator — Pixel-to-Real-World Speed Estimation
=========================================================
Estimates vehicle speed from tracked positions using pixel displacement
and a calibration factor.

How it works:
    1. Track a vehicle's center position over consecutive frames
    2. Calculate pixel displacement per frame
    3. Convert pixels to meters using a calibration factor (pixels_per_meter)
    4. Multiply by frame rate to get meters/second
    5. Convert to km/h

Calibration:
    The pixels_per_meter value depends on camera setup (height, angle, lens).
    For this aerial dataset, we estimate based on typical car length (~4.5m)
    and observed pixel width of cars in the frame (~80-120 pixels).

Usage:
    estimator = SpeedEstimator(fps=30, pixels_per_meter=20.0)
    for frame_result in tracking_results:
        speeds = estimator.update(frame_result)
    avg_speed = estimator.get_average_speed()
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SpeedEstimator:
    """
    Estimates vehicle speed from tracked positions.

    Limitations (documented for transparency):
    - Accuracy depends on calibration quality
    - Perspective distortion affects measurements (objects farther from camera appear smaller)
    - Fast-moving objects may have fewer tracking points
    - Best used for relative comparison, not absolute measurement

    For production accuracy, use:
    - Camera calibration matrix (intrinsic/extrinsic parameters)
    - Homography transformation to bird's-eye view
    - Ground truth speed measurements for validation
    """

    def __init__(
        self,
        fps: float = 30.0,
        pixels_per_meter: float = 20.0,
        smoothing_window: int = 5,
    ):
        """
        Args:
            fps: Video frame rate (frames per second).
            pixels_per_meter: Calibration factor (how many pixels = 1 meter).
            smoothing_window: Number of frames to average speed over (reduces noise).
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_window = smoothing_window

        # Track state: {track_id: [positions]}
        self._positions: Dict[int, List[Tuple[float, float]]] = {}
        self._speeds: Dict[int, List[float]] = {}  # km/h per frame

    def update(self, detection_result) -> Dict[int, float]:
        """
        Update speed estimates with new frame detections.

        Args:
            detection_result: DetectionResult from tracker.

        Returns:
            Dict mapping track_id → estimated speed (km/h).
        """
        current_speeds = {}

        for det in detection_result.detections:
            if det.track_id is None:
                continue

            tid = det.track_id
            center = det.center

            if tid not in self._positions:
                self._positions[tid] = []
                self._speeds[tid] = []

            self._positions[tid].append(center)

            # Need at least 2 positions to calculate speed
            if len(self._positions[tid]) < 2:
                continue

            # Calculate pixel displacement
            prev = self._positions[tid][-2]
            curr = self._positions[tid][-1]
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

            # Convert to real-world speed
            meters_per_frame = pixel_distance / self.pixels_per_meter
            meters_per_second = meters_per_frame * self.fps
            kmh = meters_per_second * 3.6  # m/s → km/h

            self._speeds[tid].append(kmh)

            # Smoothed speed (average over window)
            window = self._speeds[tid][-self.smoothing_window:]
            smoothed_kmh = sum(window) / len(window)
            current_speeds[tid] = round(smoothed_kmh, 1)

        return current_speeds

    def get_speed(self, track_id: int) -> Optional[float]:
        """Get the latest smoothed speed for a specific track."""
        if track_id not in self._speeds or not self._speeds[track_id]:
            return None
        window = self._speeds[track_id][-self.smoothing_window:]
        return round(sum(window) / len(window), 1)

    def get_all_speeds(self) -> Dict[int, float]:
        """Get smoothed speeds for all active tracks."""
        return {
            tid: self.get_speed(tid)
            for tid in self._speeds
            if self.get_speed(tid) is not None
        }

    def get_average_speed(self) -> float:
        """Get average speed across all tracks."""
        all_speeds = [s for s in self.get_all_speeds().values() if s and s > 0]
        return round(sum(all_speeds) / len(all_speeds), 1) if all_speeds else 0.0

    def get_max_speed(self) -> Tuple[Optional[int], float]:
        """Get the track with the highest speed."""
        speeds = self.get_all_speeds()
        if not speeds:
            return None, 0.0
        max_tid = max(speeds, key=speeds.get)
        return max_tid, speeds[max_tid]

    def get_statistics(self) -> Dict:
        """Get comprehensive speed statistics."""
        all_speeds = list(self.get_all_speeds().values())
        valid = [s for s in all_speeds if s and s > 0]

        if not valid:
            return {
                'avg_speed_kmh': 0, 'max_speed_kmh': 0,
                'min_speed_kmh': 0, 'num_tracked': 0,
            }

        return {
            'avg_speed_kmh': round(sum(valid) / len(valid), 1),
            'max_speed_kmh': round(max(valid), 1),
            'min_speed_kmh': round(min(valid), 1),
            'num_tracked': len(valid),
        }

    def reset(self):
        """Reset all speed tracking state."""
        self._positions.clear()
        self._speeds.clear()
