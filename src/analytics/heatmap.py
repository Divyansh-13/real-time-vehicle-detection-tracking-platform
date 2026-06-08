"""
Traffic Heatmap Generator
============================
Generates spatial heatmaps showing where vehicles appear most frequently.

Applications:
    - Identify high-traffic zones in the frame
    - Detect common vehicle paths / lanes
    - Visualize parking patterns
    - Identify blind spots in camera coverage

Usage:
    heatmap = TrafficHeatmap(frame_shape=(1080, 1920))
    for frame_result in tracking_results:
        heatmap.update(frame_result)
    overlay = heatmap.generate_overlay(background_image)
    heatmap.save("traffic_heatmap.png")
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrafficHeatmap:
    """
    Accumulates vehicle positions to generate traffic density heatmaps.

    The heatmap is a 2D grid where each cell counts how many detection
    centers fell within it across all processed frames. Higher counts
    indicate more frequent vehicle presence.

    The heatmap can be:
    - Rendered as a standalone image (hot colormap)
    - Overlaid on a background frame (semi-transparent)
    - Exported as a numpy array for further analysis
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int] = (1080, 1920),
        grid_resolution: Tuple[int, int] = (108, 192),
        decay_factor: float = 0.0,
    ):
        """
        Args:
            frame_shape: (height, width) of the video frames.
            grid_resolution: (rows, cols) of the heatmap grid.
                             Higher = more detail, slower.
            decay_factor: Temporal decay per frame (0 = no decay, 0.01 = slow fade).
                         Use decay for live/streaming to show recent activity.
        """
        self.frame_shape = frame_shape
        self.grid_resolution = grid_resolution
        self.decay_factor = decay_factor

        self._grid = np.zeros(grid_resolution, dtype=np.float64)
        self._cell_height = frame_shape[0] / grid_resolution[0]
        self._cell_width = frame_shape[1] / grid_resolution[1]
        self._total_frames = 0

    def update(self, detection_result) -> None:
        """
        Add detections from a single frame to the heatmap.

        Args:
            detection_result: DetectionResult from detector/tracker.
        """
        # Apply temporal decay
        if self.decay_factor > 0:
            self._grid *= (1.0 - self.decay_factor)

        for det in detection_result.detections:
            cx, cy = det.center

            # Map pixel position to grid cell
            row = min(int(cy / self._cell_height), self.grid_resolution[0] - 1)
            col = min(int(cx / self._cell_width), self.grid_resolution[1] - 1)
            row = max(0, row)
            col = max(0, col)

            # Increment with gaussian-like spread for smoother heatmaps
            self._grid[row, col] += 1.0

            # Spread to adjacent cells (smoother appearance)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_resolution[0] and 0 <= nc < self.grid_resolution[1]:
                        if dr != 0 or dc != 0:
                            self._grid[nr, nc] += 0.3

        self._total_frames += 1

    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Get the raw heatmap array.

        Args:
            normalize: If True, normalize values to [0, 1].

        Returns:
            2D numpy array of shape grid_resolution.
        """
        if normalize and self._grid.max() > 0:
            return self._grid / self._grid.max()
        return self._grid.copy()

    def generate_image(self, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Generate a colorized heatmap image.

        Args:
            size: Output image size (width, height). Defaults to frame_shape.

        Returns:
            BGR numpy array (uint8) with the heatmap visualization.
        """
        import cv2

        heatmap = self.get_heatmap(normalize=True)

        # Convert to uint8 for colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        # Resize to target size
        target_size = size or (self.frame_shape[1], self.frame_shape[0])
        heatmap_resized = cv2.resize(
            heatmap_uint8, target_size, interpolation=cv2.INTER_CUBIC
        )

        # Apply colormap (JET gives blue→green→yellow→red)
        colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        return colored

    def generate_overlay(
        self,
        background: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay the heatmap on a background image.

        Args:
            background: Background image (BGR numpy array).
            alpha: Transparency of the heatmap overlay (0 = invisible, 1 = opaque).

        Returns:
            BGR numpy array with the heatmap overlaid.
        """
        import cv2

        h, w = background.shape[:2]
        heatmap_img = self.generate_image(size=(w, h))

        # Blend
        overlay = cv2.addWeighted(background, 1.0 - alpha, heatmap_img, alpha, 0)

        return overlay

    def save(self, output_path: str, background: Optional[np.ndarray] = None):
        """Save the heatmap as an image file."""
        import cv2

        if background is not None:
            image = self.generate_overlay(background)
        else:
            image = self.generate_image()

        cv2.imwrite(output_path, image)
        logger.info(f"Heatmap saved to {output_path}")

    def get_hotspots(self, top_n: int = 5) -> list:
        """
        Get the top-N hotspot positions (highest traffic areas).

        Returns:
            List of dicts with 'row', 'col', 'pixel_x', 'pixel_y', 'count'.
        """
        flat_indices = np.argsort(self._grid.ravel())[::-1][:top_n]
        hotspots = []

        for idx in flat_indices:
            row = idx // self.grid_resolution[1]
            col = idx % self.grid_resolution[1]
            count = self._grid[row, col]

            if count <= 0:
                break

            hotspots.append({
                'row': int(row),
                'col': int(col),
                'pixel_x': int(col * self._cell_width + self._cell_width / 2),
                'pixel_y': int(row * self._cell_height + self._cell_height / 2),
                'count': float(count),
            })

        return hotspots

    def get_statistics(self) -> dict:
        """Get heatmap statistics."""
        return {
            'total_frames': self._total_frames,
            'grid_resolution': list(self.grid_resolution),
            'max_density': float(self._grid.max()),
            'mean_density': float(self._grid.mean()),
            'coverage_pct': float(
                np.count_nonzero(self._grid) / self._grid.size * 100
            ),
            'hotspots': self.get_hotspots(5),
        }

    def reset(self):
        """Reset the heatmap."""
        self._grid = np.zeros(self.grid_resolution, dtype=np.float64)
        self._total_frames = 0
