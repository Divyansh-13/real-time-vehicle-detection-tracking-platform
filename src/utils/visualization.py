"""
Visualization Utilities
=========================
Drawing functions for detection results, tracks, and analytics overlays.
Used by both the inference pipeline and analytics modules.
"""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Color palettes (BGR format for OpenCV)
CLASS_COLORS = {
    'car': (255, 178, 0),
    'minivan': (147, 37, 183),
}

TRACK_PALETTE = [
    (255, 178, 0), (0, 255, 128), (147, 37, 183), (0, 165, 255),
    (255, 0, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0),
    (128, 0, 255), (0, 255, 255), (255, 0, 128), (64, 255, 64),
    (255, 64, 64), (64, 64, 255), (255, 255, 0), (0, 192, 192),
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


def draw_box(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    label: str = '',
    color: Tuple[int, int, int] = (0, 255, 128),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw a single bounding box with label on an image."""
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label background
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        label_y1 = max(0, y1 - th - 10)
        cv2.rectangle(image, (x1, label_y1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(image, label, (x1 + 4, y1 - 4),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def draw_counting_line(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    count_up: int = 0,
    count_down: int = 0,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draw a counting line with direction counters."""
    cv2.line(image, start, end, color, 3)

    mid_x = (start[0] + end[0]) // 2
    mid_y = (start[1] + end[1]) // 2

    # Counter labels
    cv2.putText(image, f"UP: {count_up}", (mid_x - 80, mid_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(image, f"DOWN: {count_down}", (mid_x - 80, mid_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2, cv2.LINE_AA)

    return image


def draw_track_trail(
    image: np.ndarray,
    positions: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 128),
    max_points: int = 30,
    fade: bool = True,
) -> np.ndarray:
    """Draw a fading trail showing a track's recent path."""
    points = positions[-max_points:]
    for i in range(1, len(points)):
        alpha = i / len(points) if fade else 1.0
        pt_color = tuple(int(c * alpha) for c in color)
        pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        thickness = max(1, int(3 * alpha))
        cv2.line(image, pt1, pt2, pt_color, thickness)
    return image


def draw_info_overlay(
    image: np.ndarray,
    info: Dict[str, str],
    position: str = 'top-left',
) -> np.ndarray:
    """Draw an information overlay panel on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    line_height = 25
    padding = 10

    lines = [f"{k}: {v}" for k, v in info.items()]
    max_width = max(cv2.getTextSize(line, font, font_scale, 1)[0][0] for line in lines)

    # Background
    h = len(lines) * line_height + 2 * padding
    w = max_width + 2 * padding

    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x = image.shape[1] - w - 10
        y = 10
    else:
        x, y = 10, 10

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    for i, line in enumerate(lines):
        text_y = y + padding + (i + 1) * line_height
        cv2.putText(image, line, (x + padding, text_y),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def create_mosaic(images: List[np.ndarray], cols: int = 2) -> np.ndarray:
    """Create a mosaic/grid from multiple images."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Resize all to the same size
    h = min(img.shape[0] for img in images)
    w = min(img.shape[1] for img in images)
    resized = [cv2.resize(img, (w, h)) for img in images]

    rows_needed = (len(resized) + cols - 1) // cols
    # Pad to fill grid
    while len(resized) < rows_needed * cols:
        resized.append(np.zeros((h, w, 3), dtype=np.uint8))

    rows = []
    for r in range(rows_needed):
        row_images = resized[r * cols:(r + 1) * cols]
        rows.append(np.hstack(row_images))

    return np.vstack(rows)
