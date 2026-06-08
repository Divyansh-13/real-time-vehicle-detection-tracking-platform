"""
Vehicle Counter — Line-Crossing Counter
==========================================
Counts vehicles crossing a virtual line or entering/exiting a region.

Used in traffic monitoring to count:
    - Vehicles per direction (northbound/southbound)
    - Total traffic volume over time
    - Peak hour detection

How it works:
    1. Define a counting line (two points on the frame)
    2. For each tracked vehicle, monitor its center position
    3. When the center crosses the line, increment the counter
    4. Track crossing direction using the sign of the cross product

Usage:
    counter = VehicleCounter(line_start=(0, 540), line_end=(1920, 540))
    for frame_result in tracking_results:
        counts = counter.update(frame_result)
    print(counter.get_counts())
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CountingEvent:
    """A single line-crossing event."""
    track_id: int
    class_name: str
    direction: str          # 'up' or 'down' (relative to line)
    frame_number: int
    position: Tuple[float, float]
    timestamp: Optional[float] = None


class VehicleCounter:
    """
    Counts vehicles crossing a virtual line using track positions.

    The counting line divides the frame into two halves. When a tracked
    vehicle's center point crosses from one side to the other, a count
    event is generated.

    Direction detection:
    - Uses the cross product of the line vector and the movement vector
    - Positive cross product → crossing in one direction
    - Negative cross product → crossing in the other direction

    Why line-crossing over region-based:
    - More robust to partial tracks (vehicles don't need to fully enter/exit)
    - Direction-aware (can distinguish northbound vs southbound)
    - Works with any line orientation (horizontal, vertical, diagonal)
    """

    def __init__(
        self,
        line_start: Tuple[int, int] = (0, 540),
        line_end: Tuple[int, int] = (1920, 540),
        min_track_length: int = 3,
    ):
        """
        Args:
            line_start: Starting point of the counting line (x, y).
            line_end: Ending point of the counting line (x, y).
            min_track_length: Minimum track history before counting (prevents false counts).
        """
        self.line_start = line_start
        self.line_end = line_end
        self.min_track_length = min_track_length

        # Track state
        self._track_positions: Dict[int, List[Tuple[float, float]]] = {}
        self._track_classes: Dict[int, str] = {}
        self._counted_ids: set = set()  # Tracks that already crossed

        # Counters
        self.count_up = 0
        self.count_down = 0
        self.events: List[CountingEvent] = []
        self.class_counts: Dict[str, Dict[str, int]] = {}  # {class: {up: N, down: M}}

    def _cross_product_sign(
        self, p: Tuple[float, float]
    ) -> float:
        """
        Compute which side of the counting line a point is on.

        Uses the cross product: (line_end - line_start) × (p - line_start)
        Positive = left side, Negative = right side, Zero = on the line.
        """
        dx = self.line_end[0] - self.line_start[0]
        dy = self.line_end[1] - self.line_start[1]
        px = p[0] - self.line_start[0]
        py = p[1] - self.line_start[1]
        return dx * py - dy * px

    def update(self, detection_result, frame_number: int = 0) -> List[CountingEvent]:
        """
        Update counter with new frame detections.

        Args:
            detection_result: DetectionResult from the tracker.
            frame_number: Current frame number.

        Returns:
            List of new counting events (empty if no crossings).
        """
        new_events = []

        for det in detection_result.detections:
            if det.track_id is None:
                continue

            tid = det.track_id
            center = det.center

            # Update track history
            if tid not in self._track_positions:
                self._track_positions[tid] = []
                self._track_classes[tid] = det.class_name

            self._track_positions[tid].append(center)

            # Skip if already counted or too short
            if tid in self._counted_ids:
                continue
            if len(self._track_positions[tid]) < self.min_track_length:
                continue

            # Check for line crossing
            positions = self._track_positions[tid]
            if len(positions) < 2:
                continue

            prev_side = self._cross_product_sign(positions[-2])
            curr_side = self._cross_product_sign(positions[-1])

            # Crossing detected when sign changes
            if prev_side * curr_side < 0:
                direction = 'down' if curr_side > 0 else 'up'

                event = CountingEvent(
                    track_id=tid,
                    class_name=det.class_name,
                    direction=direction,
                    frame_number=frame_number,
                    position=center,
                )
                new_events.append(event)
                self.events.append(event)
                self._counted_ids.add(tid)

                # Update counters
                if direction == 'up':
                    self.count_up += 1
                else:
                    self.count_down += 1

                # Update class-specific counters
                cls = det.class_name
                if cls not in self.class_counts:
                    self.class_counts[cls] = {'up': 0, 'down': 0}
                self.class_counts[cls][direction] += 1

                logger.debug(f"Vehicle #{tid} ({cls}) crossed {direction} at frame {frame_number}")

        return new_events

    def get_counts(self) -> Dict:
        """Get current counting summary."""
        return {
            'total': self.count_up + self.count_down,
            'up': self.count_up,
            'down': self.count_down,
            'by_class': self.class_counts,
            'events_count': len(self.events),
        }

    def reset(self):
        """Reset all counters and track state."""
        self._track_positions.clear()
        self._track_classes.clear()
        self._counted_ids.clear()
        self.count_up = 0
        self.count_down = 0
        self.events.clear()
        self.class_counts.clear()
