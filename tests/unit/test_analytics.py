"""
Unit Tests — Analytics Modules
=================================
Tests for VehicleCounter, SpeedEstimator, and TrafficHeatmap.
"""

from unittest.mock import MagicMock


def _make_detection(x1, y1, x2, y2, conf, cls_id, cls_name, track_id=None):
    """Create a mock detection object."""
    det = MagicMock()
    det.x1 = x1
    det.y1 = y1
    det.x2 = x2
    det.y2 = y2
    det.confidence = conf
    det.class_id = cls_id
    det.class_name = cls_name
    det.track_id = track_id
    det.center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return det


def _make_result(detections):
    """Create a mock DetectionResult."""
    result = MagicMock()
    result.detections = detections
    return result


class TestVehicleCounter:
    """Tests for the line-crossing vehicle counter."""

    def test_no_crossing(self):
        from src.analytics.counter import VehicleCounter

        counter = VehicleCounter(
            line_start=(0, 540),
            line_end=(1920, 540),
            min_track_length=2,
        )

        # Vehicle above the line, stays above
        det = _make_detection(100, 100, 200, 200, 0.9, 0, 'car', track_id=1)
        result = _make_result([det])

        events = counter.update(result, frame_number=0)
        events2 = counter.update(result, frame_number=1)

        assert counter.count_up + counter.count_down == 0

    def test_crossing_detected(self):
        from src.analytics.counter import VehicleCounter

        counter = VehicleCounter(
            line_start=(0, 540),
            line_end=(1920, 540),
            min_track_length=2,
        )

        # Frame 1: Vehicle above line
        det1 = _make_detection(100, 400, 200, 500, 0.9, 0, 'car', track_id=1)
        det1.center = (150, 450)
        result1 = _make_result([det1])
        counter.update(result1, frame_number=0)

        # Frame 2: Still above line
        det2 = _make_detection(100, 480, 200, 530, 0.9, 0, 'car', track_id=1)
        det2.center = (150, 505)
        result2 = _make_result([det2])
        counter.update(result2, frame_number=1)

        # Frame 3: Vehicle crosses below line (y > 540)
        det3 = _make_detection(100, 560, 200, 660, 0.9, 0, 'car', track_id=1)
        det3.center = (150, 610)
        result3 = _make_result([det3])
        events = counter.update(result3, frame_number=2)

        counts = counter.get_counts()
        assert counts['total'] == 1

    def test_reset(self):
        from src.analytics.counter import VehicleCounter

        counter = VehicleCounter()
        counter.count_up = 5
        counter.count_down = 3
        counter.reset()

        assert counter.get_counts()['total'] == 0


class TestSpeedEstimator:
    """Tests for the speed estimator."""

    def test_stationary_vehicle(self):
        from src.analytics.speed_estimator import SpeedEstimator

        estimator = SpeedEstimator(fps=30, pixels_per_meter=20.0)

        # Same position across frames
        for frame in range(5):
            det = _make_detection(100, 200, 200, 300, 0.9, 0, 'car', track_id=1)
            det.center = (150, 250)
            result = _make_result([det])
            estimator.update(result)

        speed = estimator.get_speed(1)
        assert speed is not None
        assert speed < 1.0  # Should be near zero

    def test_moving_vehicle(self):
        from src.analytics.speed_estimator import SpeedEstimator

        estimator = SpeedEstimator(fps=30, pixels_per_meter=20.0)

        # Vehicle moving 10 pixels per frame
        for frame in range(10):
            x = 100 + frame * 10
            det = _make_detection(x, 200, x + 100, 300, 0.9, 0, 'car', track_id=1)
            det.center = (x + 50, 250)
            result = _make_result([det])
            estimator.update(result)

        speed = estimator.get_speed(1)
        assert speed is not None
        assert speed > 0

    def test_statistics(self):
        from src.analytics.speed_estimator import SpeedEstimator

        estimator = SpeedEstimator(fps=30)
        stats = estimator.get_statistics()
        assert stats['num_tracked'] == 0


class TestTrafficHeatmap:
    """Tests for the traffic heatmap generator."""

    def test_heatmap_update(self):
        from src.analytics.heatmap import TrafficHeatmap

        heatmap = TrafficHeatmap(
            frame_shape=(1080, 1920),
            grid_resolution=(10, 10),
        )

        det = _make_detection(100, 200, 200, 300, 0.9, 0, 'car')
        det.center = (150, 250)
        result = _make_result([det])

        heatmap.update(result)

        assert heatmap._total_frames == 1
        assert heatmap._grid.sum() > 0

    def test_heatmap_statistics(self):
        from src.analytics.heatmap import TrafficHeatmap

        heatmap = TrafficHeatmap(grid_resolution=(10, 10))

        det = _make_detection(960, 540, 1060, 640, 0.9, 0, 'car')
        det.center = (1010, 590)
        result = _make_result([det])

        for _ in range(10):
            heatmap.update(result)

        stats = heatmap.get_statistics()
        assert stats['total_frames'] == 10
        assert stats['max_density'] > 0
        assert len(stats['hotspots']) > 0

    def test_heatmap_reset(self):
        from src.analytics.heatmap import TrafficHeatmap

        heatmap = TrafficHeatmap(grid_resolution=(5, 5))

        det = _make_detection(100, 100, 200, 200, 0.9, 0, 'car')
        det.center = (150, 150)
        heatmap.update(_make_result([det]))

        heatmap.reset()
        assert heatmap._total_frames == 0
        assert heatmap._grid.sum() == 0
