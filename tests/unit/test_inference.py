"""
Unit Tests — Inference Engine
================================
Tests for the Detector, Tracker, and Pipeline.
"""

import pytest


class TestDetection:
    """Tests for the Detection and DetectionResult data classes."""

    def test_detection_properties(self):
        from src.inference.detector import Detection

        det = Detection(
            x1=100, y1=200, x2=300, y2=400,
            confidence=0.95, class_id=0, class_name='car',
        )

        assert det.width == 200
        assert det.height == 200
        assert det.area == 40000
        assert det.center == (200.0, 300.0)
        assert det.bbox == [100, 200, 300, 400]

    def test_detection_to_dict(self):
        from src.inference.detector import Detection

        det = Detection(
            x1=10, y1=20, x2=110, y2=120,
            confidence=0.8, class_id=1, class_name='minivan',
            track_id=5,
        )
        d = det.to_dict()
        assert d['class_name'] == 'minivan'
        assert d['confidence'] == 0.8
        assert d['track_id'] == 5

    def test_detection_result(self):
        from src.inference.detector import Detection, DetectionResult

        dets = [
            Detection(0, 0, 100, 100, 0.9, 0, 'car'),
            Detection(200, 200, 300, 300, 0.8, 0, 'car'),
            Detection(400, 400, 500, 500, 0.7, 1, 'minivan'),
        ]
        result = DetectionResult(dets, inference_time=0.05, image_shape=(1080, 1920))

        assert result.count == 3
        assert result.class_counts == {'car': 2, 'minivan': 1}
        assert result.fps == pytest.approx(20.0)
        assert len(result.filter_by_class('car')) == 2
        assert len(result.filter_by_confidence(0.85)) == 1


class TestTracker:
    """Tests for the TrackInfo and TrackingSummary."""

    def test_track_info(self):
        from src.inference.tracker import TrackInfo

        track = TrackInfo(track_id=1, class_name='car')
        track.positions = [(100, 100), (110, 110), (120, 120)]
        track.confidences = [0.9, 0.85, 0.88]

        assert track.avg_confidence == pytest.approx(0.8767, abs=0.01)
        assert track.travel_distance > 0

        d = track.to_dict()
        assert d['track_id'] == 1
        assert d['class_name'] == 'car'

    def test_tracking_summary(self):
        from src.inference.tracker import TrackingSummary

        summary = TrackingSummary()
        summary.total_frames = 100
        summary.total_detections = 500
        summary.total_time = 5.0

        assert summary.avg_fps == pytest.approx(20.0)
        assert summary.avg_detections_per_frame == pytest.approx(5.0)
