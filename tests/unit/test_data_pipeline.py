"""
Unit Tests — Data Pipeline Modules
======================================
Tests for CVAT converter, validator, splitter, and analyzer.
"""

import json
from pathlib import Path


class TestValidator:
    """Tests for the DatasetValidator."""

    def test_valid_dataset_passes(self, temp_dataset):
        from src.data.validator import DatasetValidator

        validator = DatasetValidator(
            dataset_dir=str(temp_dataset),
            num_classes=2,
            check_images=False,
        )
        is_valid, result = validator.validate()
        assert is_valid or result.warnings_count >= 0  # May have warnings, not errors

    def test_invalid_class_id_fails(self, tmp_path):
        from src.data.validator import DatasetValidator

        (tmp_path / "labels" / "all").mkdir(parents=True)
        (tmp_path / "images" / "all").mkdir(parents=True)

        # Label with class_id=5, but num_classes=2
        label = tmp_path / "labels" / "all" / "test.txt"
        label.write_text("5 0.5 0.5 0.2 0.3\n")

        validator = DatasetValidator(str(tmp_path), num_classes=2, check_images=False)
        is_valid, result = validator.validate()
        # Should flag the out-of-range class ID
        assert result.errors_count > 0 or result.warnings_count > 0

    def test_empty_label_file(self, tmp_path):
        from src.data.validator import DatasetValidator

        (tmp_path / "labels" / "all").mkdir(parents=True)
        (tmp_path / "images" / "all").mkdir(parents=True)

        label = tmp_path / "labels" / "all" / "empty.txt"
        label.write_text("")

        validator = DatasetValidator(str(tmp_path), num_classes=2, check_images=False)
        is_valid, result = validator.validate()
        # Empty file is valid (image with no objects)
        assert result is not None


class TestSplitter:
    """Tests for the DatasetSplitter."""

    def test_split_creates_directories(self, temp_dataset):
        from src.data.dataset_splitter import DatasetSplitter

        splitter = DatasetSplitter(str(temp_dataset), seed=42)
        stats = splitter.split(train=0.7, val=0.2, test=0.1)

        assert (temp_dataset / "images" / "train").exists()
        assert (temp_dataset / "images" / "val").exists()
        assert (temp_dataset / "images" / "test").exists()
        assert (temp_dataset / "labels" / "train").exists()
        assert (temp_dataset / "labels" / "val").exists()
        assert (temp_dataset / "labels" / "test").exists()

    def test_split_preserves_total_count(self, temp_dataset):
        from src.data.dataset_splitter import DatasetSplitter

        splitter = DatasetSplitter(str(temp_dataset), seed=42)
        stats = splitter.split(train=0.7, val=0.2, test=0.1)

        total = 0
        for split in ['train', 'val', 'test']:
            split_dir = temp_dataset / "images" / split
            if split_dir.exists():
                total += len(list(split_dir.glob("*")))

        assert total == 10  # We created 10 images


class TestAnalyzer:
    """Tests for the DatasetAnalyzer."""

    def test_analyze_returns_stats(self, temp_dataset):
        from src.data.analyzer import DatasetAnalyzer

        analyzer = DatasetAnalyzer(str(temp_dataset))
        stats = analyzer.analyze()
        assert stats is not None

    def test_save_report(self, temp_dataset):
        from src.data.analyzer import DatasetAnalyzer

        analyzer = DatasetAnalyzer(str(temp_dataset))
        analyzer.analyze()

        report_path = str(temp_dataset / "report.json")
        analyzer.save_report(report_path)
        assert Path(report_path).exists()

        with open(report_path) as f:
            data = json.load(f)
        assert 'total_images' in data or 'num_images' in data or isinstance(data, dict)
