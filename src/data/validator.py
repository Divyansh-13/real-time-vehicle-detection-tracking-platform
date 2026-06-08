"""
Dataset Validator
==================
Validates the integrity of a YOLO-format dataset before training.

Checks:
    1. Every image has a corresponding label file
    2. Every label file has a corresponding image
    3. All label values are within valid ranges [0, 1]
    4. All class IDs are in the expected range
    5. No empty label files (unless intentional background images)
    6. No corrupt or unreadable images
    7. Bounding boxes don't have zero width/height
    8. No duplicate boxes in the same frame

Usage:
    validator = DatasetValidator("datasets/processed", num_classes=2)
    is_valid, issues = validator.validate()
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation issue found in the dataset."""
    severity: str           # 'error', 'warning', 'info'
    file: str               # File where issue was found
    line: int = 0           # Line number (for label files)
    message: str = ""       # Description of the issue


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool = True
    total_images: int = 0
    total_labels: int = 0
    total_boxes_checked: int = 0
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)

    @property
    def all_issues(self) -> List[ValidationIssue]:
        return self.errors + self.warnings + self.info


class DatasetValidator:
    """
    Validates YOLO-format dataset integrity.

    Why validate before training:
    - Corrupt labels cause training crashes with cryptic errors
    - Out-of-range coordinates silently degrade model performance
    - Missing images/labels cause index errors in data loaders
    - Duplicate boxes inflate loss and distort gradients
    - Zero-area boxes cause NaN in IoU calculations

    Industry practice: Always validate data pipeline output before
    committing to expensive GPU training hours.
    """

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG'}

    def __init__(
        self,
        dataset_dir: str,
        num_classes: int = 2,
        check_images: bool = True,
    ):
        """
        Args:
            dataset_dir: Path to YOLO dataset root.
            num_classes: Expected number of classes.
            check_images: Whether to verify image files can be opened.
        """
        self.dataset_dir = Path(dataset_dir)
        self.num_classes = num_classes
        self.check_images = check_images

    def _find_files(self, subdir: str, extensions: Set[str]) -> Dict[str, Path]:
        """Find files in directory tree, return dict of stem→path."""
        files = {}
        for dirpath in ['all', 'train', 'val', 'test', '.']:
            search_dir = self.dataset_dir / subdir / dirpath
            if search_dir.exists():
                for f in search_dir.iterdir():
                    if f.suffix in extensions and f.stem not in files:
                        files[f.stem] = f
        return files

    def _validate_label_file(self, label_path: Path) -> List[ValidationIssue]:
        """Validate a single label file."""
        issues = []
        seen_boxes = set()

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines or all(line.strip() == '' for line in lines):
            issues.append(ValidationIssue(
                severity='warning',
                file=str(label_path),
                message='Empty label file (background image?)'
            ))
            return issues

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Check format (5 values expected)
            if len(parts) != 5:
                issues.append(ValidationIssue(
                    severity='error',
                    file=str(label_path),
                    line=line_num,
                    message=f'Expected 5 values, got {len(parts)}: {line}'
                ))
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as e:
                issues.append(ValidationIssue(
                    severity='error',
                    file=str(label_path),
                    line=line_num,
                    message=f'Cannot parse values: {e}'
                ))
                continue

            # Check class ID
            if class_id < 0 or class_id >= self.num_classes:
                issues.append(ValidationIssue(
                    severity='error',
                    file=str(label_path),
                    line=line_num,
                    message=f'Class ID {class_id} out of range [0, {self.num_classes - 1}]'
                ))

            # Check coordinate ranges
            for name, val in [('x_center', x_center), ('y_center', y_center),
                              ('width', width), ('height', height)]:
                if val < 0 or val > 1:
                    issues.append(ValidationIssue(
                        severity='error',
                        file=str(label_path),
                        line=line_num,
                        message=f'{name}={val:.6f} out of range [0, 1]'
                    ))

            # Check for zero-area boxes
            if width <= 0 or height <= 0:
                issues.append(ValidationIssue(
                    severity='error',
                    file=str(label_path),
                    line=line_num,
                    message=f'Zero or negative dimensions: w={width:.6f}, h={height:.6f}'
                ))

            # Check for very tiny boxes
            if width * height < 0.0001:
                issues.append(ValidationIssue(
                    severity='warning',
                    file=str(label_path),
                    line=line_num,
                    message=f'Very tiny box (area={width*height:.6f}), may be noise'
                ))

            # Check for duplicate boxes
            box_key = f"{class_id}_{x_center:.4f}_{y_center:.4f}_{width:.4f}_{height:.4f}"
            if box_key in seen_boxes:
                issues.append(ValidationIssue(
                    severity='warning',
                    file=str(label_path),
                    line=line_num,
                    message=f'Duplicate box detected'
                ))
            seen_boxes.add(box_key)

        return issues

    def _validate_image(self, image_path: Path) -> List[ValidationIssue]:
        """Validate that an image can be opened and read."""
        issues = []
        try:
            # Quick check: file size > 0
            if image_path.stat().st_size == 0:
                issues.append(ValidationIssue(
                    severity='error',
                    file=str(image_path),
                    message='Image file is empty (0 bytes)'
                ))
                return issues

            # Attempt to open with PIL if available
            try:
                from PIL import Image
                img = Image.open(image_path)
                img.verify()  # Verify without fully decoding
            except ImportError:
                pass  # PIL not available, skip deep check
            except Exception as e:
                issues.append(ValidationIssue(
                    severity='error',
                    file=str(image_path),
                    message=f'Corrupt image: {e}'
                ))
        except OSError as e:
            issues.append(ValidationIssue(
                severity='error',
                file=str(image_path),
                message=f'Cannot access image: {e}'
            ))
        return issues

    def validate(self) -> Tuple[bool, ValidationResult]:
        """
        Run full dataset validation.

        Returns:
            Tuple of (is_valid, ValidationResult).
            is_valid is False if any errors were found.
        """
        logger.info(f"Validating dataset: {self.dataset_dir}")
        result = ValidationResult()

        # Find all files
        label_files = self._find_files('labels', {'.txt'})
        image_files = self._find_files('images', self.IMAGE_EXTENSIONS)

        result.total_labels = len(label_files)
        result.total_images = len(image_files)

        # Check for mismatches
        label_stems = set(label_files.keys())
        image_stems = set(image_files.keys())

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        for stem in missing_labels:
            result.warnings.append(ValidationIssue(
                severity='warning',
                file=str(image_files[stem]),
                message='Image has no corresponding label file'
            ))

        for stem in missing_images:
            result.errors.append(ValidationIssue(
                severity='error',
                file=str(label_files[stem]),
                message='Label has no corresponding image file'
            ))

        # Validate each label file
        for stem, label_path in sorted(label_files.items()):
            issues = self._validate_label_file(label_path)
            result.total_boxes_checked += 1
            for issue in issues:
                if issue.severity == 'error':
                    result.errors.append(issue)
                elif issue.severity == 'warning':
                    result.warnings.append(issue)
                else:
                    result.info.append(issue)

        # Validate images (optional, slow)
        if self.check_images:
            for stem, image_path in sorted(image_files.items()):
                issues = self._validate_image(image_path)
                for issue in issues:
                    if issue.severity == 'error':
                        result.errors.append(issue)
                    else:
                        result.warnings.append(issue)

        result.is_valid = len(result.errors) == 0

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  VALIDATION RESULT: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        print(f"{'=' * 60}")
        print(f"  Images:   {result.total_images}")
        print(f"  Labels:   {result.total_labels}")
        print(f"  Errors:   {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")

        if result.errors:
            print(f"\n  ❌ ERRORS (must fix):")
            for issue in result.errors[:20]:  # Show first 20
                print(f"     {issue.file}:{issue.line} - {issue.message}")
            if len(result.errors) > 20:
                print(f"     ... and {len(result.errors) - 20} more errors")

        if result.warnings:
            print(f"\n  ⚠️  WARNINGS (review):")
            for issue in result.warnings[:10]:
                print(f"     {issue.file}:{issue.line} - {issue.message}")
            if len(result.warnings) > 10:
                print(f"     ... and {len(result.warnings) - 10} more warnings")

        print(f"{'=' * 60}\n")

        return result.is_valid, result


def main():
    """CLI entry point for dataset validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate YOLO format dataset')
    parser.add_argument('--dataset', type=str, default='datasets/processed',
                        help='Path to YOLO dataset directory')
    parser.add_argument('--classes', type=int, default=2,
                        help='Expected number of classes')
    parser.add_argument('--no-image-check', action='store_true',
                        help='Skip image integrity checks')
    args = parser.parse_args()

    validator = DatasetValidator(
        dataset_dir=args.dataset,
        num_classes=args.classes,
        check_images=not args.no_image_check,
    )
    is_valid, result = validator.validate()

    if not is_valid:
        exit(1)


if __name__ == '__main__':
    main()
