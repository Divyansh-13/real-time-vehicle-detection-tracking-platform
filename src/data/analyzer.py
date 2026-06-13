"""
Dataset Analyzer
=================
Generates comprehensive statistics and visualizations for the vehicle
detection dataset. Used for EDA (Exploratory Data Analysis) before training.

Produces:
    - Class distribution (bar chart)
    - Bounding box size distribution (histogram)
    - Objects per frame distribution
    - Aspect ratio analysis
    - Spatial heatmap (where objects appear in frames)
    - Summary statistics JSON

Usage:
    analyzer = DatasetAnalyzer("datasets/processed")
    report = analyzer.analyze()
    analyzer.save_report("datasets/analysis_report.json")
    analyzer.plot_all("datasets/analysis_plots/")
"""

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BoxStats:
    """Statistics for a single bounding box."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    area: float
    aspect_ratio: float


@dataclass
class AnalysisReport:
    """Complete dataset analysis report."""
    total_images: int = 0
    total_labels: int = 0
    images_without_labels: int = 0
    labels_without_images: int = 0
    total_boxes: int = 0
    class_distribution: Dict[int, int] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=lambda: {0: 'car', 1: 'minivan'})
    boxes_per_image: Dict[str, int] = field(default_factory=dict)
    avg_boxes_per_image: float = 0.0
    min_boxes_per_image: int = 0
    max_boxes_per_image: int = 0
    median_boxes_per_image: float = 0.0
    avg_box_width: float = 0.0
    avg_box_height: float = 0.0
    avg_box_area: float = 0.0
    min_box_area: float = 0.0
    max_box_area: float = 0.0
    avg_aspect_ratio: float = 0.0
    tiny_boxes: int = 0        # area < 0.001 (very small objects)
    small_boxes: int = 0       # area < 0.01
    medium_boxes: int = 0      # area < 0.05
    large_boxes: int = 0       # area >= 0.05


class DatasetAnalyzer:
    """
    Analyzes a YOLO-format dataset and produces comprehensive statistics.

    Reads label files (.txt) in YOLO format and computes metrics about
    class distribution, bounding box sizes, spatial distribution, and
    potential data quality issues.

    Why this matters:
    - Class imbalance detection → informs class weighting during training
    - Box size distribution → helps choose appropriate anchor sizes and image sizes
    - Spatial distribution → reveals camera bias (e.g., objects always in center)
    - Quality checks → catches conversion errors before expensive training runs
    """

    def __init__(
        self,
        dataset_dir: str,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            dataset_dir: Path to YOLO dataset root (containing labels/ and images/).
            class_names: Mapping of class_id to class name.
        """
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names or {0: 'car', 1: 'minivan'}
        self.all_boxes: List[BoxStats] = []
        self.report: Optional[AnalysisReport] = None

    def _find_label_files(self) -> List[Path]:
        """Find all .txt label files in the dataset directory tree."""
        label_files = []
        for labels_dir in ['labels/all', 'labels/train', 'labels/val', 'labels/test', 'labels']:
            path = self.dataset_dir / labels_dir
            if path.exists():
                label_files.extend(path.glob('*.txt'))
        # Deduplicate by filename
        seen = set()
        unique = []
        for f in label_files:
            if f.name not in seen:
                seen.add(f.name)
                unique.append(f)
        return sorted(unique)

    def _find_image_files(self) -> List[Path]:
        """Find all image files in the dataset directory tree."""
        image_files = []
        extensions = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG']
        for images_dir in ['images/all', 'images/train', 'images/val', 'images/test', 'images']:
            path = self.dataset_dir / images_dir
            if path.exists():
                for ext in extensions:
                    image_files.extend(path.glob(ext))
        seen = set()
        unique = []
        for f in image_files:
            if f.stem not in seen:
                seen.add(f.stem)
                unique.append(f)
        return sorted(unique)

    def _parse_label_file(self, label_path: Path) -> List[BoxStats]:
        """
        Parse a YOLO format label file.

        Each line: class_id x_center y_center width height
        All values are normalized [0, 1].
        """
        boxes = []
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    logger.warning(f"Invalid line {line_num} in {label_path}: {line}")
                    continue
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0.0

                    boxes.append(BoxStats(
                        class_id=class_id,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        area=area,
                        aspect_ratio=aspect_ratio,
                    ))
                except (ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Error parsing line {line_num} in {label_path}: {e}")
        return boxes

    def analyze(self) -> AnalysisReport:
        """
        Run full dataset analysis.

        Returns:
            AnalysisReport with comprehensive statistics.
        """
        logger.info(f"Analyzing dataset at: {self.dataset_dir}")
        report = AnalysisReport(class_names=self.class_names)

        label_files = self._find_label_files()
        image_files = self._find_image_files()

        label_stems = {f.stem for f in label_files}
        image_stems = {f.stem for f in image_files}

        report.total_labels = len(label_files)
        report.total_images = len(image_files)
        report.images_without_labels = len(image_stems - label_stems)
        report.labels_without_images = len(label_stems - image_stems)

        # Parse all label files
        self.all_boxes = []
        box_counts = []
        class_counter = Counter()

        for label_path in label_files:
            boxes = self._parse_label_file(label_path)
            self.all_boxes.extend(boxes)
            box_count = len(boxes)
            box_counts.append(box_count)
            report.boxes_per_image[label_path.stem] = box_count

            for box in boxes:
                class_counter[box.class_id] += 1

        report.total_boxes = len(self.all_boxes)
        report.class_distribution = dict(class_counter)

        # Box count statistics
        if box_counts:
            box_counts_sorted = sorted(box_counts)
            report.avg_boxes_per_image = sum(box_counts) / len(box_counts)
            report.min_boxes_per_image = min(box_counts)
            report.max_boxes_per_image = max(box_counts)
            mid = len(box_counts_sorted) // 2
            report.median_boxes_per_image = (
                box_counts_sorted[mid] if len(box_counts_sorted) % 2 == 1
                else (box_counts_sorted[mid - 1] + box_counts_sorted[mid]) / 2
            )

        # Box size statistics
        if self.all_boxes:
            widths = [b.width for b in self.all_boxes]
            heights = [b.height for b in self.all_boxes]
            areas = [b.area for b in self.all_boxes]
            ratios = [b.aspect_ratio for b in self.all_boxes if b.aspect_ratio > 0]

            report.avg_box_width = sum(widths) / len(widths)
            report.avg_box_height = sum(heights) / len(heights)
            report.avg_box_area = sum(areas) / len(areas)
            report.min_box_area = min(areas)
            report.max_box_area = max(areas)
            report.avg_aspect_ratio = sum(ratios) / len(ratios) if ratios else 0.0

            # Size categorization
            for area in areas:
                if area < 0.001:
                    report.tiny_boxes += 1
                elif area < 0.01:
                    report.small_boxes += 1
                elif area < 0.05:
                    report.medium_boxes += 1
                else:
                    report.large_boxes += 1

        self.report = report
        self._print_report()
        return report

    def _print_report(self):
        """Print a formatted analysis report to the console."""
        r = self.report
        if not r:
            return

        print("\n" + "=" * 70)
        print("  DATASET ANALYSIS REPORT")
        print("=" * 70)
        print(f"\n📁 Dataset: {self.dataset_dir}")
        print("\n📊 Overview:")
        print(f"   Total images:              {r.total_images}")
        print(f"   Total label files:          {r.total_labels}")
        print(f"   Images without labels:      {r.images_without_labels}")
        print(f"   Labels without images:      {r.labels_without_images}")
        print(f"   Total bounding boxes:       {r.total_boxes}")

        print("\n🏷️  Class Distribution:")
        for class_id, count in sorted(r.class_distribution.items()):
            name = r.class_names.get(class_id, f"class_{class_id}")
            pct = count / r.total_boxes * 100 if r.total_boxes > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"   {name:>10s} (id={class_id}): {count:>6d} ({pct:5.1f}%) {bar}")

        print("\n📦 Boxes per Image:")
        print(f"   Min:    {r.min_boxes_per_image}")
        print(f"   Max:    {r.max_boxes_per_image}")
        print(f"   Avg:    {r.avg_boxes_per_image:.1f}")
        print(f"   Median: {r.median_boxes_per_image:.1f}")

        print("\n📐 Box Dimensions (normalized):")
        print(f"   Avg width:        {r.avg_box_width:.4f}")
        print(f"   Avg height:       {r.avg_box_height:.4f}")
        print(f"   Avg area:         {r.avg_box_area:.4f}")
        print(f"   Min area:         {r.min_box_area:.6f}")
        print(f"   Max area:         {r.max_box_area:.4f}")
        print(f"   Avg aspect ratio: {r.avg_aspect_ratio:.2f}")

        print("\n📏 Box Size Distribution:")
        print(f"   Tiny  (<0.1%):    {r.tiny_boxes:>6d}")
        print(f"   Small (<1%):      {r.small_boxes:>6d}")
        print(f"   Medium (<5%):     {r.medium_boxes:>6d}")
        print(f"   Large (≥5%):      {r.large_boxes:>6d}")

        # Warnings
        print("\n⚠️  Potential Issues:")
        if r.images_without_labels > 0:
            print(f"   ⚠ {r.images_without_labels} images have no labels")
        if r.labels_without_images > 0:
            print(f"   ⚠ {r.labels_without_images} labels have no matching image")
        if r.tiny_boxes > 0:
            print(f"   ⚠ {r.tiny_boxes} tiny boxes (area < 0.1%) may be too small to detect")

        # Check class imbalance
        if r.class_distribution:
            counts = list(r.class_distribution.values())
            if max(counts) > 10 * min(counts):
                print(f"   ⚠ Severe class imbalance: ratio = {max(counts)/min(counts):.1f}:1")
                print("     → Consider class weighting or oversampling minority class")

        print("\n" + "=" * 70)

    def save_report(self, output_path: str):
        """Save the analysis report as JSON."""
        if not self.report:
            raise RuntimeError("Run analyze() before saving report")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = asdict(self.report)
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"Report saved to {output_path}")

    def plot_all(self, output_dir: str):
        """
        Generate all analysis plots and save to directory.

        Plots generated:
        1. class_distribution.png - Bar chart of class counts
        2. boxes_per_image.png - Histogram of objects per frame
        3. box_sizes.png - Scatter plot of box width vs height
        4. spatial_heatmap.png - Where objects appear in frames

        Requires matplotlib (optional dependency).
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot generation. "
                          "Install with: pip install matplotlib")
            return

        if not self.report or not self.all_boxes:
            raise RuntimeError("Run analyze() before plotting")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Style
        plt.rcParams.update({
            'figure.facecolor': '#1a1a2e',
            'axes.facecolor': '#16213e',
            'text.color': '#e0e0e0',
            'axes.labelcolor': '#e0e0e0',
            'xtick.color': '#a0a0a0',
            'ytick.color': '#a0a0a0',
            'axes.edgecolor': '#333355',
            'grid.color': '#333355',
            'font.size': 11,
        })

        # 1. Class distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = sorted(self.report.class_distribution.keys())
        names = [self.report.class_names.get(c, f"class_{c}") for c in classes]
        counts = [self.report.class_distribution[c] for c in classes]
        colors = ['#00d2ff', '#7928ca']
        bars = ax.bar(names, counts, color=colors[:len(names)], edgecolor='white', linewidth=0.5)
        ax.set_title('Class Distribution', fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('Number of Instances')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', color='white')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Boxes per image histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        box_counts = list(self.report.boxes_per_image.values())
        ax.hist(box_counts, bins=range(0, max(box_counts) + 2), color='#00d2ff',
                edgecolor='white', linewidth=0.5, alpha=0.8)
        ax.axvline(x=self.report.avg_boxes_per_image, color='#ff6b6b',
                   linestyle='--', linewidth=2, label=f'Mean: {self.report.avg_boxes_per_image:.1f}')
        ax.set_title('Objects per Frame', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Number of Objects')
        ax.set_ylabel('Number of Frames')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'boxes_per_image.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Box width vs height scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        for class_id in sorted(set(b.class_id for b in self.all_boxes)):
            class_boxes = [b for b in self.all_boxes if b.class_id == class_id]
            ws = [b.width for b in class_boxes]
            hs = [b.height for b in class_boxes]
            name = self.report.class_names.get(class_id, f"class_{class_id}")
            color = colors[class_id % len(colors)]
            ax.scatter(ws, hs, alpha=0.4, s=15, label=name, color=color)
        ax.set_title('Bounding Box Dimensions', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Width (normalized)')
        ax.set_ylabel('Height (normalized)')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max(b.width for b in self.all_boxes) * 1.1)
        ax.set_ylim(0, max(b.height for b in self.all_boxes) * 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / 'box_dimensions.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Spatial heatmap
        fig, ax = plt.subplots(figsize=(12, 7))
        import numpy as np
        heatmap = np.zeros((20, 32))  # Grid resolution
        for box in self.all_boxes:
            gx = min(int(box.x_center * 32), 31)
            gy = min(int(box.y_center * 20), 19)
            heatmap[gy, gx] += 1
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear', aspect='auto')
        ax.set_title('Object Spatial Distribution (Heatmap)', fontsize=16,
                     fontweight='bold', pad=15)
        ax.set_xlabel('Horizontal Position')
        ax.set_ylabel('Vertical Position')
        plt.colorbar(im, ax=ax, label='Object Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'spatial_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved 4 analysis plots to {output_dir}")


def main():
    """CLI entry point for dataset analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze YOLO format dataset')
    parser.add_argument('--dataset', type=str, default='datasets/processed',
                        help='Path to YOLO dataset directory')
    parser.add_argument('--report', type=str, default='datasets/analysis_report.json',
                        help='Output path for JSON report')
    parser.add_argument('--plots', type=str, default='datasets/analysis_plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(args.dataset)
    analyzer.analyze()
    analyzer.save_report(args.report)
    analyzer.plot_all(args.plots)


if __name__ == '__main__':
    main()
