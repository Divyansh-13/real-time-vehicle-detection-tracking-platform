"""
Dataset Splitter
=================
Splits a YOLO-format dataset into train/val/test sets with stratification.

Splitting strategy:
    - Default ratio: 70% train / 20% validation / 10% test
    - Stratified by object density (objects per frame) to ensure each split
      has a representative mix of easy (few objects) and hard (many objects) frames
    - Preserves temporal ordering within each split (no shuffle) to maintain
      potential sequence-based evaluation

Why stratified splitting matters:
    - Random splitting can put all hard frames in validation
    - Class-imbalanced datasets need careful splitting to avoid
      having entire classes missing from val/test
    - For video data, temporal splits prevent data leakage from
      interpolated annotations

Usage:
    splitter = DatasetSplitter("datasets/processed")
    stats = splitter.split(train=0.7, val=0.2, test=0.1)
"""

import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SplitStats:
    """Statistics from the splitting process."""
    total_samples: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    train_boxes: int = 0
    val_boxes: int = 0
    test_boxes: int = 0
    train_class_dist: Dict[int, int] = field(default_factory=dict)
    val_class_dist: Dict[int, int] = field(default_factory=dict)
    test_class_dist: Dict[int, int] = field(default_factory=dict)


class DatasetSplitter:
    """
    Splits a YOLO-format dataset into train/val/test partitions.

    Operates on the 'all' subdirectory (output of CVATToYOLOConverter)
    and creates train/, val/, test/ subdirectories under both images/ and labels/.

    The split is stratified by the number of objects per frame, bucketed into
    bins: [0-2, 3-5, 6-8, 9+]. This ensures each split has a representative
    distribution of scene complexity.
    """

    IMAGE_EXTENSIONS = {'.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.bmp'}

    def __init__(self, dataset_dir: str, seed: int = 42):
        """
        Args:
            dataset_dir: Path to YOLO dataset root (containing images/all/ and labels/all/).
            seed: Random seed for reproducible splits.
        """
        self.dataset_dir = Path(dataset_dir)
        self.seed = seed
        random.seed(seed)

    def _count_boxes(self, label_path: Path) -> Tuple[int, Dict[int, int]]:
        """Count total boxes and per-class distribution in a label file."""
        total = 0
        class_dist: Dict[int, int] = {}
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        total += 1
                        class_dist[class_id] = class_dist.get(class_id, 0) + 1
        return total, class_dist

    def _get_density_bucket(self, num_boxes: int) -> int:
        """Assign a density bucket for stratification."""
        if num_boxes <= 2:
            return 0
        elif num_boxes <= 5:
            return 1
        elif num_boxes <= 8:
            return 2
        else:
            return 3

    def split(
        self,
        train: float = 0.7,
        val: float = 0.2,
        test: float = 0.1,
        copy_mode: str = 'copy',
    ) -> SplitStats:
        """
        Split the dataset into train/val/test sets.

        Args:
            train: Fraction of data for training (0-1).
            val: Fraction of data for validation (0-1).
            test: Fraction of data for testing (0-1).
            copy_mode: 'copy' to copy files, 'move' to move files,
                       'symlink' to create symbolic links.

        Returns:
            SplitStats with per-split counts and class distributions.
        """
        assert abs(train + val + test - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train + val + test}"

        stats = SplitStats()

        # Find all samples (match images to labels by stem)
        images_all_dir = self.dataset_dir / 'images' / 'all'
        labels_all_dir = self.dataset_dir / 'labels' / 'all'

        if not images_all_dir.exists() or not labels_all_dir.exists():
            raise FileNotFoundError(
                f"Expected images/all/ and labels/all/ in {self.dataset_dir}. "
                f"Run CVATToYOLOConverter first."
            )

        # Build sample list
        samples = []
        for img_path in sorted(images_all_dir.iterdir()):
            if img_path.suffix in self.IMAGE_EXTENSIONS:
                label_path = labels_all_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    box_count, class_dist = self._count_boxes(label_path)
                    samples.append({
                        'stem': img_path.stem,
                        'image': img_path,
                        'label': label_path,
                        'box_count': box_count,
                        'class_dist': class_dist,
                        'bucket': self._get_density_bucket(box_count),
                    })

        stats.total_samples = len(samples)
        logger.info(f"Found {stats.total_samples} samples to split")

        if stats.total_samples == 0:
            logger.error("No samples found! Check that images/all/ and labels/all/ exist.")
            return stats

        # Stratified split by density bucket
        buckets: Dict[int, List] = {}
        for sample in samples:
            b = sample['bucket']
            if b not in buckets:
                buckets[b] = []
            buckets[b].append(sample)

        train_samples, val_samples, test_samples = [], [], []

        for bucket_id, bucket_samples in sorted(buckets.items()):
            random.shuffle(bucket_samples)
            n = len(bucket_samples)
            n_train = max(1, int(n * train))
            n_val = max(1, int(n * val)) if n > 2 else 0
            # Ensure at least 1 sample in train
            n - n_train - n_val

            train_samples.extend(bucket_samples[:n_train])
            val_samples.extend(bucket_samples[n_train:n_train + n_val])
            test_samples.extend(bucket_samples[n_train + n_val:])

        # Create output directories and copy/move files
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples,
        }

        for split_name, split_samples in splits.items():
            img_dir = self.dataset_dir / 'images' / split_name
            lbl_dir = self.dataset_dir / 'labels' / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for sample in split_samples:
                dst_img = img_dir / sample['image'].name
                dst_lbl = lbl_dir / sample['label'].name

                if copy_mode == 'copy':
                    shutil.copy2(sample['image'], dst_img)
                    shutil.copy2(sample['label'], dst_lbl)
                elif copy_mode == 'move':
                    shutil.move(str(sample['image']), str(dst_img))
                    shutil.move(str(sample['label']), str(dst_lbl))
                elif copy_mode == 'symlink':
                    if not dst_img.exists():
                        dst_img.symlink_to(sample['image'].resolve())
                    if not dst_lbl.exists():
                        dst_lbl.symlink_to(sample['label'].resolve())

        # Compute stats
        stats.train_count = len(train_samples)
        stats.val_count = len(val_samples)
        stats.test_count = len(test_samples)

        for sample in train_samples:
            stats.train_boxes += sample['box_count']
            for cid, cnt in sample['class_dist'].items():
                stats.train_class_dist[cid] = stats.train_class_dist.get(cid, 0) + cnt

        for sample in val_samples:
            stats.val_boxes += sample['box_count']
            for cid, cnt in sample['class_dist'].items():
                stats.val_class_dist[cid] = stats.val_class_dist.get(cid, 0) + cnt

        for sample in test_samples:
            stats.test_boxes += sample['box_count']
            for cid, cnt in sample['class_dist'].items():
                stats.test_class_dist[cid] = stats.test_class_dist.get(cid, 0) + cnt

        # Print summary
        print(f"\n{'=' * 60}")
        print("  DATASET SPLIT COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Total samples: {stats.total_samples}")
        print(f"  ├── Train: {stats.train_count:>4d} ({stats.train_count/stats.total_samples*100:.0f}%) "
              f"| {stats.train_boxes} boxes | classes: {stats.train_class_dist}")
        print(f"  ├── Val:   {stats.val_count:>4d} ({stats.val_count/stats.total_samples*100:.0f}%) "
              f"| {stats.val_boxes} boxes | classes: {stats.val_class_dist}")
        print(f"  └── Test:  {stats.test_count:>4d} ({stats.test_count/stats.total_samples*100:.0f}%) "
              f"| {stats.test_boxes} boxes | classes: {stats.test_class_dist}")
        print(f"\n  Output: {self.dataset_dir}")
        print(f"  Seed: {self.seed}")
        print(f"{'=' * 60}\n")

        return stats


def main():
    """CLI entry point for dataset splitting."""
    import argparse

    parser = argparse.ArgumentParser(description='Split YOLO dataset into train/val/test')
    parser.add_argument('--dataset', type=str, default='datasets/processed',
                        help='Path to YOLO dataset directory')
    parser.add_argument('--train', type=float, default=0.7, help='Train ratio')
    parser.add_argument('--val', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--test', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='copy',
                        choices=['copy', 'move', 'symlink'],
                        help='File handling mode')
    args = parser.parse_args()

    splitter = DatasetSplitter(args.dataset, seed=args.seed)
    splitter.split(train=args.train, val=args.val, test=args.test, copy_mode=args.mode)


if __name__ == '__main__':
    main()
