"""
Data Augmentation Pipeline
============================
Augments a YOLO-format dataset using Albumentations for robust model training.

Why augmentation is critical for this dataset:
    - Only 301 source frames → need 10x expansion for good generalization
    - Single camera angle → augmentation simulates different viewpoints
    - Consistent lighting → need brightness/contrast variation for day/night
    - No weather effects → synthetic rain/fog improves robustness

Pipeline produces augmented copies while preserving originals.
Bounding boxes are transformed alongside images to maintain annotation accuracy.

Usage:
    augmenter = DataAugmenter("datasets/processed", "datasets/augmented")
    augmenter.augment(multiplier=10)
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_augmentation_pipeline():
    """
    Create the Albumentations augmentation pipeline.

    Each augmentation is chosen to address a specific weakness:

    - HorizontalFlip: Cars can travel in both directions
    - RandomBrightnessContrast: Day/night, shadow variation
    - HueSaturationValue: Car color diversity
    - GaussianBlur: Rain/fog/motion blur simulation
    - GaussNoise: Sensor noise in low-light cameras
    - CLAHE: Enhance detail in dark regions (tunnel exits, shade)
    - RandomScale: Simulate different camera zoom levels
    - Rotate: Camera tilt compensation
    - CoarseDropout: Partial occlusion simulation (poles, signs)

    Returns:
        Albumentations Compose pipeline with bbox support.
    """
    import albumentations as A

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.4,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.OneOf([
            A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
            A.RandomCrop(
                height=640,
                width=640,
                p=1.0,
            ),
        ], p=0.0),  # Disabled by default - needs image size adaptation
        A.Rotate(limit=10, border_mode=0, p=0.2),
        A.CoarseDropout(
            max_holes=5,
            max_height=50,
            max_width=50,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=0,
            p=0.15,
        ),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.2,
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,  # Drop boxes that become <30% visible after transform
        min_area=100,         # Drop boxes with area < 100 pixels
    ))


def yolo_to_albumentations(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Read YOLO label file and convert to Albumentations format.

    YOLO format: class_id x_center y_center width height
    Albumentations YOLO format: x_center y_center width height (class separate)

    Returns:
        Tuple of (bboxes, class_labels)
    """
    bboxes = []
    class_labels = []

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Clamp to valid range
            x_center = max(0.001, min(0.999, x_center))
            y_center = max(0.001, min(0.999, y_center))
            width = max(0.001, min(0.999, width))
            height = max(0.001, min(0.999, height))

            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)

    return bboxes, class_labels


def albumentations_to_yolo(bboxes: List, class_labels: List[int]) -> str:
    """
    Convert Albumentations format back to YOLO label file content.

    Returns:
        String content for a YOLO .txt label file.
    """
    lines = []
    for bbox, class_id in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return '\n'.join(lines) + '\n' if lines else ''


class DataAugmenter:
    """
    Augments a YOLO-format dataset with transformed copies.

    Strategy:
    - Each original image generates N augmented variants
    - Original images are preserved (not modified)
    - Augmented images are saved with suffix _aug_001, _aug_002, etc.
    - Bounding boxes are transformed consistently with images
    - Boxes that become too small or mostly invisible are dropped

    Output structure:
        output_dir/
        ├── images/
        │   ├── frame_000000.PNG           (original, copied)
        │   ├── frame_000000_aug_001.png   (augmented)
        │   ├── frame_000000_aug_002.png
        │   └── ...
        └── labels/
            ├── frame_000000.txt           (original, copied)
            ├── frame_000000_aug_001.txt   (augmented)
            └── ...
    """

    IMAGE_EXTENSIONS = {'.png', '.PNG', '.jpg', '.jpeg', '.JPG'}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        seed: int = 42,
    ):
        """
        Args:
            input_dir: Path to source YOLO dataset (images/all/ and labels/all/).
            output_dir: Path to write augmented dataset.
            seed: Random seed for reproducibility.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)

    def augment(
        self,
        multiplier: int = 10,
        include_originals: bool = True,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Run augmentation pipeline.

        Args:
            multiplier: Number of augmented copies per original image.
            include_originals: Whether to copy originals to output.
            splits: List of splits to augment ['train'] or ['train', 'val'].
                   Defaults to ['all'] if no splits exist, or ['train'].

        Returns:
            Dict with counts: {'originals': N, 'augmented': M, 'total': N+M}
        """
        try:
            import albumentations  # noqa: F401
            import cv2
        except ImportError:
            logger.error(
                "Albumentations and OpenCV are required for augmentation.\n"
                "Install with: pip install albumentations opencv-python"
            )
            return {'originals': 0, 'augmented': 0, 'total': 0}

        pipeline = get_augmentation_pipeline()

        # Determine which splits to process
        if splits is None:
            if (self.input_dir / 'images' / 'train').exists():
                splits = ['train']  # Only augment training data
            elif (self.input_dir / 'images' / 'all').exists():
                splits = ['all']
            else:
                raise FileNotFoundError(
                    f"No images found in {self.input_dir}/images/train or images/all"
                )

        total_originals = 0
        total_augmented = 0

        for split in splits:
            images_dir = self.input_dir / 'images' / split
            labels_dir = self.input_dir / 'labels' / split

            if not images_dir.exists():
                logger.warning(f"Skipping split '{split}': {images_dir} not found")
                continue

            out_images = self.output_dir / 'images' / split
            out_labels = self.output_dir / 'labels' / split
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            image_files = sorted([
                f for f in images_dir.iterdir()
                if f.suffix in self.IMAGE_EXTENSIONS
            ])

            logger.info(f"Augmenting {len(image_files)} images from {split} "
                        f"(x{multiplier} = {len(image_files) * multiplier} new images)")

            for img_path in image_files:
                label_path = labels_dir / (img_path.stem + '.txt')

                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Cannot read image: {img_path}")
                    continue

                # Read labels
                if label_path.exists():
                    bboxes, class_labels = yolo_to_albumentations(label_path)
                else:
                    bboxes, class_labels = [], []

                # Copy original
                if include_originals:
                    import shutil
                    shutil.copy2(img_path, out_images / img_path.name)
                    if label_path.exists():
                        shutil.copy2(label_path, out_labels / label_path.name)
                    total_originals += 1

                # Generate augmented copies
                for aug_idx in range(1, multiplier + 1):
                    try:
                        augmented = pipeline(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels,
                        )

                        aug_image = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_classes = augmented['class_labels']

                        # Save augmented image
                        aug_stem = f"{img_path.stem}_aug_{aug_idx:03d}"
                        aug_img_path = out_images / f"{aug_stem}.png"
                        cv2.imwrite(str(aug_img_path), aug_image)

                        # Save augmented labels
                        aug_label_path = out_labels / f"{aug_stem}.txt"
                        label_content = albumentations_to_yolo(aug_bboxes, aug_classes)
                        with open(aug_label_path, 'w') as f:
                            f.write(label_content)

                        total_augmented += 1

                    except Exception as e:
                        logger.warning(f"Augmentation failed for {img_path.name} "
                                       f"(aug #{aug_idx}): {e}")

            logger.info(f"Split '{split}': {total_originals} originals + "
                        f"{total_augmented} augmented")

        total = total_originals + total_augmented
        print(f"\n{'=' * 60}")
        print("  AUGMENTATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Originals:  {total_originals}")
        print(f"  Augmented:  {total_augmented}")
        print(f"  Total:      {total}")
        print(f"  Multiplier: {multiplier}x")
        print(f"  Output:     {self.output_dir}")
        print(f"{'=' * 60}\n")

        return {
            'originals': total_originals,
            'augmented': total_augmented,
            'total': total,
        }


def main():
    """CLI entry point for data augmentation."""
    import argparse

    parser = argparse.ArgumentParser(description='Augment YOLO format dataset')
    parser.add_argument('--input', type=str, default='datasets/processed',
                        help='Input dataset directory')
    parser.add_argument('--output', type=str, default='datasets/augmented',
                        help='Output directory for augmented data')
    parser.add_argument('--multiplier', type=int, default=10,
                        help='Number of augmented copies per image')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    augmenter = DataAugmenter(args.input, args.output, seed=args.seed)
    augmenter.augment(multiplier=args.multiplier)


if __name__ == '__main__':
    main()
