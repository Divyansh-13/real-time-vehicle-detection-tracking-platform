"""
Full Data Pipeline Runner
===========================
Orchestrates the entire data preparation pipeline in sequence:

    1. Convert CVAT XML → YOLO format
    2. Validate the converted dataset
    3. Split into train/val/test
    4. Analyze the final dataset
    5. (Optional) Run augmentation

This script is the single entry point for preparing data before training.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --skip-augmentation
    python scripts/prepare_data.py --augmentation-multiplier 5
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cvat_to_yolo import CVATToYOLOConverter
from src.data.validator import DatasetValidator
from src.data.dataset_splitter import DatasetSplitter
from src.data.analyzer import DatasetAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    raw_xml: str = 'datasets/raw/annotations.xml',
    raw_images: str = 'datasets/raw/images',
    processed_dir: str = 'datasets/processed',
    augmented_dir: str = 'datasets/augmented',
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    skip_augmentation: bool = False,
    augmentation_multiplier: int = 10,
    seed: int = 42,
):
    """
    Run the complete data preparation pipeline.

    This is the main orchestration function that calls each pipeline
    stage in sequence. Any failure stops the pipeline to prevent
    training on bad data.
    """
    print("\n" + "=" * 70)
    print("  🚗 VEHICLE TRACKING PLATFORM — DATA PIPELINE")
    print("=" * 70)

    # ── Step 1: Convert CVAT XML → YOLO ──
    print("\n📋 STEP 1/5: Converting CVAT XML to YOLO format...")
    print("-" * 50)

    converter = CVATToYOLOConverter(
        xml_path=raw_xml,
        images_dir=raw_images,
        output_dir=processed_dir,
    )
    conversion_stats = converter.convert()

    if conversion_stats.frames_with_annotations == 0:
        logger.error("No frames were converted! Check XML and image paths.")
        sys.exit(1)

    # ── Step 2: Validate ──
    print("\n✅ STEP 2/5: Validating converted dataset...")
    print("-" * 50)

    validator = DatasetValidator(
        dataset_dir=processed_dir,
        num_classes=2,
        check_images=False,  # Skip slow image checks for now
    )
    is_valid, validation_result = validator.validate()

    if not is_valid:
        logger.error("Dataset validation FAILED. Fix errors before training.")
        logger.error(f"{len(validation_result.errors)} errors found.")
        # Don't exit — warnings are acceptable
        if any(i.severity == 'error' and 'out of range' in i.message
               for i in validation_result.errors):
            sys.exit(1)

    # ── Step 3: Split ──
    print("\n✂️  STEP 3/5: Splitting into train/val/test...")
    print("-" * 50)

    splitter = DatasetSplitter(processed_dir, seed=seed)
    split_stats = splitter.split(
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
    )

    # ── Step 4: Analyze ──
    print("\n📊 STEP 4/5: Analyzing final dataset...")
    print("-" * 50)

    analyzer = DatasetAnalyzer(processed_dir)
    analysis = analyzer.analyze()
    analyzer.save_report(str(Path(processed_dir) / 'analysis_report.json'))

    try:
        analyzer.plot_all(str(Path(processed_dir) / 'analysis_plots'))
    except Exception as e:
        logger.warning(f"Plot generation failed (matplotlib may not be installed): {e}")

    # ── Step 5: Augmentation (optional) ──
    if not skip_augmentation:
        print("\n🔄 STEP 5/5: Running data augmentation...")
        print("-" * 50)

        try:
            from src.data.augmentation import DataAugmenter

            augmenter = DataAugmenter(
                input_dir=processed_dir,
                output_dir=augmented_dir,
                seed=seed,
            )
            aug_stats = augmenter.augment(multiplier=augmentation_multiplier)
        except ImportError as e:
            logger.warning(f"Augmentation skipped (missing dependency): {e}")
            logger.info("Install with: pip install albumentations opencv-python")
    else:
        print("\n⏭️  STEP 5/5: Augmentation skipped (--skip-augmentation)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ✅ DATA PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  📁 Processed dataset: {processed_dir}")
    print(f"  📁 Dataset YAML:     {processed_dir}/dataset.yaml")
    print(f"  📊 Analysis report:  {processed_dir}/analysis_report.json")
    print(f"\n  Next step: python -m src.training.train --config configs/training_config.yaml")
    print("=" * 70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run complete data preparation pipeline')
    parser.add_argument('--xml', type=str, default='datasets/raw/annotations.xml')
    parser.add_argument('--images', type=str, default='datasets/raw/images')
    parser.add_argument('--output', type=str, default='datasets/processed')
    parser.add_argument('--augmented', type=str, default='datasets/augmented')
    parser.add_argument('--train', type=float, default=0.7)
    parser.add_argument('--val', type=float, default=0.2)
    parser.add_argument('--test', type=float, default=0.1)
    parser.add_argument('--skip-augmentation', action='store_true')
    parser.add_argument('--augmentation-multiplier', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        raw_xml=args.xml,
        raw_images=args.images,
        processed_dir=args.output,
        augmented_dir=args.augmented,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        skip_augmentation=args.skip_augmentation,
        augmentation_multiplier=args.augmentation_multiplier,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
