"""
YOLOv8 Training Script
========================
Trains a YOLOv8 model on the vehicle detection dataset with full
experiment tracking, validation, and model export.

Architecture:
    1. Load training configuration (YAML)
    2. Initialize MLflow experiment tracking
    3. Load pre-trained YOLOv8 model (transfer learning)
    4. Train on custom dataset
    5. Validate on held-out data
    6. Export best model to ONNX
    7. Log all metrics, artifacts, and model to MLflow

Why transfer learning:
    - YOLOv8 pre-trained on COCO (80 classes including 'car')
    - Our dataset is small (301 frames) → fine-tuning is essential
    - Pre-trained backbone already knows edge/texture/shape features
    - Only need to adapt detection head to our 2 classes

Usage:
    python -m src.training.train --config configs/training_config.yaml

    Or from Python:
        from src.training.train import Trainer
        trainer = Trainer("configs/training_config.yaml")
        results = trainer.train()
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class Trainer:
    """
    YOLOv8 model trainer with MLflow integration.

    This trainer wraps the Ultralytics YOLO training API with:
    - Configuration management (YAML-based)
    - MLflow experiment tracking (metrics, params, artifacts)
    - Automatic model export (ONNX, TorchScript)
    - Training resumption support
    - Comprehensive logging

    Design decision: We use Ultralytics' built-in training loop rather than
    writing a custom PyTorch training loop because:
    1. Ultralytics handles all YOLO-specific training details (mosaic, mixup, etc.)
    2. Built-in learning rate scheduling, EMA, and mixed precision
    3. Automatic validation after each epoch
    4. Built-in early stopping
    5. Saves best.pt and last.pt automatically
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to YAML training configuration file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.results = None
        self.start_time = None

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate training configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set defaults for missing keys
        defaults = {
            'model': 'yolov8n.pt',
            'dataset': 'datasets/processed/dataset.yaml',
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'patience': 20,
            'device': '0',
            'workers': 8,
            'project': 'runs/train',
            'name': 'vehicle_detector',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'close_mosaic': 10,
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'flipud': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'save': True,
            'save_period': -1,
            'val': True,
            'plots': True,
            'export_onnx': True,
            'mlflow_tracking': True,
            'mlflow_experiment': 'vehicle-detection',
            'mlflow_tracking_uri': 'mlruns',
        }

        for key, default_val in defaults.items():
            if key not in config:
                config[key] = default_val

        logger.info(f"Loaded config from {self.config_path}")
        return config

    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        if not self.config.get('mlflow_tracking', False):
            return

        try:
            import mlflow

            tracking_uri = self.config.get('mlflow_tracking_uri', 'mlruns')
            if not tracking_uri.startswith(('http://', 'https://', 'file://')):
                tracking_uri = str(Path(tracking_uri).resolve())

            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = self.config.get('mlflow_experiment', 'vehicle-detection')
            mlflow.set_experiment(experiment_name)

            logger.info(f"MLflow tracking URI: {tracking_uri}")
            logger.info(f"MLflow experiment: {experiment_name}")

        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            self.config['mlflow_tracking'] = False

    def _log_to_mlflow(self, results):
        """Log training results to MLflow."""
        if not self.config.get('mlflow_tracking', False):
            return

        import mlflow

        with mlflow.start_run(run_name=f"{self.config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            params_to_log = [
                'model', 'epochs', 'imgsz', 'batch', 'lr0', 'lrf',
                'momentum', 'weight_decay', 'patience', 'optimizer',
                'mosaic', 'mixup', 'augment', 'device',
            ]
            for param in params_to_log:
                if param in self.config:
                    mlflow.log_param(param, self.config[param])

            mlflow.log_param('dataset', self.config['dataset'])
            mlflow.log_param('training_time_seconds',
                             time.time() - self.start_time if self.start_time else 0)

            # Log metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        # Clean up metric names for MLflow
                        clean_key = key.replace('/', '_').replace('(', '').replace(')', '')
                        mlflow.log_metric(clean_key, value)

            # Log artifacts
            save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
            if save_dir and save_dir.exists():
                # Log best model
                best_model = save_dir / 'weights' / 'best.pt'
                if best_model.exists():
                    mlflow.log_artifact(str(best_model), 'model')

                # Log training plots
                for plot_file in save_dir.glob('*.png'):
                    mlflow.log_artifact(str(plot_file), 'plots')

                # Log results CSV
                results_csv = save_dir / 'results.csv'
                if results_csv.exists():
                    mlflow.log_artifact(str(results_csv), 'metrics')

                # Log confusion matrix
                confusion_matrix = save_dir / 'confusion_matrix.png'
                if confusion_matrix.exists():
                    mlflow.log_artifact(str(confusion_matrix), 'plots')

            logger.info("Results logged to MLflow successfully")

    def train(self) -> Any:
        """
        Run the full training pipeline.

        Steps:
            1. Setup MLflow tracking
            2. Load pre-trained YOLOv8 model
            3. Train on custom dataset
            4. Log results to MLflow
            5. Export to ONNX (optional)
            6. Return results object

        Returns:
            Ultralytics Results object with training metrics.
        """
        from ultralytics import YOLO

        self._setup_mlflow()
        self.start_time = time.time()

        # Load model
        model_path = self.config['model']
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Prepare training arguments
        train_args = {
            'data': self.config['dataset'],
            'epochs': self.config['epochs'],
            'imgsz': self.config['imgsz'],
            'batch': self.config['batch'],
            'patience': self.config['patience'],
            'device': self.config['device'],
            'workers': self.config['workers'],
            'project': self.config['project'],
            'name': self.config['name'],
            'exist_ok': self.config['exist_ok'],
            'pretrained': self.config['pretrained'],
            'optimizer': self.config['optimizer'],
            'lr0': self.config['lr0'],
            'lrf': self.config['lrf'],
            'momentum': self.config['momentum'],
            'weight_decay': self.config['weight_decay'],
            'warmup_epochs': self.config['warmup_epochs'],
            'warmup_momentum': self.config['warmup_momentum'],
            'warmup_bias_lr': self.config['warmup_bias_lr'],
            'close_mosaic': self.config['close_mosaic'],
            'augment': self.config['augment'],
            'mosaic': self.config['mosaic'],
            'mixup': self.config['mixup'],
            'copy_paste': self.config['copy_paste'],
            'degrees': self.config['degrees'],
            'translate': self.config['translate'],
            'scale': self.config['scale'],
            'fliplr': self.config['fliplr'],
            'flipud': self.config['flipud'],
            'hsv_h': self.config['hsv_h'],
            'hsv_s': self.config['hsv_s'],
            'hsv_v': self.config['hsv_v'],
            'save': self.config['save'],
            'save_period': self.config['save_period'],
            'val': self.config['val'],
            'plots': self.config['plots'],
        }

        # Train
        logger.info("=" * 60)
        logger.info("  STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"  Model:    {model_path}")
        logger.info(f"  Dataset:  {self.config['dataset']}")
        logger.info(f"  Epochs:   {self.config['epochs']}")
        logger.info(f"  Img Size: {self.config['imgsz']}")
        logger.info(f"  Batch:    {self.config['batch']}")
        logger.info(f"  Device:   {self.config['device']}")
        logger.info("=" * 60)

        self.results = self.model.train(**train_args)

        training_time = time.time() - self.start_time
        logger.info(f"Training completed in {training_time/60:.1f} minutes")

        # Log to MLflow
        self._log_to_mlflow(self.results)

        # Export to ONNX
        if self.config.get('export_onnx', True):
            self._export_onnx()

        return self.results

    def _export_onnx(self):
        """Export the best model to ONNX format for production inference."""
        if self.model is None:
            logger.warning("No model to export")
            return

        try:
            # Load best weights
            save_dir = Path(self.results.save_dir)
            best_model_path = save_dir / 'weights' / 'best.pt'

            if best_model_path.exists():
                best_model = __import__('ultralytics', fromlist=['YOLO']).YOLO(str(best_model_path))
                export_path = best_model.export(format='onnx', imgsz=self.config['imgsz'])
                logger.info(f"Model exported to ONNX: {export_path}")

                # Copy to models/ directory
                models_dir = Path('models')
                models_dir.mkdir(exist_ok=True)

                import shutil
                onnx_dest = models_dir / f"{self.config['name']}.onnx"
                pt_dest = models_dir / f"{self.config['name']}.pt"
                shutil.copy2(str(best_model_path), str(pt_dest))
                if Path(export_path).exists():
                    shutil.copy2(str(export_path), str(onnx_dest))

                logger.info(f"Models copied to: {models_dir}")
            else:
                logger.warning(f"Best model not found at {best_model_path}")

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

    def validate(self, model_path: Optional[str] = None) -> Any:
        """
        Run validation on the test set.

        Args:
            model_path: Path to model weights. Uses best.pt from training if None.

        Returns:
            Validation results object.
        """
        from ultralytics import YOLO

        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise RuntimeError("No model available. Train first or provide model_path.")

        results = model.val(
            data=self.config['dataset'],
            imgsz=self.config['imgsz'],
            batch=self.config['batch'],
            device=self.config['device'],
            split='test',
        )

        logger.info(f"Validation mAP@50: {results.box.map50:.4f}")
        logger.info(f"Validation mAP@50-95: {results.box.map:.4f}")

        return results


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8 vehicle detector')
    parser.add_argument('--config', type=str,
                        default='configs/training_config.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run validation only (no training)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model for validation')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    trainer = Trainer(args.config)

    if args.validate_only:
        trainer.validate(model_path=args.model)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
