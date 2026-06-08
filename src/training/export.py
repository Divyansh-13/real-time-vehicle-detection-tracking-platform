"""
Model Export Utility
======================
Exports trained YOLOv8 models to production-ready formats.

Supported formats:
    - ONNX (cross-platform, TensorRT compatible)
    - TorchScript (PyTorch native deployment)
    - OpenVINO (Intel hardware acceleration)

Usage:
    python -m src.training.export --model runs/train/vehicle_detector/weights/best.pt --format onnx
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export YOLOv8 models to deployment formats.

    Why export:
    - ONNX is 2-3x faster than PyTorch for inference
    - ONNX is framework-agnostic (runs without PyTorch installed)
    - TensorRT (via ONNX) provides GPU-optimized inference
    - Smaller deployment footprint (no PyTorch dependency)
    """

    SUPPORTED_FORMATS = ['onnx', 'torchscript', 'openvino', 'engine', 'coreml', 'tflite']

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to trained .pt model file.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def export(
        self,
        format: str = 'onnx',
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Export model to the specified format.

        Args:
            format: Export format (onnx, torchscript, openvino, etc.).
            imgsz: Input image size.
            half: Use FP16 half precision (GPU only).
            dynamic: Enable dynamic batch size (ONNX only).
            simplify: Simplify ONNX graph.
            output_dir: Directory to save exported model. Defaults to models/.

        Returns:
            Path to the exported model file.
        """
        from ultralytics import YOLO

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. "
                           f"Supported: {self.SUPPORTED_FORMATS}")

        logger.info(f"Exporting {self.model_path} to {format}...")
        model = YOLO(str(self.model_path))

        export_kwargs = {
            'format': format,
            'imgsz': imgsz,
            'half': half,
        }

        if format == 'onnx':
            export_kwargs['dynamic'] = dynamic
            export_kwargs['simplify'] = simplify

        export_path = model.export(**export_kwargs)
        logger.info(f"Exported to: {export_path}")

        # Copy to output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / Path(export_path).name
            shutil.copy2(str(export_path), str(dest))
            logger.info(f"Copied to: {dest}")
            return str(dest)

        return str(export_path)

    def export_all(
        self,
        formats: Optional[List[str]] = None,
        imgsz: int = 640,
        output_dir: str = 'models',
    ) -> dict:
        """Export to multiple formats at once."""
        formats = formats or ['onnx', 'torchscript']
        results = {}
        for fmt in formats:
            try:
                path = self.export(format=fmt, imgsz=imgsz, output_dir=output_dir)
                results[fmt] = {'status': 'success', 'path': path}
            except Exception as e:
                results[fmt] = {'status': 'failed', 'error': str(e)}
                logger.error(f"Export to {fmt} failed: {e}")
        return results

    def get_model_info(self) -> dict:
        """Get model metadata."""
        from ultralytics import YOLO
        model = YOLO(str(self.model_path))
        return {
            'path': str(self.model_path),
            'size_mb': round(self.model_path.stat().st_size / 1024 / 1024, 2),
            'task': model.task if hasattr(model, 'task') else 'detect',
            'names': model.names if hasattr(model, 'names') else {},
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Export YOLOv8 model')
    parser.add_argument('--model', type=str, required=True, help='Path to .pt model')
    parser.add_argument('--format', type=str, default='onnx',
                       choices=ModelExporter.SUPPORTED_FORMATS)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    exporter = ModelExporter(args.model)
    exporter.export(format=args.format, imgsz=args.imgsz,
                   half=args.half, output_dir=args.output)


if __name__ == '__main__':
    main()
