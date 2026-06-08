"""Quick training script — runs YOLOv8 training on the vehicle dataset."""
import os
import shutil

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
os.chdir(r'c:\Projects\Car Tracking & Object Detection Dataset')

from ultralytics import YOLO

print('=' * 60)
print('  YOLOv8 TRAINING - 25 epochs on CPU')
print('  Dataset: 210 train / 60 val images')
print('  Estimated time: ~30-60 min')
print('=' * 60)

model = YOLO('yolov8n.pt')

results = model.train(
    data='datasets/processed/dataset.yaml',
    epochs=25,
    imgsz=640,
    batch=8,
    patience=10,
    device='cpu',
    workers=0,
    project='runs/train',
    name='vehicle_detector_v2',
    exist_ok=True,
    pretrained=True,
    verbose=True,
    val=True,
    plots=True,
    save=True,
)

print()
print('=' * 60)
print('  TRAINING COMPLETE!')
print('=' * 60)

# Print final metrics
if hasattr(results, 'results_dict'):
    rd = results.results_dict
    for key, val in rd.items():
        if 'metrics' in key:
            print(f'  {key}: {val:.4f}')

# Copy best model to models/
for search_root in ['runs/train', 'runs/detect']:
    if not os.path.exists(search_root):
        continue
    for root, dirs, files in os.walk(search_root):
        if 'best.pt' in files:
            src = os.path.join(root, 'best.pt')
            os.makedirs('models', exist_ok=True)
            shutil.copy2(src, 'models/vehicle_detector.pt')
            size_mb = os.path.getsize(src) / 1024 / 1024
            print(f'  Model saved: models/vehicle_detector.pt ({size_mb:.1f} MB)')
            break

print('=' * 60)
