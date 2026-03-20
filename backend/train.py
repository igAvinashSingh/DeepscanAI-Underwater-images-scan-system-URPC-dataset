#!/usr/bin/env python3
"""
Train YOLOv8 on URPC2020 Underwater Dataset
============================================
Dataset: National Underwater Robotics Professional Competition 2020
Classes: holothurian, echinus, scallop, starfish, waterweeds

Dataset Structure Expected:
  urpc2020/
  ├── images/
  │   ├── train/   (*.jpg)
  │   └── val/     (*.jpg)
  └── labels/
      ├── train/   (*.txt - YOLO format)
      └── val/     (*.txt)

YOLO Label Format per line:
  <class_id> <x_center> <y_center> <width> <height>  (all normalized 0-1)
"""

import os
import yaml
import argparse
from pathlib import Path

# ─── DATASET CONFIG ───────────────────────────────────────────

DATASET_CONFIG = {
    'path': str(Path(__file__).parent.parent / 'urpc2020'),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 5,
    'names': ['holothurian', 'echinus', 'scallop', 'starfish', 'waterweeds']
}

# ─── TRAINING HYPERPARAMETERS ─────────────────────────────────

TRAIN_ARGS = {
    # Model
    'model': 'yolov8m.pt',       # yolov8n/s/m/l/x.pt (n=fastest, x=best)

    # Training
    'epochs': 150,
    'imgsz': 640,
    'batch': 16,                  # reduce to 8 if OOM
    'workers': 8,

    # Optimization
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,

    # Augmentation (underwater-specific)
    'augment': True,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'flipud': 0.3,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.1,
    'copy_paste': 0.1,

    # Underwater-specific: simulate water color shifts
    'erasing': 0.4,

    # Output
    'project': 'runs/detect',
    'name': 'urpc2020_yolov8',
    'save': True,
    'save_period': 10,
    'cache': True,
    'device': '0',               # GPU ID; 'cpu' for CPU
    'exist_ok': True,
    'pretrained': True,
    'patience': 30,              # Early stopping
    'close_mosaic': 10,
}

# ─── FUNCTIONS ────────────────────────────────────────────────

def create_dataset_yaml(config: dict, output_path: str = 'urpc2020.yaml') -> str:
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Dataset YAML saved: {output_path}")
    return output_path


def convert_xml_to_yolo(xml_dir: str, output_dir: str, class_names: list):
    """
    Convert URPC2020 XML annotations to YOLO format.
    
    URPC2020 annotations are in Pascal VOC XML format.
    This function converts them to YOLO .txt format.
    """
    import xml.etree.ElementTree as ET

    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for xml_file in xml_dir.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get image dimensions
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)

            lines = []
            for obj in root.findall('object'):
                name = obj.find('name').text.lower().strip()

                # Map URPC2020 class names
                name_map = {
                    'holothurian': 'holothurian',
                    'sea cucumber': 'holothurian',
                    'echinus': 'echinus',
                    'sea urchin': 'echinus',
                    'scallop': 'scallop',
                    'starfish': 'starfish',
                    'sea star': 'starfish',
                    'waterweeds': 'waterweeds',
                    'aquatic plants': 'waterweeds',
                }
                name = name_map.get(name, name)

                if name not in class_names:
                    skipped += 1
                    continue

                cls_id = class_names.index(name)

                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # Convert to YOLO format (normalized center x, y, w, h)
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h

                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write label file
            label_file = output_dir / xml_file.with_suffix('.txt').name
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))

            converted += 1

        except Exception as e:
            print(f"  ⚠️  Error processing {xml_file.name}: {e}")
            skipped += 1

    print(f"✅ Converted {converted} files, skipped {skipped}")
    return converted


def train(args: dict):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return

    # Create dataset YAML
    yaml_path = create_dataset_yaml(DATASET_CONFIG)

    # Load model
    model = YOLO(args.pop('model'))

    # Train
    print("\n🚀 Starting YOLOv8 Training on URPC2020...")
    print(f"   Model: {args.get('model', 'yolov8m.pt')}")
    print(f"   Epochs: {args['epochs']}")
    print(f"   Image Size: {args['imgsz']}")
    print(f"   Device: {args['device']}")
    print()

    results = model.train(data=yaml_path, **args)
    print("\n✅ Training complete!")
    print(f"   Best weights: {results.save_dir}/weights/best.pt")
    return results


def evaluate(model_path: str):
    from ultralytics import YOLO
    yaml_path = create_dataset_yaml(DATASET_CONFIG)
    model = YOLO(model_path)
    metrics = model.val(data=yaml_path)
    print(f"\n📊 Evaluation Results:")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")
    return metrics


def export_model(model_path: str, format: str = 'onnx'):
    """Export to ONNX, TensorRT, CoreML etc for deployment."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.export(format=format, imgsz=640, simplify=True)
    print(f"✅ Model exported to {format} format")


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URPC2020 YOLOv8 Training Pipeline')
    subparsers = parser.add_subparsers(dest='command')

    # Convert command
    conv = subparsers.add_parser('convert', help='Convert XML annotations to YOLO format')
    conv.add_argument('--xml-dir', required=True, help='Directory with XML annotation files')
    conv.add_argument('--out-dir', required=True, help='Output directory for YOLO .txt files')

    # Train command
    tr = subparsers.add_parser('train', help='Train YOLOv8 model')
    tr.add_argument('--model', default='yolov8m.pt', help='Base model (yolov8n/s/m/l/x.pt)')
    tr.add_argument('--epochs', type=int, default=150)
    tr.add_argument('--batch', type=int, default=16)
    tr.add_argument('--device', default='0')

    # Eval command
    ev = subparsers.add_parser('eval', help='Evaluate trained model')
    ev.add_argument('--weights', required=True, help='Path to model weights (.pt)')

    # Export command
    ex = subparsers.add_parser('export', help='Export model for deployment')
    ex.add_argument('--weights', required=True)
    ex.add_argument('--format', default='onnx', choices=['onnx', 'tflite', 'coreml', 'engine'])

    args = parser.parse_args()

    if args.command == 'convert':
        convert_xml_to_yolo(args.xml_dir, args.out_dir, DATASET_CONFIG['names'])
    elif args.command == 'train':
        custom = {**TRAIN_ARGS, 'model': args.model, 'epochs': args.epochs,
                  'batch': args.batch, 'device': args.device}
        train(custom)
    elif args.command == 'eval':
        evaluate(args.weights)
    elif args.command == 'export':
        export_model(args.weights, args.format)
    else:
        parser.print_help()
