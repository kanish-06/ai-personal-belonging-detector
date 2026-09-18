"""
train_yolov8.py — YOLOv8 Transfer Learning Training Script

Fine-tunes YOLOv8-nano on the merged 5-class personal belonging dataset.
Can be run locally (with GPU) or adapted for Google Colab.

Usage:
    python training/train_yolov8.py --data dataset/data.yaml --epochs 100

For Colab, use the companion notebook: train_yolov8_colab.ipynb
"""

import os
import sys
import argparse

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_gpu():
    """Check GPU availability and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU available: {gpu_name} ({gpu_mem:.1f} GB)")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ Apple MPS (Metal) GPU available")
            return 'mps'
        else:
            print("✗ No GPU available — training will be slow on CPU")
            return 'cpu'
    except ImportError:
        print("✗ PyTorch not installed")
        return 'cpu'


def train(args):
    """Run YOLOv8 transfer learning training."""
    from ultralytics import YOLO

    # Check GPU
    device = check_gpu()
    if device == 'cpu':
        print("\n⚠ WARNING: Training on CPU is very slow. Consider using Google Colab.")
        print("  See: training/train_yolov8_colab.ipynb\n")

    # Resolve paths
    data_yaml = os.path.abspath(args.data)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_yaml):
        print(f"ERROR: Dataset config not found: {data_yaml}")
        print("  Download and merge datasets first (see PROJECT_BUILD_SPEC.md Section 5)")
        sys.exit(1)

    # Load pretrained YOLOv8-nano
    print(f"\nLoading pretrained model: {args.model}")
    model = YOLO(args.model)

    # Training configuration
    train_args = {
        'data': data_yaml,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': 0 if device == 'cuda' else device,
        'patience': args.patience,          # Early stopping patience
        'save': True,
        'save_period': 10,                  # Save checkpoint every 10 epochs
        'project': output_dir,
        'name': 'belonging_detector',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': args.lr,
        'lrf': 0.01,                        # Final learning rate factor
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'weight_decay': 0.0005,
        'cos_lr': True,                     # Cosine LR scheduler
        'close_mosaic': 10,                 # Disable mosaic for last 10 epochs

        # Data augmentation (most enabled by default in ultralytics)
        'hsv_h': 0.015,                     # HSV-Hue augmentation
        'hsv_s': 0.7,                       # HSV-Saturation augmentation
        'hsv_v': 0.4,                       # HSV-Value augmentation
        'degrees': 10.0,                    # Rotation degrees
        'translate': 0.1,                   # Translation
        'scale': 0.5,                       # Scale augmentation
        'shear': 2.0,                       # Shear degrees
        'flipud': 0.0,                      # Vertical flip (off — objects have orientation)
        'fliplr': 0.5,                      # Horizontal flip
        'mosaic': 1.0,                      # Mosaic augmentation
        'mixup': 0.1,                       # Mixup augmentation
    }

    if device == 'cpu':
        train_args['device'] = 'cpu'

    print(f"\nTraining configuration:")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}/belonging_detector/")
    print()

    # Train!
    results = model.train(**train_args)

    # Copy best weights to models/detector/
    best_pt_src = os.path.join(output_dir, 'belonging_detector', 'weights', 'best.pt')
    best_pt_dst = os.path.join(PROJECT_ROOT, 'models', 'detector', 'best.pt')

    if os.path.exists(best_pt_src):
        import shutil
        os.makedirs(os.path.dirname(best_pt_dst), exist_ok=True)
        shutil.copy2(best_pt_src, best_pt_dst)
        print(f"\n✓ Best weights copied to: {best_pt_dst}")
    else:
        print(f"\n⚠ Could not find best.pt at: {best_pt_src}")

    # Validation
    print("\n--- Validation Results ---")
    val_results = model.val()
    print(f"  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")

    # Export to ONNX (for deployment)
    if args.export_onnx:
        print("\nExporting to ONNX format...")
        model.export(format='onnx', imgsz=args.imgsz)
        print("✓ ONNX export complete")

    # Export to TFLite (for Raspberry Pi)
    if args.export_tflite:
        print("\nExporting to TFLite format...")
        model.export(format='tflite', imgsz=args.imgsz)
        print("✓ TFLite export complete")

    print("\n✓ Training complete!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-nano on the 5-class belonging detection dataset",
    )
    parser.add_argument(
        "--data", type=str, 
        default=os.path.join(PROJECT_ROOT, "dataset", "data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Pretrained model to start from (default: yolov8n.pt = nano)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (default: 8, optimal for ~6GB VRAM GPUs)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience in epochs (default: 20)",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(PROJECT_ROOT, "runs"),
        help="Output directory for training runs",
    )
    parser.add_argument(
        "--export-onnx", action="store_true",
        help="Export model to ONNX format after training",
    )
    parser.add_argument(
        "--export-tflite", action="store_true",
        help="Export model to TFLite format after training (for Raspberry Pi)",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
