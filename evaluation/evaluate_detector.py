"""
evaluate_detector.py — Object Detector Evaluation

Computes mAP, precision, recall, and per-class metrics on a held-out
validation set using the trained YOLOv8 model.

Usage:
    python evaluation/evaluate_detector.py --model models/detector/best.pt --data dataset/data.yaml
"""

import os
import sys
import argparse
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from detect import CLASS_NAMES


def evaluate_detector(model_path, data_yaml, imgsz=640, conf=0.25, output_dir=None):
    """
    Run YOLOv8 validation and generate a detailed evaluation report.

    Args:
        model_path (str): Path to trained YOLOv8 weights.
        data_yaml (str): Path to data.yaml.
        imgsz (int): Image size for evaluation.
        conf (float): Confidence threshold.
        output_dir (str): Directory to save evaluation results.

    Returns:
        dict: Evaluation metrics.
    """
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    if not os.path.exists(data_yaml):
        print(f"ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    print("=" * 60)
    print("  DETECTOR EVALUATION")
    print("=" * 60)

    # Load model
    model = YOLO(model_path)

    # Run validation
    print(f"\nRunning validation on: {data_yaml}")
    print(f"  Image size: {imgsz}")
    print(f"  Confidence: {conf}")
    print()

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        save_json=True,
        plots=True,
    )

    # Extract metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'data_yaml': data_yaml,
        'overall': {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        },
        'per_class': {},
    }

    # Per-class metrics
    print("\n--- Per-Class Results ---")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 54)

    for i, class_name in CLASS_NAMES.items():
        if i < len(results.box.ap50):
            ap50 = float(results.box.ap50[i])
            ap = float(results.box.ap[i])
            # Per-class precision/recall from the results
            p = float(results.box.p[i]) if hasattr(results.box, 'p') and i < len(results.box.p) else 0
            r = float(results.box.r[i]) if hasattr(results.box, 'r') and i < len(results.box.r) else 0

            metrics['per_class'][class_name] = {
                'precision': p,
                'recall': r,
                'mAP50': ap50,
                'mAP50_95': ap,
            }
            print(f"{class_name:<12} {p:>10.4f} {r:>10.4f} {ap50:>10.4f} {ap:>10.4f}")

    print("-" * 54)
    print(f"{'Overall':<12} {metrics['overall']['precision']:>10.4f} "
          f"{metrics['overall']['recall']:>10.4f} "
          f"{metrics['overall']['mAP50']:>10.4f} "
          f"{metrics['overall']['mAP50_95']:>10.4f}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'detector_evaluation.json')
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Report saved to: {report_path}")

    # Flag underperforming classes
    print("\n--- Diagnostics ---")
    for cls_name, cls_metrics in metrics['per_class'].items():
        if cls_metrics['mAP50'] < 0.5:
            print(f"  ⚠ {cls_name}: mAP50 = {cls_metrics['mAP50']:.4f} — consider adding more training data")
        elif cls_metrics['mAP50'] < 0.7:
            print(f"  △ {cls_name}: mAP50 = {cls_metrics['mAP50']:.4f} — acceptable but could improve")
        else:
            print(f"  ✓ {cls_name}: mAP50 = {cls_metrics['mAP50']:.4f} — good")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained object detector")
    parser.add_argument("--model", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "detector", "best.pt"),
                        help="Path to trained model weights")
    parser.add_argument("--data", type=str,
                        default=os.path.join(PROJECT_ROOT, "dataset", "data.yaml"),
                        help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, "evaluation", "results"),
                        help="Output directory for evaluation results")

    args = parser.parse_args()
    evaluate_detector(args.model, args.data, args.imgsz, args.conf, args.output)


if __name__ == "__main__":
    main()
