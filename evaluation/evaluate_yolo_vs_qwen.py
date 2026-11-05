"""
evaluate_yolo_vs_qwen.py — Comparative Evaluation: YOLOv8 vs Qwen3-VL

Compares YOLOv8-nano and Qwen3-VL (via Ollama) on the same test images.
Evaluates accuracy, confidence, and latency across target indoor belongings,
with special focus on keys and watch.

Includes warm-up handling to prevent model-loading latency from skewing averages.
Outputs detailed comparison reports in JSON and CSV formats.
"""

import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detect import ObjectDetector, CLASS_NAMES

try:
    from ollama import chat
except ImportError:
    chat = None

# Known ground truth mapping for test-data images
DEFAULT_GROUND_TRUTH = {
    "glasses.jpeg": "glasses",
    "glasses2.jpeg": "glasses",
    "k_wallet.jpeg": "wallet",
    "test_image.jpg": "unknown",
    "wallet.jpg": "wallet",
    "wallet2.jpeg": "wallet",
    "keys1.jpg": "keys",
    "keys2.jpg": "keys",
    "keys3.jpg": "keys",
    "keys4.jpg": "keys",
    "watch1.jpg": "watch",
    "watch2.jpg": "watch",
    "watch3.jpg": "watch",
    "watch4.jpg": "watch",
    "phone1.jpg": "phone",
    "phone2.jpg": "phone",
}

VALID_CLASSES = ["keys", "wallet", "phone", "watch", "glasses", "unknown"]


def infer_actual_class(filename: str, custom_gt: Optional[Dict[str, str]] = None) -> str:
    """Determine actual class from custom ground truth dict, known mapping, or filename heuristic."""
    if custom_gt and filename in custom_gt:
        return custom_gt[filename]
    if filename in DEFAULT_GROUND_TRUTH:
        return DEFAULT_GROUND_TRUTH[filename]

    stem = Path(filename).stem.lower()
    for cls_name in ["keys", "wallet", "phone", "watch", "glasses"]:
        if cls_name in stem:
            return cls_name
    if "key" in stem:
        return "keys"
    return "unknown"


def normalize_qwen_prediction(raw_text: str) -> str:
    """Normalize Qwen raw output to one of the target classes or unknown."""
    if not raw_text:
        return "unknown"

    cleaned = raw_text.strip().lower().strip(".,!?:;\"' \n\r")
    if cleaned in VALID_CLASSES:
        return cleaned

    # Check words
    words = cleaned.replace("\n", " ").split()
    for cls_name in ["keys", "wallet", "phone", "watch", "glasses", "unknown"]:
        if cls_name in words:
            return cls_name

    # Check substrings / synonyms
    if "key" in cleaned:
        return "keys"
    if "watch" in cleaned:
        return "watch"
    if "wallet" in cleaned:
        return "wallet"
    if "phone" in cleaned or "cellphone" in cleaned:
        return "phone"
    if "glass" in cleaned or "spectacle" in cleaned:
        return "glasses"
    if "unknown" in cleaned or "none" in cleaned:
        return "unknown"

    return cleaned


def warmup_models(detector: ObjectDetector, qwen_model: str, warmup_image_path: str):
    """
    Run warm-up inference on both models so first-request model loading
    does not distort evaluation latency.
    """
    print("=" * 60)
    print("  WARM-UP PHASE")
    print("=" * 60)

    frame = cv2.imread(warmup_image_path)
    if frame is None:
        # Create a synthetic warm-up frame if needed
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. YOLO Warmup
    t0 = time.perf_counter()
    _ = detector.detect(frame)
    yolo_warmup_time = time.perf_counter() - t0
    print(f"  [YOLO] Warm-up inference completed in: {yolo_warmup_time:.3f} s")

    # 2. Qwen Warmup
    qwen_warmup_time = None
    if chat is not None:
        prompt = (
            "Identify the main personal belonging visible in this image. "
            "Choose exactly ONE category: keys, wallet, phone, watch, glasses, unknown. "
            "Return only the category name."
        )
        t0 = time.perf_counter()
        try:
            _ = chat(
                model=qwen_model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [warmup_image_path]
                }],
                options={"temperature": 0}
            )
            qwen_warmup_time = time.perf_counter() - t0
            print(f"  [Qwen] Model loading & warm-up completed in: {qwen_warmup_time:.3f} s")
        except Exception as e:
            print(f"  [Qwen] Warm-up failed: {e}")
    else:
        print("  [Qwen] Ollama package not installed; skipping Qwen warm-up.")

    print("=" * 60)
    return {
        "yolo_warmup_s": round(yolo_warmup_time, 3),
        "qwen_warmup_s": round(qwen_warmup_time, 3) if qwen_warmup_time else None
    }


def run_comparative_evaluation(
    test_dir: Path,
    yolo_model_path: str,
    qwen_model: str = "qwen3-vl:2b",
    confidence_threshold: float = 0.25,
    custom_gt_path: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run evaluation on all images in test_dir for both YOLO and Qwen3-VL.
    Records actual class, predictions, confidence, inference time, and correctness.
    """
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    custom_gt = {}
    if custom_gt_path and os.path.exists(custom_gt_path):
        with open(custom_gt_path, "r") as f:
            custom_gt = json.load(f)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in test_dir.iterdir()
        if p.suffix.lower() in image_exts
    ])

    if not image_paths:
        raise ValueError(f"No test images found in {test_dir}")

    print(f"\nFound {len(image_paths)} test images in {test_dir.name}/")

    # Load YOLO detector
    print(f"\nLoading YOLO model from: {yolo_model_path}")
    detector = ObjectDetector(model_path=yolo_model_path, confidence_threshold=confidence_threshold)

    # Warm-up phase using the first image
    warmup_stats = warmup_models(detector, qwen_model, str(image_paths[0].resolve()))

    prompt = """
Identify the main personal belonging visible in this image.

Choose exactly ONE category:
keys, wallet, phone, watch, glasses, unknown.

Return only the category name.
Do not guess if the object is unclear.
"""

    results = []
    yolo_latencies = []
    qwen_latencies = []

    print("\n" + "=" * 60)
    print("  RUNNING COMPARATIVE INFERENCE")
    print("=" * 60)

    for idx, img_path in enumerate(image_paths, 1):
        filename = img_path.name
        actual_class = infer_actual_class(filename, custom_gt)
        print(f"\n[{idx}/{len(image_paths)}] Evaluating: {filename}")
        print(f"  Actual class: {actual_class}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [ERROR] Could not read image: {filename}")
            continue

        # --- YOLO Inference ---
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        yolo_latency_s = time.perf_counter() - t0
        yolo_latency_ms = yolo_latency_s * 1000
        yolo_latencies.append(yolo_latency_ms)

        if detections:
            # Pick highest confidence detection
            best_det = max(detections, key=lambda d: d.confidence)
            yolo_pred = best_det.class_name
            yolo_conf = round(float(best_det.confidence), 4)
        else:
            yolo_pred = "none"
            yolo_conf = 0.0

        # YOLO correctness
        if actual_class == "unknown":
            yolo_correct = (len(detections) == 0 or yolo_pred == "none")
        else:
            # Correct if top prediction matches actual class
            yolo_correct = (yolo_pred == actual_class)

        print(f"  YOLO Prediction: {yolo_pred} (conf: {yolo_conf:.3f}, {yolo_latency_ms:.1f} ms) -> {'CORRECT' if yolo_correct else 'INCORRECT'}")

        # --- Qwen3-VL Inference ---
        qwen_pred = "error"
        qwen_raw = ""
        qwen_latency_s = 0.0
        qwen_correct = False

        if chat is not None:
            try:
                t0 = time.perf_counter()
                response = chat(
                    model=qwen_model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [str(img_path.resolve())]
                    }],
                    options={"temperature": 0}
                )
                qwen_latency_s = time.perf_counter() - t0
                qwen_latencies.append(qwen_latency_s)
                qwen_raw = response.message.content.strip()
                qwen_pred = normalize_qwen_prediction(qwen_raw)
                qwen_correct = (qwen_pred == actual_class)

                print(f"  Qwen Prediction: {qwen_pred} (raw: '{qwen_raw}', {qwen_latency_s:.2f} s) -> {'CORRECT' if qwen_correct else 'INCORRECT'}")
            except Exception as e:
                print(f"  [Qwen ERROR]: {e}")
                qwen_pred = "error"
                qwen_raw = str(e)
        else:
            qwen_pred = "ollama_not_installed"

        record = {
            "image": filename,
            "actual_class": actual_class,
            "yolo_prediction": yolo_pred,
            "yolo_confidence": yolo_conf,
            "yolo_latency_ms": round(yolo_latency_ms, 2),
            "yolo_is_correct": bool(yolo_correct),
            "yolo_detections_count": len(detections),
            "yolo_all_detections": [
                {"class": d.class_name, "confidence": round(float(d.confidence), 4)}
                for d in detections
            ],
            "qwen_prediction": qwen_pred,
            "qwen_raw_response": qwen_raw,
            "qwen_inference_time_s": round(qwen_latency_s, 3),
            "qwen_latency_ms": round(qwen_latency_s * 1000, 1),
            "qwen_is_correct": bool(qwen_correct)
        }
        results.append(record)

    # --- Summary Metrics & Aggregation ---
    total_images = len(results)
    yolo_total_correct = sum(1 for r in results if r["yolo_is_correct"])
    qwen_total_correct = sum(1 for r in results if r["qwen_is_correct"])

    yolo_overall_acc = round(yolo_total_correct / total_images, 4) if total_images else 0
    qwen_overall_acc = round(qwen_total_correct / total_images, 4) if total_images else 0

    # Per-class metrics
    per_class_summary = {}
    classes_present = sorted(list(set(r["actual_class"] for r in results)))

    for cls in classes_present:
        cls_records = [r for r in results if r["actual_class"] == cls]
        cls_total = len(cls_records)
        cls_yolo_correct = sum(1 for r in cls_records if r["yolo_is_correct"])
        cls_qwen_correct = sum(1 for r in cls_records if r["qwen_is_correct"])

        yolo_confs = [r["yolo_confidence"] for r in cls_records if r["yolo_confidence"] > 0]
        avg_yolo_conf = round(float(np.mean(yolo_confs)), 4) if yolo_confs else 0.0

        per_class_summary[cls] = {
            "total_images": cls_total,
            "yolo_correct": cls_yolo_correct,
            "yolo_accuracy": round(cls_yolo_correct / cls_total, 4) if cls_total else 0,
            "yolo_avg_confidence": avg_yolo_conf,
            "qwen_correct": cls_qwen_correct,
            "qwen_accuracy": round(cls_qwen_correct / cls_total, 4) if cls_total else 0,
            "yolo_predictions": [r["yolo_prediction"] for r in cls_records],
            "qwen_predictions": [r["qwen_prediction"] for r in cls_records],
        }

    # Special attention section for keys and watch
    special_focus = {}
    for special_cls in ["keys", "watch"]:
        if special_cls in per_class_summary:
            special_focus[special_cls] = per_class_summary[special_cls]

    report = {
        "timestamp": datetime.now().isoformat(),
        "yolo_model": str(yolo_model_path),
        "qwen_model": qwen_model,
        "test_directory": str(test_dir),
        "total_images": total_images,
        "warmup": warmup_stats,
        "latency_analysis": {
            "yolo_avg_ms": round(float(np.mean(yolo_latencies)), 2) if yolo_latencies else 0.0,
            "yolo_p50_ms": round(float(np.percentile(yolo_latencies, 50)), 2) if yolo_latencies else 0.0,
            "yolo_p95_ms": round(float(np.percentile(yolo_latencies, 95)), 2) if yolo_latencies else 0.0,
            "qwen_avg_s": round(float(np.mean(qwen_latencies)), 3) if qwen_latencies else 0.0,
            "qwen_avg_ms": round(float(np.mean(qwen_latencies) * 1000), 1) if qwen_latencies else 0.0,
            "qwen_p50_s": round(float(np.percentile(qwen_latencies, 50)), 3) if qwen_latencies else 0.0,
            "qwen_p95_s": round(float(np.percentile(qwen_latencies, 95)), 3) if qwen_latencies else 0.0,
            "note": "Averages exclude initial model loading / warm-up latency."
        },
        "overall_accuracy": {
            "yolo_correct": yolo_total_correct,
            "yolo_accuracy": yolo_overall_acc,
            "qwen_correct": qwen_total_correct,
            "qwen_accuracy": qwen_overall_acc,
        },
        "per_class_summary": per_class_summary,
        "special_focus_keys_and_watch": special_focus,
        "image_results": results
    }

    # --- Save JSON and CSV ---
    if output_dir is None:
        output_dir = PROJECT_ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "yolo_vs_qwen_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OUTPUT] JSON report saved to: {json_path}")

    csv_path = output_dir / "yolo_vs_qwen_comparison.csv"
    csv_fieldnames = [
        "image",
        "actual_class",
        "yolo_prediction",
        "yolo_confidence",
        "yolo_is_correct",
        "yolo_latency_ms",
        "qwen_prediction",
        "qwen_is_correct",
        "qwen_inference_time_s",
        "qwen_raw_response"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image": r["image"],
                "actual_class": r["actual_class"],
                "yolo_prediction": r["yolo_prediction"],
                "yolo_confidence": r["yolo_confidence"],
                "yolo_is_correct": r["yolo_is_correct"],
                "yolo_latency_ms": r["yolo_latency_ms"],
                "qwen_prediction": r["qwen_prediction"],
                "qwen_is_correct": r["qwen_is_correct"],
                "qwen_inference_time_s": r["qwen_inference_time_s"],
                "qwen_raw_response": r["qwen_raw_response"]
            })
    print(f"[OUTPUT] CSV report saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'Image':<18} {'Actual':<9} {'YOLO Pred':<11} {'Conf':>6} {'YOLO Ok':<8} {'Qwen Pred':<11} {'Time(s)':>7} {'Qwen Ok':<8}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['image'][:17]:<18} "
            f"{r['actual_class']:<9} "
            f"{r['yolo_prediction']:<11} "
            f"{r['yolo_confidence']:>6.2f} "
            f"{'YES' if r['yolo_is_correct'] else 'NO':<8} "
            f"{r['qwen_prediction']:<11} "
            f"{r['qwen_inference_time_s']:>7.2f} "
            f"{'YES' if r['qwen_is_correct'] else 'NO':<8}"
        )
    print("=" * 78)

    print("\n--- PER-CLASS COMPARISON ---")
    print(f"{'Class':<12} {'Count':>6} {'YOLO Acc':>10} {'Avg YOLO Conf':>14} {'Qwen Acc':>10}")
    print("-" * 55)
    for cls, metrics in per_class_summary.items():
        print(
            f"{cls:<12} "
            f"{metrics['total_images']:>6} "
            f"{metrics['yolo_accuracy'] * 100:>9.1f}% "
            f"{metrics['yolo_avg_confidence']:>14.3f} "
            f"{metrics['qwen_accuracy'] * 100:>9.1f}%"
        )
    print("-" * 55)
    print(
        f"{'OVERALL':<12} "
        f"{total_images:>6} "
        f"{yolo_overall_acc * 100:>9.1f}% "
        f"{'-':>14} "
        f"{qwen_overall_acc * 100:>9.1f}%"
    )
    print("=" * 55)

    print("\n--- LATENCY COMPARISON (Excluding Warm-up) ---")
    print(f"  YOLO Average Latency : {report['latency_analysis']['yolo_avg_ms']} ms (p95: {report['latency_analysis']['yolo_p95_ms']} ms)")
    print(f"  Qwen Average Latency : {report['latency_analysis']['qwen_avg_s']} s ({report['latency_analysis']['qwen_avg_ms']} ms) (p95: {report['latency_analysis']['qwen_p95_s']} s)")
    print(f"  Initial Warm-up Time : YOLO: {warmup_stats['yolo_warmup_s']} s | Qwen: {warmup_stats['qwen_warmup_s']} s")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO vs Qwen3-VL on test images.")
    parser.add_argument("--test-dir", default="test-data", help="Directory of test images")
    parser.add_argument("--model", default="models/detector/best.pt", help="Path to YOLO weights")
    parser.add_argument("--qwen-model", default="qwen3-vl:2b", help="Ollama Qwen model tag")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--ground-truth", default=None, help="Optional JSON file mapping image filename to actual class")
    parser.add_argument("--output-dir", default="evaluation/results", help="Directory to save JSON and CSV reports")

    args = parser.parse_args()

    test_dir = PROJECT_ROOT / args.test_dir
    yolo_model_path = PROJECT_ROOT / args.model
    output_dir = PROJECT_ROOT / args.output_dir

    run_comparative_evaluation(
        test_dir=test_dir,
        yolo_model_path=str(yolo_model_path),
        qwen_model=args.qwen_model,
        confidence_threshold=args.conf,
        custom_gt_path=args.ground_truth,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()

