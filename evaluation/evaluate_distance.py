"""
evaluate_distance.py — ANFIS Distance Estimation Evaluation

Compares ANFIS-based distance estimation against a naive fixed pinhole-camera
formula baseline on a set of known-distance test shots.

Usage:
    python evaluation/evaluate_distance.py
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def naive_pinhole_distance(box_area_ratio, k=90.0):
    """
    Naive baseline: fixed pinhole camera formula.
    
    distance = sqrt(k / box_area_ratio)
    
    Args:
        box_area_ratio (float): Bounding box area / frame area.
        k (float): Calibration constant.
    
    Returns:
        float: Estimated distance in cm.
    """
    if box_area_ratio <= 0:
        return 300.0
    return float(np.sqrt(k / box_area_ratio))


def evaluate_distance(anfis_model_path=None, test_data_path=None, output_dir=None):
    """
    Evaluate ANFIS distance estimation vs naive baseline.

    If no test data file is provided, uses synthetic test data.
    
    Test data CSV format: box_area_ratio,true_distance_cm
    
    Args:
        anfis_model_path (str): Path to trained ANFIS model.
        test_data_path (str): Path to test data CSV file.
        output_dir (str): Directory to save results.
    
    Returns:
        dict: Evaluation metrics.
    """
    from anfis_distance import ANFISDistanceEstimator, create_sample_calibration_data

    print("=" * 60)
    print("  DISTANCE ESTIMATION EVALUATION")
    print("=" * 60)

    # Load test data
    if test_data_path and os.path.exists(test_data_path):
        print(f"\nLoading test data from: {test_data_path}")
        data = np.loadtxt(test_data_path, delimiter=',', skiprows=1)
        test_ratios = data[:, 0]
        test_distances = data[:, 1]
    else:
        print("\nUsing synthetic test data (replace with real calibration data)")
        test_ratios, test_distances = create_sample_calibration_data()
        # Use a subset as "test" data
        np.random.seed(99)
        indices = np.random.choice(len(test_ratios), size=min(20, len(test_ratios)), replace=False)
        test_ratios = test_ratios[indices]
        test_distances = test_distances[indices]

    print(f"  Test samples: {len(test_ratios)}")
    print(f"  Distance range: [{test_distances.min():.0f}, {test_distances.max():.0f}] cm")

    # ── ANFIS predictions ─────────────────────────────────────────────────
    estimator = ANFISDistanceEstimator()
    anfis_path = anfis_model_path or estimator.DEFAULT_MODEL_PATH

    anfis_available = os.path.exists(anfis_path)
    if anfis_available:
        estimator.load(anfis_path)
        anfis_preds = estimator.predict_batch(test_ratios)
    else:
        print(f"\n⚠ ANFIS model not found at: {anfis_path}")
        print("  Using fallback heuristic for ANFIS predictions.")
        estimator._is_trained = True
        anfis_preds = np.array([estimator._fallback_predict(r) for r in test_ratios])

    # ── Naive baseline predictions ────────────────────────────────────────
    naive_preds = np.array([naive_pinhole_distance(r) for r in test_ratios])

    # ── Compute metrics ──────────────────────────────────────────────────
    def compute_metrics(preds, truth):
        errors = np.abs(preds - truth)
        return {
            'mae': float(np.mean(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'max_error': float(np.max(errors)),
            'median_error': float(np.median(errors)),
            'mean_pct_error': float(np.mean(errors / (truth + 1e-8) * 100)),
        }

    anfis_metrics = compute_metrics(anfis_preds, test_distances)
    naive_metrics = compute_metrics(naive_preds, test_distances)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n--- Results ---")
    print(f"{'Metric':<25} {'ANFIS':>12} {'Naive Baseline':>15} {'Better':>10}")
    print("-" * 64)

    for metric in ['mae', 'rmse', 'max_error', 'median_error', 'mean_pct_error']:
        a = anfis_metrics[metric]
        n = naive_metrics[metric]
        better = 'ANFIS' if a < n else 'Naive'
        unit = '%' if 'pct' in metric else 'cm'
        print(f"{metric:<25} {a:>10.2f}{unit:>2} {n:>13.2f}{unit:>2} {better:>10}")

    improvement = ((naive_metrics['mae'] - anfis_metrics['mae']) / naive_metrics['mae']) * 100
    print(f"\nANFIS MAE improvement over baseline: {improvement:+.1f}%")

    # ── Per-sample comparison ─────────────────────────────────────────────
    print("\n--- Sample Predictions ---")
    print(f"{'AreaRatio':>12} {'TrueDist':>10} {'ANFIS':>10} {'Naive':>10} {'ANFIS_Err':>10} {'Naive_Err':>10}")
    print("-" * 64)
    for i in range(min(15, len(test_ratios))):
        r = test_ratios[i]
        t = test_distances[i]
        a = anfis_preds[i]
        n = naive_preds[i]
        ae = abs(a - t)
        ne = abs(n - t)
        print(f"{r:>12.4f} {t:>10.1f} {a:>10.1f} {n:>10.1f} {ae:>10.1f} {ne:>10.1f}")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_test_samples': len(test_ratios),
        'anfis_model': anfis_path,
        'anfis_available': anfis_available,
        'anfis_metrics': anfis_metrics,
        'naive_metrics': naive_metrics,
        'improvement_pct': improvement,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'distance_evaluation.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Report saved to: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ANFIS distance estimation")
    parser.add_argument("--anfis", type=str, default=None,
                        help="Path to ANFIS model (.pkl)")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test data CSV (box_area_ratio,true_distance_cm)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, "evaluation", "results"),
                        help="Output directory")
    args = parser.parse_args()

    evaluate_distance(args.anfis, args.test_data, args.output)


if __name__ == "__main__":
    main()
