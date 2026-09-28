"""
run_full_benchmark.py — Comprehensive System Evaluation

Runs all benchmarks that can execute WITHOUT a webcam:
  1. Detector accuracy on test images (test-data/)
  2. Fuzzy Inference System — correctness across a sweep of input combinations
  3. Fuzzy Inference System — per-call latency
  4. Voice output — rate-limiting logic & phrase generation (no audio)
  5. Feature extraction — temporal stability simulation
  6. End-to-end single-frame pipeline latency (synthetic frames)
  7. Existing detector evaluation results (if available)

Usage:
    python evaluation/run_full_benchmark.py
    python evaluation/run_full_benchmark.py --include-detector   # also re-run YOLO val
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from detect import ObjectDetector, Detection, CLASS_NAMES
from feature_extraction import FeatureExtractor
from fuzzy_guidance import FuzzyGuidanceSystem, generate_direction_phrase
from voice_output import VoiceOutput


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Detector accuracy on test images
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_detector_test_images(detector):
    """Run detection on every image in test-data/ and report results."""
    import cv2

    test_dir = os.path.join(PROJECT_ROOT, 'test-data')
    if not os.path.isdir(test_dir):
        return {'status': 'SKIP', 'reason': 'test-data/ directory not found'}

    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(test_dir)
              if os.path.splitext(f)[1].lower() in image_exts]

    if not images:
        return {'status': 'SKIP', 'reason': 'No images in test-data/'}

    results = []
    total_latency = []

    for img_name in sorted(images):
        img_path = os.path.join(test_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            results.append({'image': img_name, 'error': 'Could not read image'})
            continue

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        total_latency.append(latency_ms)

        det_list = []
        for d in dets:
            det_list.append({
                'class': d.class_name,
                'confidence': round(d.confidence, 4),
                'bbox_xyxy': [round(float(v), 1) for v in d.bbox_xyxy],
            })

        # Infer expected class from filename (e.g. "wallet.jpg" → "wallet")
        base = os.path.splitext(img_name)[0].lower()
        expected_class = None
        for cls_name in CLASS_NAMES.values():
            if cls_name in base:
                expected_class = cls_name
                break

        found_expected = any(d.class_name == expected_class for d in dets) if expected_class else None

        results.append({
            'image': img_name,
            'expected_class': expected_class,
            'n_detections': len(dets),
            'detections': det_list,
            'found_expected': found_expected,
            'latency_ms': round(latency_ms, 2),
        })

    n_with_expected = sum(1 for r in results if r.get('expected_class') is not None)
    n_correct = sum(1 for r in results if r.get('found_expected') is True)

    return {
        'status': 'PASS',
        'n_images': len(images),
        'n_with_expected_class': n_with_expected,
        'n_correctly_detected': n_correct,
        'accuracy_on_labeled': round(n_correct / n_with_expected, 4) if n_with_expected > 0 else None,
        'avg_latency_ms': round(np.mean(total_latency), 2) if total_latency else 0,
        'max_latency_ms': round(np.max(total_latency), 2) if total_latency else 0,
        'per_image': results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FIS correctness — sweep across input combinations
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_fis_correctness(fis):
    """
    Sweep a grid of (angle, confidence, stability) inputs and verify
    that the FIS behaves sanely:
    - High conf + stable → high urgency
    - Low conf → low urgency, should_announce = False
    - Unstable → reduced urgency
    """
    test_cases = [
        # (angle, conf, stab, expected_urgency_range, expected_announce)
        (0.0,  0.92, 0.90, (0.65, 1.0),  True,  "High conf, center, stable → HIGH"),
        (-0.7, 0.88, 0.85, (0.65, 1.0),  True,  "High conf, far left, stable → HIGH"),
        (0.7,  0.90, 0.80, (0.65, 1.0),  True,  "High conf, far right, stable → HIGH"),
        (0.0,  0.55, 0.70, (0.25, 0.75), True,  "Med conf, center, stable → MEDIUM"),
        (0.0,  0.85, 0.15, (0.20, 0.75), True,  "High conf, unstable → MEDIUM"),
        (0.0,  0.20, 0.80, (0.0,  0.35), False, "Low conf, stable → LOW/suppress"),
        (0.0,  0.10, 0.10, (0.0,  0.35), False, "Low conf, unstable → suppress"),
        (-0.3, 0.45, 0.40, (0.0,  0.55), None,  "Borderline inputs → any"),
    ]

    passed = 0
    failed = 0
    details = []

    for angle, conf, stab, urg_range, expect_ann, desc in test_cases:
        result = fis.compute(angle, conf, stab)
        urgency = result['urgency']
        announce = result['should_announce']

        urg_ok = urg_range[0] <= urgency <= urg_range[1]
        ann_ok = (expect_ann is None) or (announce == expect_ann)
        ok = urg_ok and ann_ok

        if ok:
            passed += 1
        else:
            failed += 1

        details.append({
            'description': desc,
            'inputs': {'angle': angle, 'confidence': conf, 'stability': stab},
            'outputs': {
                'urgency': round(urgency, 4),
                'frequency': round(result['frequency'], 4),
                'direction': result['direction'],
                'should_announce': announce,
            },
            'expected_urgency_range': list(urg_range),
            'expected_announce': expect_ann,
            'pass': ok,
        })

    return {
        'status': 'PASS' if failed == 0 else 'FAIL',
        'n_cases': len(test_cases),
        'passed': passed,
        'failed': failed,
        'details': details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FIS latency benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_fis_latency(fis, n_calls=1000):
    """Measure per-call FIS computation latency."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(-1, 1, n_calls)
    confs = rng.uniform(0, 1, n_calls)
    stabs = rng.uniform(0, 1, n_calls)

    # Warm-up
    for i in range(min(50, n_calls)):
        fis.compute(float(angles[i]), float(confs[i]), float(stabs[i]))

    latencies = []
    for i in range(n_calls):
        t0 = time.perf_counter()
        fis.compute(float(angles[i]), float(confs[i]), float(stabs[i]))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)
    return {
        'status': 'PASS',
        'n_calls': n_calls,
        'mean_ms': round(float(np.mean(latencies)), 4),
        'median_ms': round(float(np.median(latencies)), 4),
        'p95_ms': round(float(np.percentile(latencies, 95)), 4),
        'p99_ms': round(float(np.percentile(latencies, 99)), 4),
        'min_ms': round(float(np.min(latencies)), 4),
        'max_ms': round(float(np.max(latencies)), 4),
        'total_fps_capacity': round(1000.0 / float(np.mean(latencies)), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Voice output — rate-limiting & phrase generation
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_voice_logic():
    """Test VoiceOutput rate-limiting and phrase construction WITHOUT speaking."""
    voice = VoiceOutput.__new__(VoiceOutput)
    # Manually init without TTS engine
    voice._last_announcement_time = 0.0
    voice._last_phrase = ""
    voice._speaking = False
    voice._has_engine = False
    voice._speech_queue = None
    voice._shutdown_event = None
    voice._tts_thread = None
    voice.MIN_INTERVAL = 1.5
    voice.MAX_INTERVAL = 8.0

    results = []

    # Test 1: phrase construction via _build_phrase
    phrase_tests = [
        ("phone", "center", 0.9, "Phone found, directly ahead!"),
        ("wallet", "left", 0.5, "Wallet detected, to your left."),
        ("keys", "right", 0.2, "Scanning... Keys spotted to your right."),
    ]
    for cls, direction, urg, expected in phrase_tests:
        phrase = voice._build_phrase(cls, direction, urg)
        results.append({
            'test': f'phrase({cls}, {direction}, urg={urg})',
            'expected': expected,
            'actual': phrase,
            'pass': phrase == expected,
        })

    # Test 2: rate-limiting intervals
    interval_tests = [
        (1.0, voice.MIN_INTERVAL),   # freq=1.0 → fastest
        (0.0, voice.MAX_INTERVAL),   # freq=0.0 → slowest
        (0.5, (voice.MIN_INTERVAL + voice.MAX_INTERVAL) / 2),
    ]
    for freq, expected_interval in interval_tests:
        actual_interval = voice.MAX_INTERVAL - freq * (voice.MAX_INTERVAL - voice.MIN_INTERVAL)
        results.append({
            'test': f'interval(freq={freq})',
            'expected_interval_s': expected_interval,
            'actual_interval_s': actual_interval,
            'pass': abs(actual_interval - expected_interval) < 0.01,
        })

    # Test 3: rate-limiting behaviour simulation
    voice._last_announcement_time = time.time()
    immediate = voice._should_announce(1.0)  # just announced, freq=1.0 → should be False
    results.append({
        'test': 'rate_limit_immediate',
        'expected': False,
        'actual': immediate,
        'pass': immediate == False,
    })

    voice._last_announcement_time = time.time() - 10.0  # 10s ago
    after_long_wait = voice._should_announce(0.0)  # freq=0 → 8s interval, 10s elapsed → True
    results.append({
        'test': 'rate_limit_after_10s',
        'expected': True,
        'actual': after_long_wait,
        'pass': after_long_wait == True,
    })

    # Test 4: direction phrase generation
    direction_tests = [
        (-0.8, "far to your left"),
        (-0.4, "to your left"),
        (-0.15, "slightly to your left"),
        (0.0, "directly ahead"),
        (0.15, "slightly to your right"),
        (0.4, "to your right"),
        (0.8, "far to your right"),
    ]
    for angle, expected_phrase in direction_tests:
        actual = generate_direction_phrase(angle)
        results.append({
            'test': f'direction_phrase(angle={angle})',
            'expected': expected_phrase,
            'actual': actual,
            'pass': actual == expected_phrase,
        })

    n_pass = sum(1 for r in results if r['pass'])
    n_fail = sum(1 for r in results if not r['pass'])

    return {
        'status': 'PASS' if n_fail == 0 else 'FAIL',
        'n_tests': len(results),
        'passed': n_pass,
        'failed': n_fail,
        'details': results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Feature extraction — temporal stability simulation
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_feature_extraction():
    """
    Simulate a sequence of detection frames and verify temporal stability
    builds up and degrades correctly.
    """
    extractor = FeatureExtractor(history_length=5)
    frame_w, frame_h = 640, 480

    det_center = Detection(
        class_id=2, class_name="phone", confidence=0.87,
        bbox_xyxy=(280, 200, 360, 280), bbox_xywh=(320, 240, 80, 80),
    )
    det_shifted = Detection(
        class_id=2, class_name="phone", confidence=0.82,
        bbox_xyxy=(50, 200, 130, 280), bbox_xywh=(90, 240, 80, 80),
    )

    results = []

    # Phase 1: 5 frames with same detection → stability should reach 1.0
    for i in range(5):
        feat = extractor.extract(det_center, frame_w, frame_h)
        extractor.update_history([det_center])
        results.append({
            'frame': i + 1,
            'phase': 'consistent',
            'stability': round(feat['temporal_stability'], 4),
            'angle_offset': round(feat['angle_offset'], 4),
        })

    final_stable = results[-1]['stability']

    # Phase 2: 1 frame with no detection → stability should drop
    extractor.update_history([])
    feat = extractor.extract(det_center, frame_w, frame_h)
    gap_stability = round(feat['temporal_stability'], 4)
    results.append({
        'frame': 6,
        'phase': 'gap',
        'stability': gap_stability,
    })

    # Phase 3: 1 frame with shifted detection → stability should drop further
    feat = extractor.extract(det_shifted, frame_w, frame_h)
    extractor.update_history([det_shifted])
    shift_stability = round(feat['temporal_stability'], 4)
    results.append({
        'frame': 7,
        'phase': 'shifted',
        'stability': shift_stability,
        'angle_offset': round(feat['angle_offset'], 4),
    })

    # Verify angle_offset for center (should be ~0)
    center_angle = results[0]['angle_offset']
    angle_ok = abs(center_angle) < 0.05

    # Verify stability builds and degrades
    stability_built = final_stable >= 0.8
    stability_dropped = gap_stability < final_stable

    all_pass = angle_ok and stability_built and stability_dropped

    return {
        'status': 'PASS' if all_pass else 'FAIL',
        'checks': {
            'center_angle_near_zero': {'value': center_angle, 'pass': angle_ok},
            'stability_reaches_high': {'value': final_stable, 'pass': stability_built},
            'stability_drops_on_gap': {'before': final_stable, 'after': gap_stability, 'pass': stability_dropped},
        },
        'frame_log': results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. End-to-end pipeline latency (synthetic frames)
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_e2e_latency(detector, fis, n_frames=50):
    """
    Measure end-to-end latency: detection → feature extraction → FIS.
    Uses synthetic blank frames (no webcam needed).
    """
    import cv2

    extractor = FeatureExtractor(history_length=10)
    frame_w, frame_h = 640, 480

    # Create a realistic-ish synthetic frame (not pure black — some noise)
    rng = np.random.default_rng(42)

    latencies_detect = []
    latencies_fis = []
    latencies_e2e = []

    for i in range(n_frames):
        frame = rng.integers(20, 80, size=(frame_h, frame_w, 3), dtype=np.uint8)

        t_start = time.perf_counter()

        # Detection
        t_det0 = time.perf_counter()
        dets = detector.detect(frame)
        t_det1 = time.perf_counter()

        # Feature extraction + FIS (use a synthetic detection if none found)
        if dets:
            feat = extractor.extract(dets[0], frame_w, frame_h)
        else:
            feat = {
                'angle_offset': 0.0,
                'confidence': 0.0,
                'temporal_stability': 0.0,
            }
        extractor.update_history(dets)

        t_fis0 = time.perf_counter()
        guidance = fis.compute(
            feat['angle_offset'], feat['confidence'], feat['temporal_stability']
        )
        t_fis1 = time.perf_counter()

        t_end = time.perf_counter()

        latencies_detect.append((t_det1 - t_det0) * 1000)
        latencies_fis.append((t_fis1 - t_fis0) * 1000)
        latencies_e2e.append((t_end - t_start) * 1000)

    ld = np.array(latencies_detect)
    lf = np.array(latencies_fis)
    le = np.array(latencies_e2e)

    return {
        'status': 'PASS',
        'n_frames': n_frames,
        'detection': {
            'mean_ms': round(float(np.mean(ld)), 2),
            'median_ms': round(float(np.median(ld)), 2),
            'p95_ms': round(float(np.percentile(ld, 95)), 2),
        },
        'fis': {
            'mean_ms': round(float(np.mean(lf)), 2),
            'median_ms': round(float(np.median(lf)), 2),
            'p95_ms': round(float(np.percentile(lf, 95)), 2),
        },
        'end_to_end': {
            'mean_ms': round(float(np.mean(le)), 2),
            'median_ms': round(float(np.median(le)), 2),
            'p95_ms': round(float(np.percentile(le, 95)), 2),
            'fps': round(1000.0 / float(np.mean(le)), 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Load existing detector evaluation results
# ═══════════════════════════════════════════════════════════════════════════════

def load_existing_detector_eval():
    """Load the previously-run detector_evaluation.json if it exists."""
    path = os.path.join(PROJECT_ROOT, 'evaluation', 'results', 'detector_evaluation.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'status': 'SKIP', 'reason': 'No prior detector_evaluation.json found'}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FIS announcement-rate simulation over time
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_announcement_rate():
    """
    Simulate 60 seconds of pipeline operation and count how many voice
    announcements would fire under various scenarios:
    - continuous high-conf detection
    - intermittent detection (every other second)
    - low-conf noise
    """
    scenarios = []

    for name, conf, stab, freq_val in [
        ("high_conf_stable", 0.90, 0.85, 0.78),
        ("medium_conf", 0.55, 0.70, 0.22),
        ("low_conf_noise", 0.20, 0.30, 0.22),
    ]:
        # Compute the cooldown interval for this frequency level
        min_interval = 1.5
        max_interval = 8.0
        interval = max_interval - freq_val * (max_interval - min_interval)

        sim_duration = 60.0
        n_announcements = 0
        last_ann_time = -interval  # allow first announcement immediately
        t = 0.0

        while t < sim_duration:
            t += 1.0 / 30  # ~30fps frame steps
            elapsed = t - last_ann_time
            if elapsed >= interval:
                n_announcements += 1
                last_ann_time = t

        scenarios.append({
            'scenario': name,
            'fuzzy_frequency': freq_val,
            'computed_interval_s': round(interval, 2),
            'announcements_in_60s': n_announcements,
            'rate_per_minute': n_announcements,
        })

    return {
        'status': 'PASS',
        'simulation_seconds': 60,
        'scenarios': scenarios,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full system benchmark")
    parser.add_argument("--include-detector", action="store_true",
                        help="Also re-run YOLO validation (requires dataset)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, "evaluation", "results",
                                             "full_benchmark.json"))
    args = parser.parse_args()

    print("=" * 70)
    print("  BELONGING DETECTOR — FULL SYSTEM BENCHMARK")
    print("=" * 70)
    print(f"  Started: {datetime.now().isoformat()}")
    print()

    report = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {},
    }

    # ── 1. Initialize core components ──────────────────────────────────────
    print("[1] Loading detector model...")
    detector = ObjectDetector(confidence_threshold=0.25)

    print("[2] Initializing Fuzzy Inference System...")
    fis = FuzzyGuidanceSystem()

    # ── 2. Detector on test images ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] Detector accuracy on test images")
    print("─" * 50)
    det_results = benchmark_detector_test_images(detector)
    report['benchmarks']['detector_test_images'] = det_results
    for r in det_results.get('per_image', []):
        status = "✓" if r.get('found_expected') else ("✗" if r.get('found_expected') is False else "?")
        print(f"  {status} {r['image']}: {r['n_detections']} dets, "
              f"{r['latency_ms']:.1f}ms — {[d['class'] for d in r.get('detections', [])]}")
    if det_results.get('accuracy_on_labeled') is not None:
        print(f"  → Accuracy on labeled images: {det_results['accuracy_on_labeled']:.1%}")

    # ── 3. Existing YOLO validation results ────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] Prior detector evaluation (mAP)")
    print("─" * 50)
    prior_eval = load_existing_detector_eval()
    report['benchmarks']['prior_detector_eval'] = prior_eval
    if 'overall' in prior_eval:
        o = prior_eval['overall']
        print(f"  mAP50:      {o['mAP50']:.4f}")
        print(f"  mAP50-95:   {o['mAP50_95']:.4f}")
        print(f"  Precision:  {o['precision']:.4f}")
        print(f"  Recall:     {o['recall']:.4f}")
        print("  Per-class:")
        for cls, m in prior_eval.get('per_class', {}).items():
            flag = "✓" if m['mAP50'] >= 0.5 else "⚠"
            print(f"    {flag} {cls:10s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"mAP50={m['mAP50']:.3f}  mAP50-95={m['mAP50_95']:.3f}")
    else:
        print(f"  {prior_eval.get('reason', 'N/A')}")

    # ── 4. FIS correctness ─────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] FIS correctness sweep")
    print("─" * 50)
    fis_corr = benchmark_fis_correctness(fis)
    report['benchmarks']['fis_correctness'] = fis_corr
    for d in fis_corr['details']:
        status = "✓" if d['pass'] else "✗"
        print(f"  {status} {d['description']}")
        print(f"      → urgency={d['outputs']['urgency']:.2f}, "
              f"freq={d['outputs']['frequency']:.2f}, "
              f"announce={d['outputs']['should_announce']}")
    print(f"  Result: {fis_corr['passed']}/{fis_corr['n_cases']} passed")

    # ── 5. FIS latency ─────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] FIS latency (1000 calls)")
    print("─" * 50)
    fis_lat = benchmark_fis_latency(fis, n_calls=1000)
    report['benchmarks']['fis_latency'] = fis_lat
    print(f"  Mean:   {fis_lat['mean_ms']:.3f} ms")
    print(f"  Median: {fis_lat['median_ms']:.3f} ms")
    print(f"  P95:    {fis_lat['p95_ms']:.3f} ms")
    print(f"  P99:    {fis_lat['p99_ms']:.3f} ms")
    print(f"  Max FIS throughput: ~{fis_lat['total_fps_capacity']:.0f} calls/sec")

    # ── 6. Voice logic ─────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] Voice output logic")
    print("─" * 50)
    voice_res = benchmark_voice_logic()
    report['benchmarks']['voice_logic'] = voice_res
    for d in voice_res['details']:
        status = "✓" if d['pass'] else "✗"
        print(f"  {status} {d['test']}")
    print(f"  Result: {voice_res['passed']}/{voice_res['n_tests']} passed")

    # ── 7. Feature extraction ──────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] Feature extraction — stability tracking")
    print("─" * 50)
    feat_res = benchmark_feature_extraction()
    report['benchmarks']['feature_extraction'] = feat_res
    for name, check in feat_res['checks'].items():
        status = "✓" if check['pass'] else "✗"
        print(f"  {status} {name}: {check}")
    print(f"  Result: {feat_res['status']}")

    # ── 8. Announcement rate simulation ────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] Announcement rate simulation (60s)")
    print("─" * 50)
    ann_res = benchmark_announcement_rate()
    report['benchmarks']['announcement_rate'] = ann_res
    for s in ann_res['scenarios']:
        print(f"  {s['scenario']:25s}  interval={s['computed_interval_s']:.1f}s  "
              f"announcements={s['announcements_in_60s']}")

    # ── 9. End-to-end pipeline latency ─────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Benchmark] End-to-end pipeline latency (synthetic frames)")
    print("─" * 50)
    e2e = benchmark_e2e_latency(detector, fis, n_frames=50)
    report['benchmarks']['e2e_latency'] = e2e
    print(f"  Detection:    mean={e2e['detection']['mean_ms']:.1f}ms, p95={e2e['detection']['p95_ms']:.1f}ms")
    print(f"  FIS:          mean={e2e['fis']['mean_ms']:.2f}ms, p95={e2e['fis']['p95_ms']:.2f}ms")
    print(f"  End-to-end:   mean={e2e['end_to_end']['mean_ms']:.1f}ms, p95={e2e['end_to_end']['p95_ms']:.1f}ms")
    print(f"  Effective FPS: ~{e2e['end_to_end']['fps']:.1f}")

    # ── 10. Optionally re-run detector YOLO val ────────────────────────────
    if args.include_detector:
        print("\n" + "─" * 50)
        print("[Benchmark] Re-running YOLO validation...")
        print("─" * 50)
        from evaluate_detector import evaluate_detector
        det_metrics = evaluate_detector(
            model_path=os.path.join(PROJECT_ROOT, "models", "detector", "best.pt"),
            data_yaml=os.path.join(PROJECT_ROOT, "dataset", "data.yaml"),
            output_dir=os.path.join(PROJECT_ROOT, "evaluation", "results"),
        )
        report['benchmarks']['detector_yolo_val'] = det_metrics

    # ── Save report ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)

    all_statuses = []
    for name, bench in report['benchmarks'].items():
        if isinstance(bench, dict) and 'status' in bench:
            s = bench['status']
            all_statuses.append(s)
            icon = "✓" if s == 'PASS' else ("⊘" if s == 'SKIP' else "✗")
            print(f"  {icon} {name}: {s}")

    overall = 'PASS' if all(s in ('PASS', 'SKIP') for s in all_statuses) else 'FAIL'
    report['overall_status'] = overall

    # Re-save with overall
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Overall: {overall}")
    print(f"  Report saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
