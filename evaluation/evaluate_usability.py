"""
evaluate_usability.py — Usability and Latency Evaluation

Measures real-world usability metrics:
- Time-to-locate-object across test trials
- False announcement count per trial
- End-to-end latency (FPS and detection-to-speech delay)

Usage:
    python evaluation/evaluate_usability.py --model models/detector/best.pt

This script runs in interactive mode: it captures from the camera and logs
events for later analysis.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


class UsabilityLogger:
    """
    Logs events during a usability test trial for later analysis.
    
    Events logged:
        - trial_start: when the user begins searching
        - detection: each time an object is detected (with class, confidence, latency)
        - announcement: each time a voice phrase is spoken
        - false_alarm: user marks a detection as incorrect (press 'f')
        - located: user confirms they found the object (press 'l')
        - trial_end: when the trial ends
    """

    def __init__(self):
        self.trials = []
        self.current_trial = None

    def start_trial(self, target_class):
        """Start a new trial."""
        self.current_trial = {
            'target_class': target_class,
            'start_time': time.time(),
            'end_time': None,
            'located_time': None,
            'time_to_locate': None,
            'events': [],
            'total_detections': 0,
            'total_announcements': 0,
            'false_alarms': 0,
            'correct_detections': 0,
            'frame_count': 0,
            'fps_samples': [],
        }
        print(f"\n=== Trial started: searching for '{target_class}' ===")

    def log_detection(self, class_name, confidence, latency_ms):
        """Log a detection event."""
        if self.current_trial is None:
            return
        self.current_trial['total_detections'] += 1
        self.current_trial['events'].append({
            'type': 'detection',
            'time': time.time() - self.current_trial['start_time'],
            'class_name': class_name,
            'confidence': confidence,
            'latency_ms': latency_ms,
        })

    def log_announcement(self, phrase):
        """Log a voice announcement."""
        if self.current_trial is None:
            return
        self.current_trial['total_announcements'] += 1
        self.current_trial['events'].append({
            'type': 'announcement',
            'time': time.time() - self.current_trial['start_time'],
            'phrase': phrase,
        })

    def log_false_alarm(self):
        """User marks the last detection as a false alarm."""
        if self.current_trial is None:
            return
        self.current_trial['false_alarms'] += 1
        self.current_trial['events'].append({
            'type': 'false_alarm',
            'time': time.time() - self.current_trial['start_time'],
        })
        print("  [!] False alarm logged")

    def log_located(self):
        """User confirms they located the object."""
        if self.current_trial is None:
            return
        t = time.time()
        self.current_trial['located_time'] = t
        self.current_trial['time_to_locate'] = t - self.current_trial['start_time']
        self.current_trial['events'].append({
            'type': 'located',
            'time': t - self.current_trial['start_time'],
        })
        print(f"  ✓ Object located! Time: {self.current_trial['time_to_locate']:.1f}s")

    def log_fps(self, fps):
        """Log an FPS measurement."""
        if self.current_trial is None:
            return
        self.current_trial['fps_samples'].append(fps)

    def end_trial(self):
        """End the current trial and save it."""
        if self.current_trial is None:
            return None

        self.current_trial['end_time'] = time.time()
        elapsed = self.current_trial['end_time'] - self.current_trial['start_time']

        # Compute summary stats
        self.current_trial['duration_seconds'] = elapsed
        if self.current_trial['fps_samples']:
            self.current_trial['avg_fps'] = sum(self.current_trial['fps_samples']) / len(self.current_trial['fps_samples'])
        else:
            self.current_trial['avg_fps'] = 0

        detection_latencies = [
            e['latency_ms'] for e in self.current_trial['events']
            if e['type'] == 'detection'
        ]
        if detection_latencies:
            self.current_trial['avg_detection_latency_ms'] = sum(detection_latencies) / len(detection_latencies)
            self.current_trial['max_detection_latency_ms'] = max(detection_latencies)
        else:
            self.current_trial['avg_detection_latency_ms'] = 0
            self.current_trial['max_detection_latency_ms'] = 0

        trial = self.current_trial
        self.trials.append(trial)
        self.current_trial = None

        print(f"\n=== Trial ended ===")
        print(f"  Duration:         {elapsed:.1f}s")
        print(f"  Time to locate:   {trial.get('time_to_locate', 'N/A')}")
        print(f"  Detections:       {trial['total_detections']}")
        print(f"  Announcements:    {trial['total_announcements']}")
        print(f"  False alarms:     {trial['false_alarms']}")
        print(f"  Avg FPS:          {trial['avg_fps']:.1f}")
        print(f"  Avg latency:      {trial['avg_detection_latency_ms']:.1f}ms")

        return trial

    def save_report(self, output_path):
        """Save all trial results to a JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_trials': len(self.trials),
            'trials': self.trials,
            'summary': self._compute_summary(),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n✓ Usability report saved to: {output_path}")

    def _compute_summary(self):
        """Compute aggregate statistics across all trials."""
        if not self.trials:
            return {}

        locate_times = [
            t['time_to_locate'] for t in self.trials
            if t.get('time_to_locate') is not None
        ]
        false_alarms = [t['false_alarms'] for t in self.trials]
        fps_vals = [t['avg_fps'] for t in self.trials if t['avg_fps'] > 0]
        latencies = [t['avg_detection_latency_ms'] for t in self.trials if t['avg_detection_latency_ms'] > 0]

        summary = {
            'total_trials': len(self.trials),
            'successful_locates': len(locate_times),
        }

        if locate_times:
            summary['avg_time_to_locate_s'] = sum(locate_times) / len(locate_times)
            summary['min_time_to_locate_s'] = min(locate_times)
            summary['max_time_to_locate_s'] = max(locate_times)

        if false_alarms:
            summary['avg_false_alarms_per_trial'] = sum(false_alarms) / len(false_alarms)
            summary['total_false_alarms'] = sum(false_alarms)

        if fps_vals:
            summary['avg_fps'] = sum(fps_vals) / len(fps_vals)

        if latencies:
            summary['avg_detection_latency_ms'] = sum(latencies) / len(latencies)

        return summary


def run_latency_benchmark(model_path, n_frames=100, imgsz=640):
    """
    Run a pure latency benchmark: measure detection speed without voice output.
    
    Args:
        model_path (str): Path to model weights.
        n_frames (int): Number of frames to benchmark.
        imgsz (int): Image size.
    
    Returns:
        dict: Latency metrics.
    """
    import cv2
    import numpy as np
    from detect import ObjectDetector

    print("\n" + "=" * 60)
    print("  LATENCY BENCHMARK")
    print("=" * 60)

    detector = ObjectDetector(model_path=model_path, confidence_threshold=0.25)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {frame_w}x{frame_h}")
    print(f"Benchmarking {n_frames} frames...\n")

    # Warm up
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            detector.detect(frame)

    latencies = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        t_start = time.perf_counter()
        dets = detector.detect(frame)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 20 == 0:
            print(f"  Frame {i+1}/{n_frames}: {latency_ms:.1f}ms, {len(dets)} detections")

    cap.release()

    latencies = np.array(latencies)
    results = {
        'n_frames': n_frames,
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'fps': float(1000.0 / np.mean(latencies)),
    }

    print(f"\n--- Latency Results ---")
    print(f"  Mean:   {results['mean_latency_ms']:.1f} ms")
    print(f"  Median: {results['median_latency_ms']:.1f} ms")
    print(f"  P95:    {results['p95_latency_ms']:.1f} ms")
    print(f"  P99:    {results['p99_latency_ms']:.1f} ms")
    print(f"  FPS:    {results['fps']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Usability and latency evaluation")
    parser.add_argument("--model", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "detector", "best.pt"))
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark only")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames for benchmark")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, "evaluation", "results", "usability_report.json"))

    args = parser.parse_args()

    if args.benchmark:
        results = run_latency_benchmark(args.model, args.frames)
        if results:
            output_dir = os.path.dirname(args.output)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'latency_benchmark.json'), 'w') as f:
                json.dump(results, f, indent=2)
    else:
        print("Usability evaluation requires the full pipeline running interactively.")
        print("Use the pipeline.py with the UsabilityLogger integrated, or run --benchmark for latency only.")


if __name__ == "__main__":
    main()
