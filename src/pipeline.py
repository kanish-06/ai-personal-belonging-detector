"""
pipeline.py — End-to-End Real-Time Pipeline

Ties together: Camera -> Detection -> Feature Extraction -> Fuzzy Guidance -> Voice Output
into a continuous real-time loop.

This is the main entry point for running the belonging detector system.
"""

import os
import sys
import time
import argparse
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[ERROR] OpenCV not installed. Install with: pip install opencv-python")

from detect import ObjectDetector, CLASS_NAMES
from feature_extraction import FeatureExtractor
from fuzzy_guidance import FuzzyGuidanceSystem, generate_direction_phrase
from voice_output import VoiceOutput


class BelongingDetectorPipeline:
    """
    Complete real-time pipeline for personal belonging detection and guidance.

    Flow per frame:
        1. Capture frame from camera
        2. Run YOLOv8 object detection
        3. Extract features (angle_offset, confidence, temporal_stability)
        4. Compute guidance via Fuzzy Inference System (urgency, frequency, direction)
        5. Speak guidance via Text-to-Speech
    """

    def __init__(self, 
                 model_path=None, 
                 camera_index=0,
                 confidence_threshold=0.35,
                 target_class=None,
                 show_gui=True,
                 history_length=10):
        """
        Args:
            model_path (str): Path to trained YOLOv8 weights (best.pt).
            camera_index (int): Camera device index for OpenCV.
            confidence_threshold (float): Minimum detection confidence.
            target_class (str): If set, only announce this class (e.g., "phone").
            show_gui (bool): Whether to display the camera feed with overlays.
            history_length (int): Frames of history for stability tracking.
        """
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.target_class = target_class
        self.show_gui = show_gui

        print("=" * 60)
        print("  BELONGING DETECTOR — Initializing Pipeline")
        print("=" * 60)

        # Initialize components
        print("\n[1/4] Loading object detector...")
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

        print("[2/4] Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor(history_length=history_length)

        print("[3/4] Initializing fuzzy guidance system...")
        self.fuzzy = FuzzyGuidanceSystem()

        print("[4/4] Initializing voice output...")
        self.voice = VoiceOutput(rate=160, volume=0.9)

        self._running = False
        self._frame_count = 0
        self._fps = 0.0

        print("\n" + "=" * 60)
        print("  Pipeline ready!")
        print("=" * 60)

    def run(self):
        """
        Start the real-time detection and guidance loop.
        Press 'q' to quit (when GUI is enabled).
        """
        if not HAS_CV2:
            print("[ERROR] OpenCV is required for the pipeline.")
            return

        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera at index {self.camera_index}")
            return

        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nCamera opened: {frame_width}x{frame_height}")

        if self.target_class:
            print(f"Searching for: {self.target_class}")
        else:
            print(f"Searching for: all objects ({', '.join(CLASS_NAMES.values())})")

        # Announce startup
        self.voice.announce_startup()

        self._running = True
        fps_timer = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to read frame from camera.")
                    continue

                self._frame_count += 1

                # ── Step 1: Object Detection ───────────────────────────────
                detections = self.detector.detect(frame, self.confidence_threshold)

                # Filter by target class if specified
                if self.target_class:
                    detections = [d for d in detections if d.class_name == self.target_class]

                # ── Step 2: Feature Extraction ─────────────────────────────
                features_list = []
                for det in detections:
                    features = self.feature_extractor.extract(det, frame_width, frame_height)
                    features_list.append(features)

                # Update tracking history
                self.feature_extractor.update_history(detections)

                # ── Step 3-4: Fuzzy Guidance + Voice for each detection ────
                display_frame = frame.copy() if self.show_gui else None

                if len(features_list) == 0:
                    # No detections — periodically say so
                    if self._frame_count % 90 == 0:  # ~every 3 seconds at 30fps
                        self.voice.announce_no_objects()
                else:
                    # Process the highest-confidence detection
                    best_features = max(features_list, key=lambda f: f['confidence'])

                    # Fuzzy guidance computation
                    guidance = self.fuzzy.compute(
                        angle_offset=best_features['angle_offset'],
                        confidence=best_features['confidence'],
                        stability=best_features['temporal_stability'],
                    )

                    # Voice announcement
                    if guidance['should_announce']:
                        self.voice.announce_with_angle(
                            class_name=best_features['class_name'],
                            angle_offset=best_features['angle_offset'],
                            urgency=guidance['urgency'],
                            frequency=guidance['frequency'],
                        )

                # ── GUI Display ────────────────────────────────────────────
                if self.show_gui and display_frame is not None:
                    display_frame = self._draw_overlays(
                        display_frame, detections, features_list, 
                        frame_width, frame_height,
                    )

                    # FPS counter
                    if time.time() - fps_timer >= 1.0:
                        self._fps = self._frame_count / (time.time() - fps_timer + 1e-8)
                        self._frame_count = 0
                        fps_timer = time.time()

                    cv2.putText(
                        display_frame, f"FPS: {self._fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )

                    cv2.imshow("Belonging Detector", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Pipeline] Quit signal received.")
                        break
                    elif key == ord('r'):
                        self.feature_extractor.reset()
                        print("[Pipeline] Tracking history reset.")

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted by user.")

        finally:
            self._running = False
            cap.release()
            if self.show_gui:
                cv2.destroyAllWindows()
            self.voice.shutdown()
            print("[Pipeline] Shutdown complete.")

    def _draw_overlays(self, frame, detections, features_list, frame_w, frame_h):
        """Draw bounding boxes, labels, and guidance info on the frame."""
        colors = {
            0: (0, 255, 255),    # keys — yellow
            1: (0, 165, 255),    # wallet — orange
            2: (255, 0, 0),      # phone — blue
            3: (0, 255, 0),      # watch — green
            4: (255, 0, 255),    # glasses — magenta
        }

        for det, feat in zip(detections, features_list):
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = colors.get(det.class_id, (255, 255, 255))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label with class, confidence, and angle
            label = f"{det.class_name} {det.confidence:.2f} | angle:{feat['angle_offset']:.2f}"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - lh - baseline - 5), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Stability indicator
            stab = feat['temporal_stability']
            stab_color = (0, int(255 * stab), int(255 * (1 - stab)))
            cv2.putText(frame, f"stab:{stab:.2f}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, stab_color, 1)

        # Center crosshair
        cx, cy = frame_w // 2, frame_h // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (100, 100, 100), 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (100, 100, 100), 1)

        # Info panel
        info_y = frame_h - 20
        if self.target_class:
            cv2.putText(frame, f"Searching: {self.target_class}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, f"Detections: {len(detections)}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def stop(self):
        """Signal the pipeline to stop."""
        self._running = False


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Belonging Detector — AI-powered personal item finder with voice guidance",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained YOLOv8 weights (default: models/detector/best.pt)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.35,
        help="Minimum detection confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        choices=["keys", "wallet", "phone", "watch", "glasses"],
        help="Only search for a specific object class",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run without GUI display (headless mode for Raspberry Pi)",
    )
    parser.add_argument(
        "--history", type=int, default=10,
        help="Number of frames for temporal stability tracking (default: 10)",
    )

    args = parser.parse_args()

    pipeline = BelongingDetectorPipeline(
        model_path=args.model,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        target_class=args.target,
        show_gui=not args.no_gui,
        history_length=args.history,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
