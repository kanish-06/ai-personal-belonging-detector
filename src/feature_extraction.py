"""
feature_extraction.py — Feature Extraction Module

Derives high-level features from raw detections for use by the fuzzy guidance system:
  - box_area_ratio: relative size of detection box vs frame (distance proxy)
  - angle_offset:   horizontal position offset from center (-1 = far left, +1 = far right)
  - confidence:     raw detector confidence score
  - temporal_stability: fraction of recent frames with consistent detection (reduces flicker)
"""

from collections import defaultdict, deque
import numpy as np


class FeatureExtractor:
    """
    Extracts fuzzy-system-ready features from raw object detections.
    
    Maintains a rolling history per class for temporal stability computation.
    """

    def __init__(self, history_length=10, iou_threshold=0.3):
        """
        Args:
            history_length (int): Number of recent frames to track for stability.
            iou_threshold (float): Minimum IoU between current and historical
                                   detections to consider them "the same object".
        """
        self.history_length = history_length
        self.iou_threshold = iou_threshold

        # Per-class rolling history of bounding boxes (as xywh)
        # Each entry is either None (no detection) or (cx, cy, w, h)
        self._history = defaultdict(lambda: deque(maxlen=history_length))

    def extract(self, detection, frame_width, frame_height):
        """
        Compute features for a single detection.

        Args:
            detection: A Detection object with bbox_xywh and confidence attributes.
            frame_width (int): Width of the camera frame in pixels.
            frame_height (int): Height of the camera frame in pixels.

        Returns:
            dict: {
                'class_id': int,
                'class_name': str,
                'box_area_ratio': float,   # 0 to 1, larger = closer
                'angle_offset': float,     # -1 (far left) to +1 (far right)
                'confidence': float,       # 0 to 1
                'temporal_stability': float # 0 to 1
            }
        """
        cx, cy, w, h = detection.bbox_xywh
        frame_area = frame_width * frame_height

        # --- box_area_ratio ---
        # Ratio of bounding box area to total frame area (proxy for distance)
        box_area = w * h
        box_area_ratio = box_area / frame_area if frame_area > 0 else 0.0
        box_area_ratio = np.clip(box_area_ratio, 0.0, 1.0)

        # --- angle_offset ---
        # Normalized horizontal offset: -1 (far left) to +1 (far right)
        frame_center_x = frame_width / 2.0
        angle_offset = (cx - frame_center_x) / frame_center_x if frame_center_x > 0 else 0.0
        angle_offset = np.clip(angle_offset, -1.0, 1.0)

        # --- confidence ---
        confidence = float(detection.confidence)

        # --- temporal_stability ---
        stability = self._compute_stability(detection)

        return {
            'class_id': detection.class_id,
            'class_name': detection.class_name,
            'box_area_ratio': float(box_area_ratio),
            'angle_offset': float(angle_offset),
            'confidence': float(confidence),
            'temporal_stability': float(stability),
        }

    def update_history(self, detections):
        """
        Record this frame's detections into the rolling history.
        
        Call this once per frame AFTER extracting features for all detections.

        Args:
            detections: List of Detection objects for this frame.
        """
        # Track which classes were detected this frame
        detected_classes = set()

        for det in detections:
            cls_id = det.class_id
            detected_classes.add(cls_id)
            self._history[cls_id].append(det.bbox_xywh)

        # For classes that were NOT detected this frame, record a None
        for cls_id in list(self._history.keys()):
            if cls_id not in detected_classes:
                self._history[cls_id].append(None)

    def _compute_stability(self, detection):
        """
        Compute temporal stability: fraction of recent frames where this class
        was detected in roughly the same location (IoU > threshold).

        Args:
            detection: Current Detection object.

        Returns:
            float: Stability score between 0 and 1.
        """
        cls_id = detection.class_id
        history = self._history[cls_id]

        if len(history) == 0:
            return 0.0

        current_box = detection.bbox_xywh
        consistent_count = 0

        for past_box in history:
            if past_box is not None:
                iou = self._compute_iou_xywh(current_box, past_box)
                if iou >= self.iou_threshold:
                    consistent_count += 1

        stability = consistent_count / len(history)
        return stability

    @staticmethod
    def _compute_iou_xywh(box1, box2):
        """
        Compute IoU between two boxes in (cx, cy, w, h) format.

        Args:
            box1 (tuple): (cx, cy, w, h)
            box2 (tuple): (cx, cy, w, h)

        Returns:
            float: Intersection over Union (0-1).
        """
        # Convert to (x1, y1, x2, y2)
        cx1, cy1, w1, h1 = box1
        cx2, cy2, w2, h2 = box2

        x1_a, y1_a = cx1 - w1 / 2, cy1 - h1 / 2
        x2_a, y2_a = cx1 + w1 / 2, cy1 + h1 / 2

        x1_b, y1_b = cx2 - w2 / 2, cy2 - h2 / 2
        x2_b, y2_b = cx2 + w2 / 2, cy2 + h2 / 2

        # Intersection
        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Union
        area_a = w1 * h1
        area_b = w2 * h2
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def reset(self):
        """Clear all tracking history."""
        self._history.clear()


if __name__ == "__main__":
    # Quick demo with synthetic data
    from detect import Detection

    extractor = FeatureExtractor(history_length=5)

    # Simulate a phone detection in the center of a 640x480 frame
    det = Detection(
        class_id=2,
        class_name="phone",
        confidence=0.87,
        bbox_xyxy=(280, 200, 360, 280),
        bbox_xywh=(320, 240, 80, 80),
    )

    frame_w, frame_h = 640, 480

    # Simulate several frames with the same detection to build stability
    for i in range(5):
        features = extractor.extract(det, frame_w, frame_h)
        extractor.update_history([det])
        print(f"Frame {i+1}: {features}")

    print("\n--- Simulating detection disappearing ---")
    extractor.update_history([])  # no detections this frame
    features = extractor.extract(det, frame_w, frame_h)
    print(f"After gap: {features}")
