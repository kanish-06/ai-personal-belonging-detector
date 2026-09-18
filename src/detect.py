"""
detect.py — Object Detection Module

Loads a trained YOLOv8 model and runs inference on camera frames.
Returns structured detection results (bounding boxes, confidence scores, class labels).
"""

import os
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[WARNING] ultralytics not installed. Detection will not work.")
    print("         Install with: pip install ultralytics")


# Class ID -> name mapping (matches data.yaml)
CLASS_NAMES = {0: "keys", 1: "wallet", 2: "phone", 3: "watch", 4: "glasses"}
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

# Default model path
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "detector", "best.pt"
)


class Detection:
    """Represents a single object detection result."""

    def __init__(self, class_id, class_name, confidence, bbox_xyxy, bbox_xywh):
        """
        Args:
            class_id (int): Class index (0-4).
            class_name (str): Human-readable class name.
            confidence (float): Detection confidence score (0-1).
            bbox_xyxy (tuple): Bounding box as (x1, y1, x2, y2) in pixels.
            bbox_xywh (tuple): Bounding box as (center_x, center_y, width, height) in pixels.
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_xyxy = bbox_xyxy  # (x1, y1, x2, y2)
        self.bbox_xywh = bbox_xywh  # (cx, cy, w, h)

    def __repr__(self):
        return (
            f"Detection(class={self.class_name}, conf={self.confidence:.2f}, "
            f"bbox_xyxy={tuple(round(v, 1) for v in self.bbox_xyxy)})"
        )


class ObjectDetector:
    """Wrapper around a YOLOv8 model for personal belonging detection."""

    def __init__(self, model_path=None, confidence_threshold=0.25, device=None):
        """
        Args:
            model_path (str): Path to the trained YOLOv8 .pt weights file.
                              Defaults to models/detector/best.pt.
            confidence_threshold (float): Minimum confidence to keep a detection.
            device (str): Device to run inference on ('cpu', 'cuda', 'mps', etc.).
                          None = auto-detect.
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics package is required. Install with: pip install ultralytics"
            )

        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n"
                f"Train the model first (see training/train_yolov8.py) or place "
                f"a trained best.pt in models/detector/."
            )

        self.model = YOLO(self.model_path)
        if device:
            self.model.to(device)

        print(f"[ObjectDetector] Loaded model from: {self.model_path}")

    def detect(self, frame, confidence_threshold=None):
        """
        Run detection on a single frame.

        Args:
            frame (np.ndarray): BGR image from OpenCV (H, W, 3).
            confidence_threshold (float): Override instance threshold for this call.

        Returns:
            list[Detection]: List of Detection objects found in the frame.
        """
        conf_thresh = confidence_threshold or self.confidence_threshold

        # Run YOLOv8 inference
        results = self.model(frame, conf=conf_thresh, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Extract bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Compute center-format box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                # Extract class and confidence
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

                det = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywh=(cx, cy, w, h),
                )
                detections.append(det)

        return detections

    def detect_and_draw(self, frame, confidence_threshold=None):
        """
        Run detection and draw bounding boxes + labels on the frame.

        Args:
            frame (np.ndarray): BGR image from OpenCV.
            confidence_threshold (float): Override threshold.

        Returns:
            tuple: (annotated_frame, list[Detection])
        """
        import cv2

        detections = self.detect(frame, confidence_threshold)
        annotated = frame.copy()

        # Colors for each class (BGR)
        colors = {
            0: (0, 255, 255),    # keys — yellow
            1: (0, 165, 255),    # wallet — orange
            2: (255, 0, 0),      # phone — blue
            3: (0, 255, 0),      # watch — green
            4: (255, 0, 255),    # glasses — magenta
        }

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = colors.get(det.class_id, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color, -1,
            )
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        return annotated, detections


if __name__ == "__main__":
    # Quick test: run detection on a webcam feed
    import cv2

    try:
        detector = ObjectDetector()
    except FileNotFoundError as e:
        print(e)
        print("\nTo test, first train the model or place best.pt in models/detector/")
        exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        exit(1)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, dets = detector.detect_and_draw(frame)
        for d in dets:
            print(d)

        cv2.imshow("Belonging Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
