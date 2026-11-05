"""
hybrid_detector.py — Hybrid YOLOv8 + Qwen3-VL Object Detector

Combines YOLOv8 real-time object detection with Qwen3-VL visual verification.
- YOLO runs on every frame as primary real-time perception.
- Confident detections (conf >= reliable_threshold, default 0.35) bypass Qwen.
- Uncertain detections (uncertain_threshold <= conf < reliable_threshold) are verified
  using Qwen3-VL on the cropped bounding box.
- Spatial-temporal caching and asynchronous worker prevent camera loop blocking
  and repeated queries for the same object across consecutive frames.
- Preserves original YOLO bounding boxes and confidence scores.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from detect import Detection, ObjectDetector, CLASS_NAMES, CLASS_IDS
from qwen_verifier import QwenVerifier, SUPPORTED_CLASSES


def compute_iou(boxA: Tuple[float, float, float, float],
                boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def crop_bounding_box(frame: np.ndarray,
                      bbox_xyxy: Tuple[float, float, float, float],
                      padding_ratio: float = 0.10) -> Optional[np.ndarray]:
    """
    Crops the bounding box from the frame with a proportional margin for context.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    cx1 = max(0, int(x1) - pad_x)
    cy1 = max(0, int(y1) - pad_y)
    cx2 = min(w, int(x2) + pad_x)
    cy2 = min(h, int(y2) + pad_y)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    return crop.copy()


class HybridDetector:
    """
    Hybrid YOLOv8 + Qwen3-VL detector.
    """

    def __init__(
        self,
        yolo_detector: ObjectDetector,
        verifier: Optional[QwenVerifier] = None,
        reliable_threshold: float = 0.35,
        uncertain_threshold: float = 0.20,
        cooldown_seconds: float = 3.0,
        async_mode: bool = True,
        enabled: bool = True,
    ):
        """
        Args:
            yolo_detector (ObjectDetector): Primary YOLO detector instance.
            verifier (QwenVerifier, optional): Secondary Qwen3-VL verifier.
            reliable_threshold (float): Confidence at or above which YOLO is trusted directly.
            uncertain_threshold (float): Minimum confidence to consider an object for verification.
            cooldown_seconds (float): Time window to cache verification results per spatial region.
            async_mode (bool): If True, runs Qwen verification asynchronously to avoid blocking live feeds.
            enabled (bool): Toggle hybrid verification on or off.
        """
        self.yolo_detector = yolo_detector
        self.verifier = verifier or QwenVerifier()
        self.reliable_threshold = reliable_threshold
        self.uncertain_threshold = uncertain_threshold
        self.cooldown_seconds = cooldown_seconds
        self.async_mode = async_mode
        self.enabled = enabled

        # Cache of recent verification results:
        # Each item: {'bbox': (x1, y1, x2, y2), 'verified_class': str, 'timestamp': float}
        self._cache: List[Dict[str, Any]] = []
        self._cache_lock = threading.Lock()

        # In-flight tracking to avoid submitting duplicate jobs
        self._in_flight_boxes: List[Tuple[float, float, float, float]] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="QwenVerifier")

    def _match_cache(self, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """Finds any recent verification for this spatial region."""
        now = time.time()
        with self._cache_lock:
            # Prune expired entries
            self._cache = [e for e in self._cache if (now - e['timestamp']) < self.cooldown_seconds]

            for entry in self._cache:
                iou = compute_iou(bbox, entry['bbox'])
                if iou >= 0.30:
                    return entry['verified_class']

        return None

    def _is_in_flight(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Checks if a verification job is currently running for an overlapping box."""
        with self._cache_lock:
            for active_box in self._in_flight_boxes:
                if compute_iou(bbox, active_box) >= 0.30:
                    return True
        return False

    def _async_verify_worker(self, crop: np.ndarray, bbox: Tuple[float, float, float, float]):
        """Runs in background thread to verify an uncertain patch."""
        try:
            verified_class = self.verifier.verify_crop(crop)
            now = time.time()
            with self._cache_lock:
                self._cache.append({
                    'bbox': bbox,
                    'verified_class': verified_class,
                    'timestamp': now
                })
        finally:
            with self._cache_lock:
                self._in_flight_boxes = [
                    b for b in self._in_flight_boxes if compute_iou(b, bbox) < 0.30
                ]

    def detect(self, frame: np.ndarray, sync: bool = False) -> List[Detection]:
        """
        Runs the hybrid detection workflow on a single frame.

        Args:
            frame (np.ndarray): Input image/frame (BGR format).
            sync (bool): If True, forces synchronous verification (useful for single test images).

        Returns:
            List[Detection]: Reconciled and verified detections.
        """
        if frame is None or frame.size == 0:
            return []

        # 1. Run YOLO detector with uncertain_threshold to capture candidates
        raw_detections = self.yolo_detector.detect(
            frame, confidence_threshold=self.uncertain_threshold
        )

        if not raw_detections:
            return []

        if not self.enabled:
            # If hybrid is disabled, just return detections >= reliable_threshold
            for d in raw_detections:
                d.is_verified = False
                d.verified_by = "yolo"
            return [d for d in raw_detections if d.confidence >= self.reliable_threshold]

        final_detections: List[Detection] = []

        for det in raw_detections:
            # Preserve original confidence separately
            det.original_confidence = det.confidence
            det.verified_by = "yolo"
            det.is_verified = False

            # 2. Reliable detection: Confidence >= reliable_threshold
            if det.confidence >= self.reliable_threshold:
                det.verified_by = "yolo"
                det.is_verified = False
                final_detections.append(det)
                continue

            # 3. Uncertain detection: confidence < reliable_threshold
            # Check spatial cache first
            cached_class = self._match_cache(det.bbox_xyxy)
            if cached_class is not None:
                if cached_class in SUPPORTED_CLASSES:
                    # Reconciliation with cached result
                    det.class_name = cached_class
                    det.class_id = CLASS_IDS.get(cached_class, det.class_id)
                    det.verified_by = "qwen_cache"
                    det.is_verified = True
                    final_detections.append(det)
                # If cached as 'unknown' or error, suppress detection
                continue

            # Crop the bounding box for verification
            crop = crop_bounding_box(frame, det.bbox_xyxy, padding_ratio=0.10)
            if crop is None:
                continue

            # If sync mode requested (e.g. static image test) or async_mode disabled:
            if sync or not self.async_mode:
                verified_class = self.verifier.verify_crop(crop)
                # Store in cache
                with self._cache_lock:
                    self._cache.append({
                        'bbox': det.bbox_xyxy,
                        'verified_class': verified_class,
                        'timestamp': time.time()
                    })

                # 4. Reconciliation
                if verified_class == det.class_name:
                    det.verified_by = "qwen"
                    det.is_verified = True
                    final_detections.append(det)
                elif verified_class in SUPPORTED_CLASSES:
                    det.class_name = verified_class
                    det.class_id = CLASS_IDS.get(verified_class, det.class_id)
                    det.verified_by = "qwen_reconciled"
                    det.is_verified = True
                    final_detections.append(det)
                else:
                    # 'unknown' or unverified -> suppress announcement / detection
                    pass

            else:
                # Asynchronous mode for live video streams
                if not self._is_in_flight(det.bbox_xyxy):
                    with self._cache_lock:
                        self._in_flight_boxes.append(det.bbox_xyxy)
                    self._executor.submit(self._async_verify_worker, crop, det.bbox_xyxy)

                # While verification is pending, suppress the uncertain announcement
                pass

        return final_detections

    def shutdown(self):
        """Cleans up background executor."""
        self._executor.shutdown(wait=False)
