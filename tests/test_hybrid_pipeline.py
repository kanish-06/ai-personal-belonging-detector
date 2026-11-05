"""
test_hybrid_pipeline.py — Functional & Regression Verification for YOLOv8 + Qwen3-VL Hybrid Pipeline

Tests:
1. High-confidence YOLO detection (>= 0.35) bypasses Qwen.
2. Low-confidence YOLO detection (< 0.35) invokes Qwen.
3. Qwen verification preserves original YOLO bounding box and confidence.
4. Unknown / failed verification suppresses the announcement.
5. Resulting detection passes through depth estimation, feature extraction, fuzzy guidance, and voice output.
6. Audible voice execution path check (TTS child process exits cleanly with code 0).
7. Regression test for low-confidence wallet (conf = 0.272):
   - Verifies Qwen-confirmed detection passes the production announcement gate (should_announce == True)
     and reaches the voice-output queue (phrase is not None).
   - Verifies an unknown Qwen response remains suppressed.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detect import ObjectDetector, Detection
from qwen_verifier import QwenVerifier
from hybrid_detector import HybridDetector
from depth_estimation import DepthEstimator
from feature_extraction import FeatureExtractor
from fuzzy_guidance import FuzzyGuidanceSystem
from voice_output import VoiceOutput


class CallTrackingVerifier(QwenVerifier):
    """Subclass to count and inspect Qwen verification calls during tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0
        self.last_crop = None
        self.last_result = None

    def verify_crop(self, crop_bgr):
        self.call_count += 1
        self.last_crop = crop_bgr
        res = super().verify_crop(crop_bgr)
        self.last_result = res
        return res


def run_hybrid_verification_tests():
    print("=" * 70)
    print("  RUNNING HYBRID PIPELINE FUNCTIONAL & REGRESSION VERIFICATION")
    print("=" * 70)

    # Load components
    model_path = str(PROJECT_ROOT / "models" / "detector" / "best.pt")
    yolo = ObjectDetector(model_path=model_path, confidence_threshold=0.20)
    verifier = CallTrackingVerifier(model_name="qwen3-vl:2b")

    hybrid = HybridDetector(
        yolo_detector=yolo,
        verifier=verifier,
        reliable_threshold=0.35,
        uncertain_threshold=0.20,
        async_mode=False  # synchronous for deterministic test inspection
    )

    test_data_dir = PROJECT_ROOT / "test-data"

    # ──────────────────────────────────────────────────────────────────────────
    # Test 1: High-confidence YOLO detection bypasses Qwen
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Test 1] Verifying high-confidence detection bypasses Qwen...")
    glasses_img = cv2.imread(str(test_data_dir / "glasses.jpeg"))
    assert glasses_img is not None, "glasses.jpeg missing"

    verifier.call_count = 0
    dets_high = hybrid.detect(glasses_img, sync=True)
    print(f"  Detections: {dets_high}")
    print(f"  Qwen calls made: {verifier.call_count}")

    assert len(dets_high) > 0, "Expected at least 1 detection on glasses.jpeg"
    top_det = dets_high[0]
    assert top_det.confidence >= 0.35, f"Expected conf >= 0.35, got {top_det.confidence}"
    assert top_det.verified_by == "yolo", f"Expected verified_by 'yolo', got {top_det.verified_by}"
    assert verifier.call_count == 0, f"Qwen should not have been called, but was called {verifier.call_count} times"
    print("  [PASS] High-confidence YOLO detection bypassed Qwen.")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 2: Low-confidence YOLO detection invokes Qwen
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Test 2] Verifying low-confidence detection invokes Qwen...")
    # wallet2.jpeg has YOLO confidence ~0.272 (< 0.35)
    wallet2_img = cv2.imread(str(test_data_dir / "wallet2.jpeg"))
    assert wallet2_img is not None, "wallet2.jpeg missing"

    # First check raw YOLO confidence on wallet2
    raw_dets = yolo.detect(wallet2_img, confidence_threshold=0.20)
    print(f"  Raw YOLO detections on wallet2: {raw_dets}")
    assert any(0.20 <= d.confidence < 0.35 for d in raw_dets), "Expected an uncertain detection (0.20 <= conf < 0.35)"

    verifier.call_count = 0
    dets_low = hybrid.detect(wallet2_img, sync=True)
    print(f"  Hybrid detections on wallet2: {dets_low}")
    print(f"  Qwen calls made: {verifier.call_count}")

    assert verifier.call_count > 0, "Qwen should have been invoked for low-confidence detection"
    assert len(dets_low) > 0, "Expected detection to be retained after Qwen verification"
    low_det = dets_low[0]
    assert "qwen" in low_det.verified_by, f"Expected verified_by to contain 'qwen', got {low_det.verified_by}"
    assert low_det.is_verified is True, "Expected is_verified to be True"
    print(f"  [PASS] Low-confidence detection invoked Qwen and verified as '{low_det.class_name}'.")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 3: Bounding box and original confidence preservation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Test 3] Verifying bounding box and original confidence preservation...")
    raw_wallet_det = next(d for d in raw_dets if 0.20 <= d.confidence < 0.35)
    print(f"  Original YOLO Box: {raw_wallet_det.bbox_xyxy}")
    print(f"  Hybrid Result Box: {low_det.bbox_xyxy}")
    print(f"  Original YOLO Conf: {raw_wallet_det.confidence:.4f}")
    print(f"  Hybrid Result Conf: {low_det.confidence:.4f}")

    assert low_det.bbox_xyxy == raw_wallet_det.bbox_xyxy, "Bounding box must be preserved exactly"
    assert abs(low_det.confidence - raw_wallet_det.confidence) < 1e-5, "Original YOLO confidence must be preserved"
    print("  [PASS] Exact bounding box coordinates and original confidence score preserved.")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 4: Unknown / failed verification suppresses announcement
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Test 4] Verifying unknown/failed verification suppresses detection...")
    class FailingVerifier(QwenVerifier):
        def verify_crop(self, crop_bgr):
            return "unknown"  # Simulates inability to confirm or failure

    failing_hybrid = HybridDetector(
        yolo_detector=yolo,
        verifier=FailingVerifier(model_name="qwen3-vl:2b"),
        reliable_threshold=0.35,
        uncertain_threshold=0.20,
        async_mode=False
    )

    dets_suppressed = failing_hybrid.detect(wallet2_img, sync=True)
    print(f"  Detections with failing/unknown verifier: {dets_suppressed}")
    assert len(dets_suppressed) == 0, "Uncertain detection must be suppressed when Qwen returns 'unknown'"
    print("  [PASS] Uncertain detection successfully suppressed on unknown verification.")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 5: Full pipeline flow (Depth -> Features -> Fuzzy -> Voice)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Test 5] Verifying downstream pipeline execution (Depth, Features, Fuzzy)...")
    depth_estimator = DepthEstimator()
    feature_extractor = FeatureExtractor()
    fuzzy = FuzzyGuidanceSystem()

    frame_h, frame_w = wallet2_img.shape[:2]
    depth_map = depth_estimator.estimate_depth(wallet2_img)
    distance = depth_estimator.extract_depth_for_box(depth_map, low_det.bbox_xyxy)
    print(f"  Estimated depth distance: {distance:.2f} m")

    features = feature_extractor.extract(low_det, frame_w, frame_h)
    features['distance'] = distance
    print(f"  Extracted features: angle_offset={features['angle_offset']:.2f}, "
          f"confidence={features['confidence']:.4f}, stability={features['temporal_stability']:.2f}, "
          f"is_verified={features['is_verified']}")

    guidance = fuzzy.compute(
        angle_offset=features['angle_offset'],
        confidence=features['confidence'],
        stability=features['temporal_stability'],
        is_verified=features['is_verified']
    )
    print(f"  Fuzzy guidance result: {guidance}")
    assert "should_announce" in guidance, "Missing should_announce in guidance"
    assert "urgency" in guidance and "direction" in guidance, "Missing fuzzy outputs"
    print("  [PASS] Full perception-guidance pipeline execution verified.")

    # ──────────────────────────────────────────────────────────────────────────
    # Test 6: Audible Voice Output Execution Verification
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Test 6] Verifying audible voice output execution path...")
    voice = VoiceOutput(rate=160, volume=0.9)
    print(f"  VoiceOutput has_engine: {voice.has_engine}")

    phrase = voice.announce_with_angle(
        class_name=low_det.class_name,
        angle_offset=features['angle_offset'],
        urgency=guidance['urgency'],
        frequency=guidance['frequency'],
        distance=features['distance']
    )
    print(f"  Spoken phrase generated: '{phrase}'")

    if voice.has_engine:
        idle_ok = voice.wait_until_idle(timeout=10.0)
        print(f"  Child TTS process finished: idle={idle_ok}, exit_code={voice.last_exit_code}")
        assert voice.last_exit_code == 0, f"TTS child process failed with exit code {voice.last_exit_code}"
        print("  [PASS] Voice output process executed cleanly with exit code 0.")
    else:
        print("  [INFO] System running in console-only mode (COM/SAPI audio unavailable in sandbox).")

    voice.shutdown()

    # ──────────────────────────────────────────────────────────────────────────
    # Test 7: Regression Test for Low-Confidence Wallet (conf = 0.272)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Test 7] Running regression test for low-confidence wallet announcement gate...")
    # Exact scenario:
    # Original YOLO confidence = 0.272
    # Qwen verification = wallet (is_verified = True)
    # FIS urgency = 0.1667 (> 0.15)
    
    # Check 7A: Qwen-confirmed low-confidence wallet MUST pass announcement gate
    wallet_features = feature_extractor.extract(low_det, frame_w, frame_h)
    wallet_features['distance'] = distance
    assert wallet_features['confidence'] < 0.35, "Expected confidence < 0.35"
    assert wallet_features['is_verified'] is True, "Expected is_verified is True"

    guidance_verified = fuzzy.compute(
        angle_offset=wallet_features['angle_offset'],
        confidence=wallet_features['confidence'],
        stability=wallet_features['temporal_stability'],
        is_verified=wallet_features['is_verified']
    )

    print(f"  7A Guidance for verified wallet (conf={wallet_features['confidence']:.3f}, urgency={guidance_verified['urgency']:.3f}):")
    print(f"     should_announce = {guidance_verified['should_announce']}")
    assert guidance_verified['should_announce'] is True, (
        f"Regression failure: Qwen-confirmed wallet (conf={wallet_features['confidence']:.3f}, "
        f"urgency={guidance_verified['urgency']:.3f}) was suppressed by announcement gate!"
    )

    # Verify that in production pipeline, announcement actually reaches voice output
    voice_reg = VoiceOutput(rate=160, volume=0.9)
    spoken_phrase = None
    if guidance_verified['should_announce']:
        spoken_phrase = voice_reg.announce_with_angle(
            class_name=wallet_features['class_name'],
            angle_offset=wallet_features['angle_offset'],
            urgency=guidance_verified['urgency'],
            frequency=guidance_verified['frequency'],
            distance=wallet_features['distance'],
        )
    print(f"  7A Spoken phrase delivered to queue: '{spoken_phrase}'")
    assert spoken_phrase is not None, "Announcement phrase should have been queued for voice output"
    voice_reg.wait_until_idle(timeout=10.0)
    voice_reg.shutdown()
    print("  [PASS] 7A: Verified wallet (conf=0.272) successfully passed announcement gate and reached voice queue.")

    # Check 7B: Unverified / unknown response MUST remain suppressed
    # Simulate same detection but unverified (is_verified = False)
    guidance_unverified = fuzzy.compute(
        angle_offset=wallet_features['angle_offset'],
        confidence=wallet_features['confidence'],
        stability=wallet_features['temporal_stability'],
        is_verified=False
    )
    print(f"  7B Guidance for unverified wallet (conf={wallet_features['confidence']:.3f}, urgency={guidance_unverified['urgency']:.3f}):")
    print(f"     should_announce = {guidance_unverified['should_announce']}")
    assert guidance_unverified['should_announce'] is False, (
        "Regression failure: Unverified detection with conf < 0.30 should be suppressed!"
    )

    # Check 7C: Normal YOLO detection with conf >= 0.35 preserves existing behavior
    guidance_normal = fuzzy.compute(
        angle_offset=0.0,
        confidence=0.85,
        stability=0.8,
        is_verified=False
    )
    print(f"  7C Guidance for normal YOLO detection (conf=0.85, is_verified=False):")
    print(f"     should_announce = {guidance_normal['should_announce']}")
    assert guidance_normal['should_announce'] is True, (
        "Regression failure: Normal YOLO detection (conf >= 0.35) should announce when urgency > 0.15!"
    )
    print("  [PASS] 7B & 7C: Unverified detection suppressed, normal YOLO detection preserved.")

    hybrid.shutdown()
    failing_hybrid.shutdown()

    print("\n" + "=" * 70)
    print("  ALL HYBRID PIPELINE TESTS AND REGRESSION CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_hybrid_verification_tests()
