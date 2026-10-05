import os
import cv2
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/src")

from detect import ObjectDetector
from feature_extraction import FeatureExtractor
from fuzzy_guidance import FuzzyGuidanceSystem
from depth_estimation import DepthEstimator
from voice_output import VoiceOutput

def test_image(image_path, model_path=None):
    if not os.path.exists(image_path):
        print(f"Error: image {image_path} not found.")
        return

    print("Initializing components...")
    detector = ObjectDetector(model_path=model_path)
    feature_extractor = FeatureExtractor()
    fuzzy = FuzzyGuidanceSystem()
    depth_estimator = DepthEstimator()
    voice = VoiceOutput()
    
    frame = cv2.imread(image_path)
    frame_height, frame_width = frame.shape[:2]
    
    start_time = time.time()
    
    # 1. Detect
    detections = detector.detect(frame)
    
    # 2. Depth
    depth_map = None
    depth_latency = 0.0
    if detections:
        d_start = time.time()
        depth_map = depth_estimator.estimate_depth(frame)
        depth_latency = time.time() - d_start
        print(f"Depth Inference Latency: {depth_latency*1000:.1f} ms")
        
    features_list = []
    for det in detections:
        feat = feature_extractor.extract(det, frame_width, frame_height)
        distance = depth_estimator.extract_depth_for_box(depth_map, det.bbox_xyxy) if depth_map is not None else -1.0
        feat['distance'] = distance
        features_list.append(feat)
        
    feature_extractor.update_history(detections)
    
    # 3. Fuzzy & Voice
    if features_list:
        best_features = max(features_list, key=lambda f: f['confidence'])
        guidance = fuzzy.compute(
            angle_offset=best_features['angle_offset'],
            confidence=best_features['confidence'],
            stability=best_features['temporal_stability'],
        )
        print("Best Detection Features:", best_features)
        print("Fuzzy Guidance:", guidance)
        
        # Test voice output string generation
        if guidance['should_announce']:
            # Actually invoke voice to hear it if local
            voice.announce_with_angle(
                class_name=best_features['class_name'],
                angle_offset=best_features['angle_offset'],
                urgency=guidance['urgency'],
                frequency=guidance['frequency'],
                distance=best_features['distance']
            )
            
    end_time = time.time()
    print(f"Total end-to-end latency: {(end_time - start_time)*1000:.1f} ms")
    
    voice.wait_until_idle()
    voice.shutdown()

if __name__ == "__main__":
    test_image(r"test-data\wallet.jpg")
