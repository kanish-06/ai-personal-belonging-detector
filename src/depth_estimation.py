import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import cv2

class DepthEstimator:
    """
    Handles metric depth estimation using Depth Anything V2.
    """
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
        print(f"[Depth] Loading depth model: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        print(f"[Depth] Model loaded on {self.device}")

    def estimate_depth(self, frame_bgr):
        """
        Runs depth estimation on a BGR frame (from OpenCV).
        Returns a 2D numpy array of metric depth in meters matching the frame's shape.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame_bgr.shape[0], frame_bgr.shape[1]),
            mode="bicubic",
            align_corners=False,
        )
        depth_map = prediction.squeeze().cpu().numpy()
        return depth_map

    def extract_depth_for_box(self, depth_map, bbox_xyxy):
        """
        Extracts a robust depth value (median) from the depth map for a bounding box.
        
        Args:
            depth_map (np.ndarray): 2D depth map in meters.
            bbox_xyxy (tuple): Bounding box coordinates (x1, y1, x2, y2).
            
        Returns:
            float: Estimated distance in meters.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        
        if x2 <= x1 or y2 <= y1:
            return -1.0
            
        box_depth = depth_map[y1:y2, x1:x2]
        
        # Extract median to be robust against background/foreground edges inside the box
        robust_distance = np.median(box_depth)
        return float(robust_distance)
