import time
import requests
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def main():
    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    print(f"Loading model {model_id}...")
    
    # Load processor and model
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Load sample image
    print("Loading sample image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    
    # Run inference
    print("Running depth estimation...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    prediction = prediction.squeeze().cpu().numpy()
    
    print(f"Metric depth map shape: {prediction.shape}")
    print(f"Min depth: {prediction.min():.2f} meters")
    print(f"Max depth: {prediction.max():.2f} meters")
    print(f"Mean depth: {prediction.mean():.2f} meters")
    
    assert prediction.ndim == 2, "Depth map should be a 2D array"
    assert prediction.shape == image.size[::-1], "Depth map should match original image dimensions"
    
    print("Test passed! Metric depth map produced successfully.")

if __name__ == "__main__":
    main()
