# Colab Training Notebook — train_yolov8_colab.ipynb
# This file provides the cell contents to paste into a Google Colab notebook.
# Since .ipynb is a complex JSON format, copy these cells into Colab manually.

"""
=== CELL 1: Install Dependencies ===
"""
# !pip install ultralytics
# # Upload your dataset to Google Drive or directly to Colab

"""
=== CELL 2: Mount Google Drive (optional, for saving weights) ===
"""
# from google.colab import drive
# drive.mount('/content/drive')

"""
=== CELL 3: Upload and Extract Dataset ===
"""
# # Option A: Upload from Roboflow (recommended)
# # Go to your Roboflow project → Generate → Export → YOLOv8 → Show Download Code
# # Paste the download code here:
#
# # !pip install roboflow
# # from roboflow import Roboflow
# # rf = Roboflow(api_key="YOUR_API_KEY")
# # project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
# # version = project.version(1)
# # dataset = version.download("yolov8")
#
# # Option B: Upload a zip file manually
# # from google.colab import files
# # uploaded = files.upload()  # upload dataset.zip
# # !unzip dataset.zip -d /content/dataset

"""
=== CELL 4: Verify Dataset ===
"""
# import yaml
# import os
#
# data_yaml_path = "/content/dataset/data.yaml"  # adjust path
# with open(data_yaml_path, 'r') as f:
#     data_config = yaml.safe_load(f)
# print("Dataset config:", data_config)
#
# # Count images
# train_imgs = os.listdir(os.path.join(os.path.dirname(data_yaml_path), data_config['train']))
# val_imgs = os.listdir(os.path.join(os.path.dirname(data_yaml_path), data_config['val']))
# print(f"Train images: {len(train_imgs)}")
# print(f"Val images:   {len(val_imgs)}")

"""
=== CELL 5: Check GPU ===
"""
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available:  {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

"""
=== CELL 6: Train YOLOv8-nano ===
"""
# from ultralytics import YOLO
#
# # Load pretrained YOLOv8-nano
# model = YOLO('yolov8n.pt')
#
# # Train with transfer learning
# results = model.train(
#     data=data_yaml_path,
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     patience=20,          # Early stopping
#     save=True,
#     save_period=10,
#     project='/content/runs',
#     name='belonging_detector',
#     exist_ok=True,
#     pretrained=True,
#     optimizer='AdamW',
#     lr0=0.01,
#     cos_lr=True,
#     close_mosaic=10,
#     # Augmentation (defaults are good, these are fine-tuned)
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=10.0,
#     translate=0.1,
#     scale=0.5,
#     fliplr=0.5,
#     mosaic=1.0,
#     mixup=0.1,
# )

"""
=== CELL 7: Validate ===
"""
# val_results = model.val()
# print(f"mAP50:    {val_results.box.map50:.4f}")
# print(f"mAP50-95: {val_results.box.map:.4f}")

"""
=== CELL 8: View Training Curves ===
"""
# from IPython.display import Image, display
# display(Image(filename='/content/runs/belonging_detector/results.png'))

"""
=== CELL 9: Test on Sample Images ===
"""
# import glob
# val_images = glob.glob('/content/dataset/images/val/*.jpg')[:5]
# for img_path in val_images:
#     results = model.predict(img_path, save=True, conf=0.35)
#     print(f"{img_path}: {len(results[0].boxes)} detections")

"""
=== CELL 10: Download Trained Weights ===
"""
# # Copy best.pt to Drive
# import shutil
# src = '/content/runs/belonging_detector/weights/best.pt'
# dst = '/content/drive/MyDrive/belonging_detector_best.pt'
# shutil.copy2(src, dst)
# print(f"Saved to: {dst}")
#
# # Or download directly
# from google.colab import files
# files.download(src)
