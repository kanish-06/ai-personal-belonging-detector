# Belonging Detector — AI-Based Personal Belonging Detection System

An AI-powered assistive system that helps blind users locate personal belongings indoors using a camera and spoken voice guidance.

## Project Overview and Problem Statement

The goal of this project is to build an object-detection system that assists visually impaired or blind users in locating small, commonly misplaced personal belongings in indoor environments. The system uses a camera to perceive the environment and spoken voice guidance to navigate the user toward the detected object. 

This project combines a deep learning object detector for perception with a Mamdani Fuzzy Inference System for decision-making and voice guidance.

## Architecture and End-to-End Workflow

The system is split into two connected components:

1. **Neural Component (Perception)** — A CNN-based object detector (YOLOv8-nano) that finds target objects in a camera frame and outputs bounding boxes and confidence scores.
2. **Fuzzy Component (Decision & Guidance)** — A Mamdani Fuzzy Inference System (FIS) that takes the object's horizontal position, detector confidence, and temporal stability, converting them into smooth, human-friendly voice guidance (direction, urgency, announcement frequency).

### End-to-End Workflow
```
Camera
  |
  v
Object Detector (CNN: YOLOv8-nano, fine-tuned on 5 classes)
  |  outputs: class label, bounding box (x, y, w, h), confidence score
  v
Feature Extractor
  |  derives: angle_offset (horizontal position), confidence, temporal_stability
  v
Fuzzy Inference System (Mamdani FIS)
  |  inputs: angle, confidence, stability
  |  outputs: guidance urgency, announcement frequency, direction
  v
Text-to-Speech / Voice Guidance
  |  speaks short phrases: e.g. "Phone found, directly ahead!"
  v
User (spoken feedback)
```

## Target Objects (5 classes)

| ID | Class    | Description                |
|----|----------|----------------------------|
| 0  | keys     | House/car keys             |
| 1  | wallet   | Wallet, billfold           |
| 2  | phone    | Smartphone, cellphone      |
| 3  | watch    | Wristwatch                 |
| 4  | glasses  | Spectacles, sunglasses     |

## Repository Structure

```
belonging-detector/
  dataset/
    classes.txt              # Class names (one per line)
    data.yaml                # YOLOv8 dataset configuration
    images/train/, images/val/
    labels/train/, labels/val/
    remap_dataset.py         # Script to normalize class names/IDs
  models/
    detector/                # Trained YOLOv8 weights (best.pt)
  src/
    detect.py                # YOLOv8 detection wrapper
    feature_extraction.py    # Raw detection -> fuzzy features (angle, confidence, stability)
    fuzzy_guidance.py        # Mamdani FIS for guidance decisions
    voice_output.py          # Text-to-speech output
    pipeline.py              # End-to-end real-time pipeline
  training/
    train_yolov8.py          # Transfer learning training script
  evaluation/
    evaluate_detector.py     # mAP, precision, recall evaluation
    evaluate_usability.py    # Time-to-locate, latency benchmarks
  requirements.txt
  README.md
  .gitignore
```

## Setup and Installation

### 1. Create Python Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

The project requires PyTorch and OpenCV.

```bash
pip install -r requirements.txt
```

## Dataset Source, Preparation, and Class Remapping

The primary datasets were sourced from Roboflow Universe (e.g., Object Detection HW, personal items datasets).
Because public datasets have inconsistent naming (e.g., `Car-key` vs `keys`, `mobile-phone` vs `phone`), we use a custom script to standardize them.

1. **Download:** Export datasets in YOLOv8 format into `raw_download/`.
2. **Remap & Normalize:** Run `dataset/remap_dataset.py` to handle class name and ID normalization, ensuring all labels strictly map to our 5 target classes (`keys`, `wallet`, `phone`, `watch`, `glasses`).
3. **Verify:** Use `dataset/verify_dataset.py` to confirm the final dataset integrity.

## YOLOv8 Training Instructions

We use transfer learning to fine-tune `yolov8n.pt` for our 5 classes. 

```bash
# Requires GPU for fast training
python training/train_yolov8.py --data dataset/data.yaml --epochs 100 --batch 8 --imgsz 640
```
The best weights are saved to `models/detector/best.pt`.

## Fuzzy Logic Inputs, Outputs, and Rules

The Mamdani Fuzzy Inference System converts numerical inputs into semantic voice guidance.

**Inputs:**
- **Angle (-1 to +1):** Left, Center, Right
- **Confidence (0-1):** Low, Medium, High
- **Stability (0-1):** Unstable, Stable (prevents flickering announcements)

**Outputs:**
- **Guidance Urgency:** Low, Medium, High
- **Announcement Frequency:** Slow, Fast

**Example Rules:**
- IF confidence is High AND angle is Center AND stability is Stable THEN urgency is High, frequency is Fast. -> "Phone found, directly ahead!"
- IF confidence is Medium AND stability is Stable THEN urgency is Medium, frequency is Slow.
- IF stability is Unstable THEN urgency is Low (suppress announcements).

## How to Run the Full Pipeline

```bash
# Full GUI mode for debugging/testing:
python src/pipeline.py

# Search for a specific object only:
python src/pipeline.py --target phone

# Headless mode (e.g., Raspberry Pi deployment):
python src/pipeline.py --no-gui
```

## Evaluation Methods

```bash
# 1. Detector metrics (mAP, precision, recall)
python evaluation/evaluate_detector.py

# 2. Latency benchmark and usability metrics
python evaluation/evaluate_usability.py --benchmark
```

## Current Project Status

- [x] **Environment Setup:** COMPLETED (PyTorch & OpenCV operational)
- [x] **Dataset Pipeline:** COMPLETED (Mappings normalized via `remap_dataset.py`)
- [x] **YOLOv8 Training:** COMPLETED 
- [x] **Evaluation Script Fixes:** COMPLETED
- [x] **Fuzzy System Implementation:** COMPLETED
- [x] **End-to-End Pipeline Integration:** COMPLETED

### Current YOLO Results and Dataset Statistics
- **Model:** YOLOv8-nano
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
- **Validation Result:** `mAP50 = 0.6406`

## Troubleshooting and Limitations

- **CUDA/PyTorch Issues:** If the detector falls back to CPU, ensure you installed the PyTorch version compiled for your specific CUDA toolkit. Check with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Out of Memory (OOM) during Training:** If YOLOv8 training crashes, reduce the batch size in `train_yolov8.py` (e.g., from `--batch 8` to `--batch 4`).
- **Glasses Detection:** The `glasses` class may underperform due to reflections and thin frames. Consider augmenting the dataset with more glasses images if precision/recall is noticeably lower.
- **Hardware Limitations:** The system requires a GPU for training, but inference can run on CPU or edge devices (e.g., Raspberry Pi) using TFLite/ONNX exports.
