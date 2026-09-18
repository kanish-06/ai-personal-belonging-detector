# Belonging Detector — AI-Based Personal Belonging Detection System

An AI-powered assistive system that helps blind users locate personal belongings indoors using a camera and spoken voice guidance.

## Project Overview and Problem Statement

The goal of this project is to build an object-detection system that assists visually impaired or blind users in locating small, commonly misplaced personal belongings in indoor environments. The system uses a camera to perceive the environment and spoken voice guidance to navigate the user toward the detected object. 

This project is evaluated under **Neuro-Fuzzy Techniques** (NFT) and combines a neural network for perception with a fuzzy logic system for decision-making and guidance.

## Neuro-Fuzzy Architecture and End-to-End Workflow

The system is split into deliberate, connected halves:

1. **Neural Component (Perception)** — A CNN-based object detector (YOLOv8-nano) that finds the target objects in a camera frame and outputs bounding boxes and confidence scores.
2. **Neuro-Fuzzy Component (Distance Estimation)** — An ANFIS (Adaptive Neuro-Fuzzy Inference System) that converts bounding box size into a real-world distance estimate (in cm), trained on a self-collected calibration dataset.
3. **Fuzzy Component (Decision & Guidance)** — A Mamdani Fuzzy Inference System (FIS) that takes the noisy outputs of the detector and distance estimator, and converts them into smooth, human-friendly voice guidance (urgency, direction, announcement frequency) instead of jerky robotic numbers.

### End-to-End Workflow
```
Camera
  |
  v
Object Detector (CNN: YOLOv8-nano, fine-tuned on 5 classes)
  |  outputs: class label, bounding box (x, y, w, h), confidence score
  v
Feature Extractor
  |  derives: box_area_ratio (distance proxy), angle_offset (position),
  |           confidence, temporal_stability (across recent frames)
  v
ANFIS Distance Model
  |  maps: box_area_ratio -> estimated real-world distance (cm)
  v
Fuzzy Inference System (Mamdani, guidance brain)
  |  inputs: distance, angle, confidence, stability
  |  outputs: guidance urgency/verbosity, announcement frequency, direction phrase
  v
Text-to-Speech / Voice Guidance
  |  speaks short phrases: e.g. "Phone found, two o'clock, near."
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
    calibration/             # ANFIS calibration photos + distance labels
    remap_dataset.py         # Script to normalize class names/IDs
  models/
    detector/                # Trained YOLOv8 weights (best.pt)
    anfis_distance_model.pkl # Trained ANFIS distance model
  src/
    detect.py                # YOLOv8 detection wrapper
    feature_extraction.py    # Raw detection → fuzzy features
    anfis_distance.py        # ANFIS distance estimation
    fuzzy_guidance.py        # Mamdani FIS for guidance decisions
    voice_output.py          # Text-to-speech output
    pipeline.py              # End-to-end real-time pipeline
  training/
    train_yolov8.py          # Transfer learning training script
  evaluation/
    evaluate_detector.py     # mAP, precision, recall evaluation
    evaluate_distance.py     # ANFIS vs baseline distance evaluation
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

The project requires a CUDA-enabled PyTorch environment on Windows for optimal training.

```bash
pip install -r requirements.txt
```
*(If PyTorch CPU is installed by default, replace it with the CUDA 12.4 version following official PyTorch instructions).*

## Dataset Source, Preparation, and Class Remapping

The primary datasets were sourced from Roboflow Universe (e.g., Object Detection HW, personal items datasets).
Because public datasets have inconsistent naming (e.g., `Car-key` vs `keys`, `mobile-phone` vs `phone`), we use a custom script to standardize them.

1. **Download:** Export datasets in YOLOv8 format into `raw_download/`.
2. **Remap & Normalize:** Run `dataset/remap_dataset.py` to handle class name and ID normalization, ensuring all labels strictly map to our 5 target classes (`keys`, `wallet`, `phone`, `watch`, `glasses`).
3. **Verify:** Use `dataset/verify_dataset.py` to confirm the final dataset integrity.

## YOLOv8 Training Instructions

We use transfer learning to fine-tune `yolov8n.pt` for our 5 classes. 

```bash
# Requires an NVIDIA GPU (e.g., RTX 4050 6GB)
python training/train_yolov8.py --data dataset/data.yaml --epochs 100 --batch 8 --imgsz 640
```
*(Batch size is set to 8 conservatively for a 6GB VRAM GPU to avoid Out-Of-Memory errors).*
The best weights are saved to `models/detector/best.pt`.

## ANFIS Distance Calibration and Training

The system currently defaults to an inverse-square distance heuristic. To calibrate the ANFIS model for real-world distance estimation:

1. **Collect Calibration Data:** Photograph the 5 objects at known distances (e.g., 20cm, 50cm, 100cm, 200cm). 
2. **Train:** Run the ANFIS training script to generate `models/anfis_distance_model.pkl`.
```bash
python src/anfis_distance.py
```

## Fuzzy Logic Inputs, Outputs, and Rules

The Mamdani Fuzzy Inference System converts numerical inputs into semantic voice guidance.

**Inputs:**
- **Distance (cm):** Near (0-60), Medium (40-150), Far (120-300+)
- **Angle (-1 to +1):** Left, Center, Right
- **Confidence (0-1):** Low, Medium, High
- **Stability (0-1):** Unstable, Stable (prevents flickering announcements)

**Outputs:**
- **Guidance Urgency / Verbosity:** Low, Medium, High
- **Announcement Frequency:** Slow, Fast

**Example Rules:**
- IF distance is Near AND angle is Center AND confidence is High THEN urgency is High, frequency is Fast. → "Object is right in front of you, reach forward."
- IF distance is Medium AND confidence is Medium OR High THEN urgency is Medium, frequency is Slow.
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

# 2. Distance estimation error (ANFIS vs fixed formula baseline)
python evaluation/evaluate_distance.py

# 3. Latency benchmark and usability metrics
python evaluation/evaluate_usability.py --benchmark
```

## Current Project Status

- [x] **Environment Setup:** COMPLETED (CUDA 12.4 enabled PyTorch operational)
- [x] **Dataset Pipeline:** COMPLETED (Mappings normalized via `remap_dataset.py`)
- [x] **YOLOv8 Training:** COMPLETED 
- [x] **Evaluation Script Fixes:** COMPLETED (Fixed property access bug in `evaluate_detector.py`)
- [ ] **ANFIS Distance Calibration:** NOT STARTED
- [ ] **Fuzzy System Implementation:** IN PROGRESS / TODO
- [ ] **End-to-End Pipeline Integration:** IN PROGRESS / TODO

### Current YOLO Results and Dataset Statistics
- **Model:** YOLOv8-nano
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
- **Validation Result:** `mAP50 = 0.6406`

## Exact Next Steps for Teammates

1. **ANFIS Calibration (`src/anfis_distance.py`):** Collect actual distance calibration photos (or generate synthetic data) to train the `anfis_distance_model.pkl`.
2. **Fuzzy Logic Design (`src/fuzzy_guidance.py`):** Implement the `scikit-fuzzy` Mamdani rules outlined in the specification.
3. **Pipeline Integration (`src/pipeline.py`):** Connect the trained detector, the ANFIS distance estimator, and the fuzzy guidance system into the real-time camera loop.


## Troubleshooting and Limitations

- **CUDA/PyTorch Issues:** If the detector falls back to CPU, ensure you installed the PyTorch version compiled for your specific CUDA toolkit (currently CUDA 12.4). Check with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Out of Memory (OOM) during Training:** If YOLOv8 training crashes, reduce the batch size in `train_yolov8.py` (e.g., from `--batch 8` to `--batch 4`).
- **Glasses Detection:** The `glasses` class may underperform due to reflections and thin frames. Consider augmenting the dataset with more glasses images if precision/recall is noticeably lower.
- **Hardware Limitations:** The system requires a GPU for training, but inference can run on CPU or edge devices (e.g., Raspberry Pi) using TFLite/ONNX exports.
