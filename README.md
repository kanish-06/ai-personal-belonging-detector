# Belonging Detector — AI-Based Personal Belonging Detection System

An AI-powered assistive system that helps blind users locate personal belongings indoors using a camera and spoken voice guidance.

## Project Overview and Problem Statement

The goal of this project is to build an object-detection system that assists visually impaired or blind users in locating small, commonly misplaced personal belongings in indoor environments. The system uses a camera to perceive the environment and spoken voice guidance to navigate the user toward the detected object. 

This project combines a real-time deep learning object detector (YOLOv8-nano), selective multimodal visual verification (Qwen3-VL via Ollama), metric monocular depth estimation (Depth Anything V2), and a Mamdani Fuzzy Inference System for decision-making and voice guidance.

## Architecture and End-to-End Workflow

The system is split into connected perception and decision components:

1. **Primary Perception (YOLOv8-nano)**: Runs on every camera frame to produce bounding boxes and initial confidence scores.
2. **Selective Secondary Verifier (Qwen3-VL 2B)**: When YOLO produces a low-confidence detection (`0.20 <= conf < 0.35`), the bounding box crop is passed to a local Qwen3-VL model running via Ollama. High-confidence detections (`conf >= 0.35`) bypass Qwen entirely.
3. **Metric Depth Estimation (Depth Anything V2 Metric Indoor Small)**: Computes a metric depth map of the environment and extracts physical distance (in meters) from the verified bounding box.
4. **Fuzzy Decision & Guidance (Mamdani FIS)**: Converts horizontal angle offset, detector confidence, and temporal tracking stability into smooth, human-friendly voice guidance (direction, urgency, announcement frequency).
5. **Spoken Voice Guidance (pyttsx3)**: Delivers rate-limited, spoken announcements with distance and direction.

### End-to-End Workflow Diagram

```text
Camera Frame
    │
    ▼
YOLOv8-nano Object Detector (Runs every frame)
    │
    ├─► High Confidence (conf >= 0.35) ──► Direct Pass-Through
    │
    └─► Uncertain (0.20 <= conf < 0.35) ──► Crop Bounding Box
                                                  │
                                                  ▼
                                       Qwen3-VL (2B via Ollama)
                                       - Confirmed / Reconciled: Retain YOLO Box
                                       - Unknown / Error: Suppress Announcement
    │
    ▼
Depth Estimation (Depth Anything V2 Metric Indoor Small)
    │  outputs: metric distance (meters) extracted from the bounding box
    ▼
Feature Extractor
    │  derives: angle_offset (horizontal position), distance, confidence, temporal_stability
    ▼
Fuzzy Inference System (Mamdani FIS)
    │  inputs: angle, confidence, stability
    │  outputs: guidance urgency, announcement frequency, direction
    ▼
Text-to-Speech / Voice Guidance (pyttsx3)
    │  speaks short phrases: e.g. "Wallet detected at 0.9 meters, slightly to your right."
    ▼
User (spoken audio feedback)
```

## Hybrid Detection Architecture (YOLOv8 + Qwen3-VL)

Small items such as keys, wristwatches, and folded wallets often present ambiguous visual features under varying angles and lighting, causing bounding-box CNNs to produce marginal confidence scores or misclassifications.

To address this without sacrificing 30 FPS camera responsiveness:
- **YOLO remains the primary perception engine**: It localizes objects and drives real-time tracking at ~25–35 ms latency.
- **Qwen3-VL acts as a selective secondary verifier**:
  - Only invoked when YOLO detects an object with confidence in the uncertain range `[0.20, 0.35)`.
  - Analyzes the cropped object patch and returns strictly one of: `keys`, `wallet`, `phone`, `watch`, `glasses`, or `unknown`.
  - **Reconciliation Rules**:
    - If Qwen confirms the YOLO class: the detection is retained.
    - If Qwen identifies a different supported class: the class is updated to the verified class, while preserving YOLO's original bounding box and original confidence score.
    - If Qwen cannot identify the object (`unknown`) or fails: the detection and announcement are suppressed.
  - **No YOLO Detection Rule**: If YOLO detects nothing in the frame, Qwen is never called and no synthetic boxes are created.
- **Asynchronous Execution & Spatial Caching**:
  - Live webcam execution runs verification in a background worker thread (`ThreadPoolExecutor`), preventing the camera feed from freezing during the ~1.0–1.5s VLM response.
  - Spatial-temporal caching (IoU matching within a 3.0-second window) prevents repetitive queries for the same object across consecutive video frames.

> **Evaluation Note & Empirical Limitation:**  
> In preliminary benchmark testing across a curated set of 16 test images, Qwen3-VL demonstrated strong classification capabilities, especially on keys and watches where YOLO produced lower confidence scores. However, **this sample size is limited** and does not constitute statistically definitive proof of superior general accuracy across all uncontrolled indoor conditions. The hybrid architecture is designed as a safety/verification layer, not a replacement for domain-specific fine-tuning.

## Target Objects (5 classes)

| ID | Class    | Description                |
|:---|:---------|:---------------------------|
| 0  | keys     | House/car keys             |
| 1  | wallet   | Wallet, billfold           |
| 2  | phone    | Smartphone, cellphone      |
| 3  | watch    | Wristwatch                 |
| 4  | glasses  | Spectacles, sunglasses     |

## Repository Structure

```text
belonging-detector/
│
├── dataset/
│   ├── classes.txt
│   ├── data.yaml
│   ├── remap_dataset.py
│   ├── images/ (train/, val/)
│   └── labels/ (train/, val/)
│
├── models/
│   └── detector/
│       └── best.pt
│
├── src/
│   ├── detect.py              # YOLOv8 object detector
│   ├── qwen_verifier.py       # Qwen3-VL Ollama verifier & response normalizer
│   ├── hybrid_detector.py     # YOLO + Qwen hybrid coordinator & spatial cache
│   ├── depth_estimation.py    # Depth Anything V2 monocular depth
│   ├── feature_extraction.py  # Spatial angle & stability tracking
│   ├── fuzzy_guidance.py      # Mamdani FIS guidance logic
│   ├── voice_output.py        # Subprocess pyttsx3 voice guidance
│   └── pipeline.py            # End-to-end real-time loop
│
├── training/
│   ├── train_yolov8.py
│   └── train_yolov8_colab.py
│
├── evaluation/
│   ├── evaluate_detector.py
│   ├── evaluate_usability.py
│   ├── evaluate_yolo_vs_qwen.py # YOLO vs Qwen comparative evaluation
│   ├── run_full_benchmark.py
│   └── results/
│       ├── yolo_vs_qwen_comparison.json
│       └── yolo_vs_qwen_comparison.csv
│
├── tests/
│   ├── test_hybrid_pipeline.py      # Functional verification suite
│   ├── test_voice_pipeline_path.py  # Voice execution path check
│   ├── test_depth.py
│   ├── test_pipeline_image.py
│   ├── verify_cuda.py
│   └── verify_dataset.py
│
├── test-data/                 # Test images for comparative evaluation
├── requirements.txt
├── README.md
└── .gitignore
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

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and Run Ollama for Qwen3-VL

The secondary verifier requires Ollama running locally:

1. Download and install Ollama from [ollama.com](https://ollama.com/).
2. Pull and start the Qwen3-VL 2B model:
   ```bash
   ollama run qwen3-vl:2b
   ```
3. Ollama runs as a local background service on `http://127.0.0.1:11434`. If Ollama is offline, the pipeline gracefully falls back to pure YOLO mode without crashing.

## How to Run the Pipeline

```bash
# Run full hybrid pipeline (YOLO + Qwen3-VL verification enabled by default):
python src/pipeline.py

# Adjust reliable threshold (e.g., trust YOLO directly if conf >= 0.40):
python src/pipeline.py --confidence 0.40

# Adjust uncertainty floor (e.g., only verify detections with conf >= 0.15):
python src/pipeline.py --uncertain-thresh 0.15

# Search for a specific object only:
python src/pipeline.py --target keys

# Disable Qwen3-VL (run pure YOLO mode):
python src/pipeline.py --no-hybrid

# Headless mode (for single-board computers / Raspberry Pi):
python src/pipeline.py --no-gui
```

## Running Evaluations and Tests

### 1. Hybrid Pipeline Functional Verification
Verifies high-confidence bypass, low-confidence invocation, bounding box preservation, unknown suppression, and voice execution:
```bash
python tests/test_hybrid_pipeline.py
```

### 2. YOLO vs Qwen Comparative Evaluation
Runs side-by-side evaluation across test images, recording per-image predictions, confidence, inference times (with cold-start warm-up handling), and generating JSON & CSV reports:
```bash
python evaluation/evaluate_yolo_vs_qwen.py
```

### 3. Detector Metrics and Full Benchmark
```bash
python evaluation/evaluate_detector.py
python evaluation/run_full_benchmark.py
```

## Current Project Status

- [x] **Environment Setup:** COMPLETED
- [x] **Dataset Pipeline & Remapping:** COMPLETED
- [x] **YOLOv8 Training & Weights:** COMPLETED (`models/detector/best.pt`)
- [x] **Fuzzy Guidance (Mamdani FIS):** COMPLETED
- [x] **Monocular Depth Estimation:** COMPLETED (Depth Anything V2)
- [x] **Hybrid YOLO + Qwen3-VL Architecture:** COMPLETED
- [x] **Voice Output Stability & Audio Path Verification:** COMPLETED
- [ ] **Edge Optimization** 
