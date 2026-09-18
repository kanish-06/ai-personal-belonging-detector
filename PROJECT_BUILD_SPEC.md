# AI-Based Personal Belonging Detection System — Full Build Specification

## Purpose of This Document

This is a complete, self-contained build specification for an AI-powered assistive system that helps blind users locate personal belongings indoors using a camera and spoken voice guidance. It is written to be executed directly: every section describes what to build, in what order, with what tools, and what "done" looks like.

---

## 1. Problem Statement

Build an object-detection system that helps blind users locate personal belongings in indoor environments using a camera and voice guidance.

---

## 2. Target Object Classes

Final class list (5 classes — chosen because they are small, commonly misplaced indoor personal items):

1. `keys`
2. `wallet`
3. `phone`
4. `watch`
5. `glasses` (spectacles/sunglasses)

---

## 3. System Architecture & Component Roles

The system is split into two connected halves:

1. **Neural Component (Perception)** — a CNN-based object detector (YOLOv8-nano) that finds the 5 target objects in a camera frame and outputs bounding boxes + confidence scores.
2. **Fuzzy Component (Guidance Decision)** — a Mamdani Fuzzy Inference System (FIS) that takes the perception features (horizontal angle offset, detector confidence, temporal stability across frames) and converts them into smooth, human-friendly voice guidance (urgency, direction, announcement frequency).

---

## 4. Full System Architecture

```
Camera
  |
  v
YOLOv8-nano Object Detector
  |  outputs: class label, bounding box, confidence score
  v
Feature Extractor
  |  derives: angle_offset (position), confidence, temporal_stability
  v
Mamdani Fuzzy Inference System (FIS)
  |  inputs: angle, confidence, stability
  |  outputs: direction, urgency, announcement frequency
  v
Text-to-Speech (pyttsx3)
  |  speaks guidance phrases
  v
User (spoken feedback)
```

---

## 5. Datasets

Use public datasets as the primary training data, supplemented with self-collected photos for the demo environment.

### Primary multi-class datasets (Roboflow Universe — export in YOLOv8 format)

1. **Object Detection HW** — classes: `glasses`, `watch`, `wallet`, `phone`, `plasticbottle`.
2. **Combined** — classes: `keys`, `carkeys`, `Wallet`, `mobile-phone`, `watch`.
3. **personal items** — classes: `keys`, `wallet`, `cellphone`, `watch`, `sunglass`.
4. **item** — classes: `phone`, `wallet`, `laptop`, `pen`, `AirPods`, `credit_card`.
5. **yolohassalt** — classes: `phone`, `wallet`, `Key`, `laptop`, `Flashdisk`.

---

## 6. Technology Stack

- **Object detection:** YOLOv8-nano (`ultralytics` Python package).
- **Fuzzy logic:** Python `scikit-fuzzy` (`skfuzzy`) library for the Mamdani FIS.
- **Voice guidance:** `pyttsx3` (fully offline TTS).
- **Camera interface:** OpenCV (`cv2`) for frame capture and visual overlays.

---

## 7. Repository Structure

```
belonging-detector/
  dataset/
    classes.txt                  # keys, wallet, phone, watch, glasses
    data.yaml                    # YOLOv8 dataset config
    images/train/, images/val/
    labels/train/, labels/val/
  models/
    detector/                    # trained YOLOv8 weights (best.pt)
  src/
    detect.py                    # loads YOLOv8 model, runs inference
    feature_extraction.py        # box -> {angle_offset, confidence, stability}
    fuzzy_guidance.py            # Mamdani FIS: features -> urgency/direction/frequency
    voice_output.py              # text-to-speech output (pyttsx3)
    pipeline.py                  # ties camera -> detect -> features -> fuzzy -> voice
  training/
    train_yolov8.py              # transfer-learning training script
  evaluation/
    evaluate_detector.py         # mAP, precision/recall per class
    evaluate_usability.py        # logging for time-to-locate, latency benchmark
  requirements.txt
  README.md
```

---

## 8. Execution Steps

### Step 1 — Environment setup
Set up a Python environment with `ultralytics`, `opencv-python`, `scikit-fuzzy`, `pyttsx3`, `numpy`, `scikit-learn`.

### Step 2 — Dataset preparation & remapping
Remap classes to the final 5-class scheme (`keys`, `wallet`, `phone`, `watch`, `glasses`).

### Step 3 — Train the detector
Fine-tune YOLOv8-nano (`yolov8n.pt`) on the merged dataset. Save `best.pt` in `models/detector/`.

### Step 4 — Feature extraction module
Implement `feature_extraction.py`: extract `angle_offset`, `confidence`, and `temporal_stability`.

### Step 5 — Fuzzy Inference System
Implement `fuzzy_guidance.py` using `scikit-fuzzy` (inputs: angle, confidence, stability -> outputs: urgency, frequency, direction).

### Step 6 — Voice guidance integration
Implement `voice_output.py` to map FIS outputs into spoken phrases via `pyttsx3`.

### Step 7 — End-to-end pipeline
Implement `pipeline.py`: Camera -> YOLOv8-nano -> Features -> Mamdani FIS -> Voice Output.

### Step 8 — Evaluation
Run evaluation scripts in `evaluation/`.

---

## 9. Fuzzy Inference System — Design

### Inputs and fuzzy sets

| Input | Fuzzy Sets |
|---|---|
| Angle (from angle_offset, -1 to +1) | Left, Center, Right |
| Confidence (detector confidence, 0-1) | Low, Medium, High |
| Stability (temporal_stability, 0-1) | Unstable, Stable |

### Outputs and fuzzy sets

| Output | Fuzzy Sets |
|---|---|
| Guidance urgency | Low, Medium, High |
| Announcement frequency | Slow, Fast |

### Core rule set

1. IF confidence is High AND angle is Center AND stability is Stable THEN urgency is High, frequency is Fast.
2. IF confidence is High AND angle is Left/Right AND stability is Stable THEN urgency is High, frequency is Fast.
3. IF confidence is Medium AND stability is Stable THEN urgency is Medium, frequency is Slow.
4. IF confidence is High AND stability is Unstable THEN urgency is Medium, frequency is Slow.
5. IF confidence is Low THEN urgency is Low, frequency is Slow.
6. IF stability is Unstable AND confidence is Low/Medium THEN urgency is Low, frequency is Slow.

---

## 10. Evaluation Metrics

- **Detector:** mAP50 and mAP50-95 across all 5 classes.
- **Usability & Latency:** average time-to-locate, FPS, and detection-to-speech delay.
