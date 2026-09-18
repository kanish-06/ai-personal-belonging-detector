# AI-Based Personal Belonging Detection System — Full Build Specification

## Purpose of This Document

This is a complete, self-contained build specification for an AI-powered assistive system that helps blind users locate personal belongings indoors using a camera and spoken voice guidance. It is written to be executed directly: every section describes what to build, in what order, with what tools, and what "done" looks like. No prior context is assumed.

Course framing: this project is evaluated under a **Neuro-Fuzzy Techniques** subject, so the final product must genuinely combine a **neural network** (for perception/object detection) with a **fuzzy logic system** (for decision-making/guidance) — not just a plain object detector with a text-to-speech wrapper bolted on.

---

## 1. Problem Statement

Build an object-detection system that helps blind users locate personal belongings in indoor environments using a camera and voice guidance.

---

## 2. Target Object Classes

Final class list (5 classes — chosen because they are small, commonly misplaced indoor personal items, and adding them costs almost nothing in compute: in a CNN object detector, the backbone/feature-extraction cost stays fixed regardless of class count, and only the final classification head grows by a negligible amount per extra class):

1. `keys`
2. `wallet`
3. `phone`
4. `watch`
5. `glasses` (spectacles/sunglasses)

These five were also picked because strong public datasets already exist that cover most or all of them together (see Section 5), minimizing the amount of custom data collection needed.

---

## 3. Why Neuro-Fuzzy, and Where Each Part Lives

A plain object detector plus text-to-speech does not use fuzzy logic meaningfully and will not satisfy the subject's evaluation criteria. The system is therefore split into two deliberate, connected halves:

1. **Neural half** — a CNN-based object detector (YOLOv8-nano) that finds the 5 target objects in a camera frame and outputs bounding boxes + confidence scores.
2. **Fuzzy half** — a Fuzzy Inference System (FIS) that takes the noisy, uncertain outputs of the detector (box size, box position, confidence, frame-to-frame stability) and converts them into smooth, human-friendly voice guidance, instead of jerky, robotic distance numbers or flickering announcements.

Additionally, an **ANFIS (Adaptive Neuro-Fuzzy Inference System)** is used for one specific, well-scoped sub-task: converting bounding-box size into a real-world distance estimate, trained on a small self-collected calibration dataset. This is the literal, textbook "neuro-fuzzy" component — a fuzzy system whose membership functions/rules are tuned via a neural learning procedure on real data, rather than hand-tuned by guesswork.

---

## 4. Full System Architecture

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
  |  (trained on self-collected calibration data: object at known distances)
  v
Fuzzy Inference System (Mamdani, guidance brain)
  |  inputs:  distance, angle, confidence, stability
  |  outputs: guidance urgency/verbosity, announcement frequency, direction phrase
  v
Text-to-Speech / Voice Guidance
  |  speaks short phrases: e.g. "Phone found, two o'clock, near."
  v
User (spoken feedback)
```

---

## 5. Datasets (use these instead of collecting everything from scratch)

Use public datasets as the primary training data, supplemented with a small amount of self-collected photos for the demo environment and for ANFIS distance calibration (which must be self-collected regardless, since it requires precise known distances).

### Primary multi-class datasets (Roboflow Universe — export in YOLOv8 format)

1. **Object Detection HW** — classes: `glasses`, `watch`, `wallet`, `phone`, `plasticbottle` — covers 4 of our 5 target classes directly.
   https://universe.roboflow.com/robotics-lab-3-nryht/object-detection-hw

2. **Combined** — classes: `keys`, `carkeys`, `Wallet`, `mobile-phone`, `watch` — 5.62k images, covers keys/wallet/phone/watch.
   https://universe.roboflow.com/wallet-voncm/combined-fxxyu

3. **personal items** — classes: `keys`, `wallet`, `cellphone`, `watch`, `sunglass` — 882 images, covers all 5 target classes (sunglass ≈ glasses).
   https://universe.roboflow.com/memoiadataset/personal-items

4. **item** (lost-item style dataset) — classes: `phone`, `wallet`, `laptop`, `pen`, `AirPods`, `credit_card`, `Tablet_PC` — 3.92k images, strong phone/wallet coverage.
   https://universe.roboflow.com/lostitem/item-8frmn

5. **yolohassalt** — classes: `phone`, `wallet`, `Key`, `laptop`, `Flashdisk` — 4.76k images.
   https://universe.roboflow.com/finder/yolohassalt-deqzt

### Supplementary single-class datasets (use to boost a weak class after initial training)

6. **Cell phone** (Kaggle) — 1000 JPEG images + YOLO-format bounding box labels, phone-only.
   https://www.kaggle.com/datasets/samuelayman/cell-phone

7. **Wallet** (Roboflow Universe) — 1.5k images, wallet-only.
   https://universe.roboflow.com/senior-project-final-datasets/wallet-nfk4a

8. **Glasses detection** (Roboflow Universe) — glasses-only, several small options if the glasses class needs more data:
   https://universe.roboflow.com/pkh0308/glasses_detection-hfwtd
   https://universe.roboflow.com/it-3pwlf/glasses-detection-7anef

### How to combine datasets

- Download each dataset from Roboflow in **YOLOv8 format** (gives `images/`, `labels/`, `data.yaml`).
- Create one new Roboflow project, upload images from each source dataset into it, and **remap every class name to exactly one of the 5 final class names** (`keys`, `wallet`, `phone`, `watch`, `glasses`) during import — Roboflow's class-renaming step in the upload/annotate flow handles this.
- Generate one final merged version and export it in YOLOv8 format. This becomes the single training dataset.
- Final class ID mapping to use everywhere in the project: `0=keys, 1=wallet, 2=phone, 3=watch, 4=glasses`.

---

## 6. Technology Stack

- **Object detection:** YOLOv8-nano (`ultralytics` Python package) — small enough for real-time use, supports easy transfer learning, exportable to TFLite/ONNX for edge devices.
- **Training environment:** Google Colab (free GPU tier) is recommended for the actual fine-tuning run — a full `ultralytics`/PyTorch install with CUDA support needs several GB of disk and a GPU is strongly preferred for reasonable training time. A local machine with a GPU also works.
- **Fuzzy logic:** Python `scikit-fuzzy` (`skfuzzy`) library for the Mamdani Fuzzy Inference System.
- **ANFIS:** Either a small custom ANFIS implementation in Python (a Sugeno-type fuzzy layer combined with gradient-based tuning, several open reference implementations exist on GitHub — search "ANFIS python" on GitHub for a base to adapt), or the `anfis` PyPI package, adapted for a single-input (box_area_ratio) single-output (distance) regression.
- **Voice guidance:** `pyttsx3` (fully offline TTS) as the default; Google Text-to-Speech (`gTTS`) as an optional online alternative for more natural voices.
- **Camera interface:** OpenCV (`cv2`) for frame capture and drawing/debugging overlays.
- **Target hardware:** Raspberry Pi 4/5 with a Pi Camera module and earphones/speaker is the recommended deployment target — cheap, portable, and looks like a real assistive device. A laptop with a USB webcam is an acceptable fallback for development and demoing if a Pi is not available. Optional: an ultrasonic distance sensor can be added as a second distance input, fused with the vision-based distance estimate inside the fuzzy system, as a bonus neuro-fuzzy sensor-fusion demonstration.

---

## 7. Repository Structure to Create

```
belonging-detector/
  dataset/
    classes.txt                  # keys, wallet, phone, watch, glasses
    data.yaml                    # YOLOv8 dataset config
    images/train/, images/val/
    labels/train/, labels/val/
    calibration/                 # ANFIS distance-calibration photos + distance labels
  models/
    detector/                    # trained YOLOv8 weights (best.pt) go here
    anfis_distance_model.pkl     # trained ANFIS model for box-size -> distance
  src/
    detect.py                    # loads YOLOv8 model, runs inference on a frame
    feature_extraction.py        # box -> {box_area_ratio, angle_offset, confidence, stability}
    anfis_distance.py            # trains + runs the ANFIS distance regression model
    fuzzy_guidance.py            # Mamdani FIS: features -> urgency/direction/frequency
    voice_output.py              # converts fuzzy output into spoken phrases (TTS)
    pipeline.py                  # ties camera -> detect -> features -> ANFIS -> fuzzy -> voice together, real-time loop
  training/
    train_yolov8.py or train_yolov8_colab.ipynb   # transfer-learning training script/notebook
  evaluation/
    evaluate_detector.py         # mAP, precision/recall per class
    evaluate_distance.py         # ANFIS distance error vs fixed-formula baseline
    evaluate_usability.py        # logging for time-to-locate, false-announcement count
  requirements.txt
  README.md
```

---

## 8. Execution Steps (in order — no fixed timeline, just dependency order)

### Step 1 — Environment setup
Set up a Python environment (locally or on Colab) with: `ultralytics`, `opencv-python`, `scikit-fuzzy`, `pyttsx3`, `numpy`, `scikit-learn`. Confirm a GPU is available if training locally; otherwise plan to use Colab for the training step only.

### Step 2 — Acquire and merge datasets
Download the datasets listed in Section 5, remap classes to the final 5-class scheme, and export one merged YOLOv8-format dataset into `dataset/`.

### Step 3 — Supplement with self-collected photos
Take additional photos of the team's own keys/wallet/phone/watch/glasses in the actual demo environment (varying angle, distance 20cm-300cm, lighting, occlusion, background clutter; include ~10-15% negative images with none of the target objects). Label them the same way (Roboflow or LabelImg, YOLO format) and merge into the same `dataset/` folders. This closes the gap between generic public-dataset images and the team's real demo conditions.

### Step 4 — Train the detector (transfer learning)
Fine-tune YOLOv8-nano (`yolov8n.pt` as the starting checkpoint) on the merged 5-class dataset. Use data augmentation (rotation, brightness/contrast jitter, mosaic — enabled by default in `ultralytics`). Train until validation mAP plateaus, using early stopping. Export the trained weights (`best.pt`) into `models/detector/`.

### Step 5 — Build the feature extraction module
Implement `feature_extraction.py`: given a raw detection (bounding box, confidence), compute:
- `box_area_ratio` = (box width × box height) / (frame width × frame height) — a proxy for closeness (larger = closer).
- `angle_offset` = normalized horizontal offset of the box center from the frame center, ranging -1 (far left) to +1 (far right).
- `confidence` = detector's confidence score, passed through directly.
- `temporal_stability` = fraction of the last N frames in which this object class was detected in roughly the same location (reduces flicker/false positives).

### Step 6 — Collect ANFIS calibration data and train the distance model
Place each target object at a series of known distances from the camera (e.g. 20, 40, 60, 100, 150, 200, 300 cm), record the resulting `box_area_ratio` at each distance, and build a small labeled dataset of (box_area_ratio → true distance) pairs. Train an ANFIS model (or, as a fallback if ANFIS proves too complex to get working reliably, a simple regression such as polynomial or Sugeno fuzzy rules that are hand-fit) on this data. Save the trained model to `models/anfis_distance_model.pkl`. This becomes the actual "distance" input used by the fuzzy guidance system.

### Step 7 — Design and implement the Fuzzy Inference System
Build the Mamdani-style FIS in `fuzzy_guidance.py` using `scikit-fuzzy`, with the inputs/outputs/rules defined in Section 9 below. Validate its behavior against constructed test cases (e.g. manually feed in "distance=near, angle=center, confidence=high" and confirm the output is "high urgency, center direction").

### Step 8 — Voice guidance integration
Implement `voice_output.py`: map the fuzzy system's urgency/direction/frequency output into short spoken phrases (e.g. "Wallet found, slightly left, close by.") and speak them via `pyttsx3`. Keep phrases short, consistent in structure, and rate-limited according to the fuzzy "announcement frequency" output so the user isn't overwhelmed with constant chatter.

### Step 9 — End-to-end real-time pipeline
Implement `pipeline.py`: continuously capture frames from the camera, run them through detection → feature extraction → ANFIS distance → fuzzy guidance → voice output, in a loop. Test this live, holding each of the 5 objects at varying distances/angles and confirming the spoken guidance makes sense.

### Step 10 — Deploy to target hardware
If using Raspberry Pi: export the trained YOLOv8 model to a Pi-friendly format (TFLite or ONNX, via `model.export(format='tflite')` in `ultralytics`), install the same pipeline dependencies on the Pi, connect the Pi Camera and speaker/earphones, and run `pipeline.py` on-device. Measure and optimize frame rate if it's too slow (reduce input resolution, use the nano model variant, skip frames if needed).

### Step 11 — Evaluation
Run the scripts in `evaluation/`:
- Detector: mAP, precision, recall per class on a held-out validation set.
- Distance estimation: compare ANFIS-based distance error against a naive fixed pinhole-camera-formula baseline, on a separate set of known-distance test shots.
- Usability: time-to-locate-object trials, false-announcement counts, ideally with a blindfolded test user.
- Latency: end-to-end frames-per-second and detection-to-speech delay, on the actual target hardware.

### Step 12 — Polish and document
Write up the final report/documentation covering the architecture, the specific neuro-fuzzy contribution (Section 3), datasets used, training results, fuzzy rule table, evaluation numbers, and a demo video/script. Prepare a live demo run-through on the target hardware.

---

## 9. Fuzzy Inference System — Full Design

### Inputs and fuzzy sets

| Input | Fuzzy Sets |
|---|---|
| Distance (from ANFIS, in cm) | Near, Medium, Far |
| Angle (from angle_offset, -1 to +1) | Left, Center, Right |
| Confidence (detector confidence, 0-1) | Low, Medium, High |
| Stability (temporal_stability, 0-1) | Unstable, Stable |

### Outputs and fuzzy sets

| Output | Fuzzy Sets |
|---|---|
| Guidance urgency / verbosity | Low, Medium, High |
| Announcement frequency | Slow, Fast |

### Suggested membership function ranges (tune during Step 7 based on real test data)

- Distance (cm): Near = 0-60 (peak ~20-40), Medium = 40-150 (peak ~90), Far = 120-300+ (peak ~200+).
- Angle (-1 to +1): Left = -1 to -0.15, Center = -0.3 to +0.3, Right = +0.15 to +1.
- Confidence (0-1): Low = 0-0.5, Medium = 0.35-0.75, High = 0.6-1.0.
- Stability (0-1): Unstable = 0-0.5, Stable = 0.4-1.0.

### Core rule set (extend with more rules during tuning)

1. IF distance is Near AND angle is Center AND confidence is High THEN urgency is High, frequency is Fast. → "Object is right in front of you, reach forward."
2. IF distance is Near AND angle is Left AND confidence is High THEN urgency is High, frequency is Fast. → "Object is close, slightly to your left."
3. IF distance is Near AND angle is Right AND confidence is High THEN urgency is High, frequency is Fast. → "Object is close, slightly to your right."
4. IF distance is Medium AND confidence is Medium OR High THEN urgency is Medium, frequency is Slow. → periodic direction update, not urgent.
5. IF distance is Far THEN urgency is Low, frequency is Slow. → occasional "still scanning" style update, not a full announcement.
6. IF stability is Unstable THEN urgency is Low regardless of other inputs. → suppress announcements from flickering/spurious detections.
7. IF confidence is Low THEN do not announce at all (handled as a hard override before the FIS, or by forcing urgency to its minimum output).

---

## 10. ANFIS Distance Model — Design Notes

- **Input:** `box_area_ratio` (single input, 0-1).
- **Output:** estimated distance in cm.
- **Training data:** self-collected calibration set — same object photographed at a range of known distances (Step 6). Aim for at least 30-50 calibration points per object class, spread across the full 20-300cm range, to get a reasonably smooth learned mapping.
- **Why ANFIS over a fixed formula:** a fixed pinhole-camera formula assumes a constant real-world object size and a fixed camera focal length/orientation; ANFIS instead learns the actual relationship from data, which naturally absorbs perspective effects, slight object-orientation variation, and camera-specific quirks, without needing to hand-derive a formula.
- **Fallback if ANFIS training proves unstable or too complex to finish:** fit a Sugeno-type fuzzy system with manually-set, small number of rules (e.g. 3-4 rules mapping ranges of `box_area_ratio` to distance formulas), or a plain regression (polynomial or log fit) as a placeholder while ANFIS is refined — the rest of the pipeline (Section 4) does not change either way.

---

## 11. Evaluation Metrics and Success Criteria

- **Detector:** mAP50 and mAP50-95 across all 5 classes; per-class precision/recall to catch any class that's significantly underperforming (commonly `glasses`, being small/thin/reflective, may need more data).
- **Distance estimation:** mean absolute error (cm) of ANFIS predictions vs. ground truth, compared against a naive fixed-formula baseline, on a held-out set of known distances.
- **Usability:** average time-to-locate an object across test trials; number of false or unhelpful announcements per trial; qualitative feedback if a blindfolded or visually-impaired test user is available.
- **Latency:** end-to-end frames-per-second and detection-to-spoken-guidance delay, measured on the actual deployment hardware (Raspberry Pi if used).
- **Overall success criterion:** the system reliably detects all 5 object classes in varied real-world conditions, gives voice guidance that a blindfolded test user can follow to locate the object faster than by unaided searching, and the write-up clearly demonstrates the neural (detector) + fuzzy (guidance FIS) + neuro-fuzzy (ANFIS distance model) components working together as required by the subject.

---

## 12. Known Constraints and Practical Notes

- Installing the full `ultralytics` + PyTorch (with CUDA) stack requires several GB of disk space and benefits heavily from a GPU; do the actual training on Google Colab (free GPU tier) or a machine with adequate resources, not on a disk-constrained sandbox or low-end machine.
- Adding classes 4 and 5 (`watch`, `glasses`) does not meaningfully change training time, model size, or inference speed compared to a 3-class version — the backbone computation dominates regardless of class count.
- `glasses` may be the hardest class to detect reliably (thin frames, reflective lenses, variable styles) — plan to allocate more training images to this class than the others if per-class validation metrics show it lagging.
- Keep voice guidance phrases short and infrequent when confidence is low or objects are far away — over-announcing erodes user trust in the system faster than under-announcing.
