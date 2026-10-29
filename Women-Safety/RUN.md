# Women Safety AI — Incident Detection & Alert System

A production-grade, modular computer-vision safety surveillance system on Windows 10 (Python 3.11) with PyTorch ResNet18-GRU, Ultralytics YOLOv8, MediaPipe, OpenCV, and central configuration.

---

## 1. Quick Start (Windows Command Prompt)

Open **Windows Command Prompt (`cmd`)**:

```cmd
cd /d D:\project\Women-Safety
venv\Scripts\activate
python main.py
```

The interactive menu prompts you:
```text
======================================================================
  WOMEN SAFETY AI - INCIDENT DETECTION & ALERT SYSTEM
======================================================================
  Select what you want to monitor:
    [1] Webcam (Live camera feed)
    [2] Video File (Upload / browse .mp4, .avi, etc.)
    [3] Image File (Upload / browse .jpg, .png, etc.)
    [4] RTSP / IP Camera Stream (Enter stream URL)
    [5] Exit
======================================================================
```
- Press **`1`** (or press Enter) to start your live webcam.
- Selecting **`2`** or **`3`** opens the native Windows File Explorer dialog to click and select any video or photo directly!
- Press **`q`** in the window to exit at any time.

### Keyboard Controls During Live Monitoring
| Key | Action | Description |
|:---:|:---|:---|
| **`s`** | **Manual SOS Trigger** | Immediately flags an active distress state and elevates risk to emergency level |
| **`r`** | **Reset State Engine** | Manually clears risk scores and returns state machine to `NORMAL` |
| **`q`** | **Quit / Exit** | Gracefully closes video stream, releases camera, and saves evidence |

---

## 2. Project Organization

The repository has been structured cleanly into modular subdirectories:

```text
Women-Safety/
├── main.py                     # Central interactive application entry point
├── config.py                   # Central application configuration & thresholds
├── requirements.txt            # Python dependencies
├── README.md                   # Architecture & system overview
├── RUN.md                      # Operational guide & verification commands
├── .env.example                # Safe environment variable template
├── .gitignore                  # Git exclusions for outputs, logs, and environments
├── models/                     # Central organized model checkpoints
│   ├── person_detection/       # yolov8n.pt (YOLOv8 + ByteTrack tracker)
│   ├── gender_classification/  # working_m_&_F.pt (YOLO classes {0: female, 1: male})
│   ├── violence_detection/     # best.pt (PyTorch ResNet18-GRU on RWF-2000)
│   ├── fall_detection/         # yolov8n-pose.pt (COCO 17-keypoint skeleton)
│   └── sos_detection/          # Documentation & heuristic distress detectors
├── src/                        # Modular source code
│   ├── detection/              # Person, Face, Gender, and Pose detectors
│   ├── tracking/               # Multi-object tracking utilities
│   ├── incident_detection/     # Violence, Snatching, SOS, and Risk assessment
│   ├── alerts/                 # Background async alert queue, Telegram, and Call
│   └── utilities/              # Incident snapshot manager and video processor
├── outputs/
│   └── incidents/              # Qualifying incident snapshots & JSON logs (YYYY-MM-DD/)
├── tests/                      # Automated test suite (test_basic.py)
└── logs/                       # System diagnostic logs
```

---

## 3. Integrated Models & Verification

| Module | Model File | Framework | Architecture & Input Specs | Classes / Output |
|:---|:---|:---|:---|:---|
| **Person Detection & Tracking** | `models/person_detection/yolov8n.pt` | Ultralytics YOLO | 640x640 RGB + ByteTrack tracker | Bounding boxes, Track IDs, Person counts |
| **Gender Classification** | `models/gender_classification/working_m_&_F.pt` | Ultralytics YOLO | Frame-level & crop IoU matching + Multi-frame EMA voting | `{0: 'female', 1: 'male'}`, fallback: `'Unknown'` |
| **Violence / Fight Detection** | `models/violence_detection/best.pt` | PyTorch | ResNet18 backbone (512-dim) + 1-layer GRU (hidden=256) + Linear(256, 2). 8 frames, 160x160, ImageNet norm. Val Acc: ~74.25% | `['NonFight', 'Fight']` |
| **Fall Detection** | `models/fall_detection/yolov8n-pose.pt` | Ultralytics YOLO | 17 COCO Keypoints + Temporal pose state machine | Collapses, aspect ratio inversion |
| **SOS & Distress Gestures** | Heuristic + YOLOv8-pose | PyTorch / OpenCV | Wrists elevated above head / defensive shielding | Overhead wave, shielding, manual 'S' key |
| **Theft & Snatching** | Heuristic + YOLOv8 | PyTorch / OpenCV | Carried objects + departure velocity differential ($> 650\text{ px/s}$) | Suden grab-and-flee actions |

---

## 4. Female Victim Highlighting & Snapshot Organization

### A. Complete Removal of Restricted Area
The restricted zone polygon, boundaries, warnings, and UI overlays have been **completely removed** from the application.

### B. Female Victim Red-Box Highlighting
When an incident is corroborated involving a woman:
- Only the identified female victim is highlighted in **BOLD RED** (`VICTIM_BOX_COLOR = (0, 0, 255)`, thickness: 3).
- Label above box: `ID <p_id> | FEMALE | POTENTIAL WOMAN SAFETY INCIDENT`.
- Uninvolved females remain in purple (`ID <p_id> | FEMALE`).
- Males are drawn in blue (`ID <p_id> | MALE`).
- Unknown individuals are drawn in slate gray (`ID <p_id> | UNKNOWN`).

### C. Gated Snapshot Organization
Snapshots are recorded **strictly for qualifying women-safety incidents or verified emergency SOS events**:
- Organized into dated folders:
  ```text
  outputs/incidents/YYYY-MM-DD/
  ├── INCIDENT_YYYYMMDD_HHMMSS_001.jpg
  ├── INCIDENT_YYYYMMDD_HHMMSS_001.json
  └── incidents_history.json
  ```
- **Snapshots are NOT saved for**:
  - Ordinary walking, standing, or calm gathering.
  - A woman simply being in the vicinity of men without aggression.
  - Male-only fights (status: `FIGHT SUSPECTED` on console/HUD, no woman snapshot).
  - Violence where female involvement is unconfirmed (`VIOLENCE DETECTED — WOMAN INVOLVEMENT UNCONFIRMED`).
  - Routine non-incident surveillance frames.
- **Duplicate Prevention**: A 30-second cooldown (`QUALIFYING_INCIDENT_COOLDOWN_SECONDS`) prevents saving duplicate snapshots for the same continuous incident.

---

## 5. Verification & Automated Test Commands

Run the comprehensive unit test suite:

```cmd
python -m unittest tests/test_basic.py
```

### Test Coverage (12 Passing Tests):
1. `test_01_config_and_safety_defaults`: Validates safety gates (`TEST_MODE=True`, `ALERTS_ENABLED=False`) and confirms restricted area is completely removed.
2. `test_02_person_detection_and_tracking`: Tests YOLOv8n + ByteTrack tracking.
3. `test_03_gender_detection_and_counter_equality`: Tests `working_m_&_F.pt` and verifies $\text{TOTAL} = \text{MALE} + \text{FEMALE} + \text{UNKNOWN}$.
4. `test_04_violence_cnn_gru_model_loading`: Tests strict load of `best.pt` into `ResNet18GRU` and forward pass on 8 frames.
5. `test_05_scenario_a_female_victim_highlighted`: Confirms female victim attribution, red box, and qualifying snapshot.
6. `test_06_scenario_b_male_only_fight`: Confirms `victim_female_ids == []`, status `FIGHT SUSPECTED`, and NO woman snapshot.
7. `test_07_scenario_c_woman_normal`: Confirms normal female presence remains `WOMAN STATUS: SECURE (SAFE)` with no snapshot.
8. `test_08_scenario_d_violence_unconfirmed_female`: Confirms distant female is not falsely flagged with a red box and no snapshot is taken.
9. `test_09_fall_and_universal_sos`: Confirms female fall attribution and universal SOS dispatching regardless of gender.
10. `test_10_incident_storage_organization_and_naming`: Confirms saving to `outputs/incidents/YYYY-MM-DD/` with `INCIDENT_...` format and non-qualifying suppression.
11. `test_11_police_station_locator`: Tests OpenStreetMap distance calculation and emergency hotline directory (112, 1091).
12. `test_12_evaluation_metrics_computation`: Tests accuracy, precision, recall, and F1 calculations.

### Single Image & Video Tests:
```cmd
python test_image.py
python test_video.py
```
Outputs are saved in `outputs/incidents/test_image_result.jpg` and `outputs/incidents/annotated_video.mp4`.

---

## 6. Performance Benchmarks & Optimizations

Empirical benchmark measured on 45 continuous frames (CPU 2-threads):

| Metric | Baseline (Pre-Optimization) | Optimized Implementation | Improvement |
|:---|:---:|:---:|:---:|
| **Total Processing Time (45 Frames)** | **55.37 s** | **40.89 s** | **26.2% Faster** |
| **Average Processing Latency** | **1230.5 ms / frame** | **908.6 ms / frame** | **-321.9 ms / frame** |
| **Unit Test Suite Execution (`test_basic.py`)** | **22.71 s** | **4.40 s** | **5.16x Faster** |
| **Webcam / Stream Pipeline Lag** | Queue buildup (~2-5s lag) | **0.0s (Latest-frame buffer)** | **Lag Eliminated** |
| **Buffer RAM Usage (32 Frames)** | ~200 MB (uncompressed HD) | **~2.4 MB (pre-resized RGB)** | **98.8% RAM reduction** |

### Key Optimization Techniques Applied:
1. **Threaded Latest-Frame Video Capture (`VideoStream`)**: Background grabber thread with `cv2.CAP_PROP_BUFFERSIZE = 1` continuously pulls the newest frame, completely preventing frame queue lag.
2. **Inference Interval Decimation**:
   - Person detection & ByteTrack tracking: runs every frame (`PERSON_DETECTION_INTERVAL = 1`).
   - YOLO Gender classifier (`working_m_&_F.pt`): runs every 4 frames (`GENDER_INFERENCE_INTERVAL = 4`) with fast cache returns for stable tracks.
   - YOLOv8-pose skeleton: runs every 3 frames (`POSE_INFERENCE_INTERVAL = 3`) with pose cache reuse.
   - Carried object detection (snatching): cached over 8 frames (`CARRIED_OBJECTS_INTERVAL = 8`), removing 7 of 8 redundant YOLO forward passes.
   - Violence sequence classifier (`best.pt` ResNet18-GRU): evaluates 8-frame rolling buffer at `VIOLENCE_CHECK_INTERVAL_SEC = 0.4s`.
3. **Pre-resized Frame Buffer**: Pre-resizes frames to `160x160` RGB upon buffer insertion, eliminating redundant resizing and cutting buffer memory by ~98%.
4. **PyTorch Inference Mode**: Replaced `torch.no_grad()` with `torch.inference_mode()` for C++ acceleration.
5. **Real-Time HUD Performance Pill**: Displays live FPS, latency (ms), and hardware device (`[CPU]` or `[CUDA]`).