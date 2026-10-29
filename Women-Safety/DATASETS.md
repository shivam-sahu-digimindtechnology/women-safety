# Dataset Strategy, Manifest & Model Evaluation Standards

This document establishes the official dataset manifest, research sources, licensing, train/val/test splitting standards, and quantitative evaluation metrics for the Women Safety AI system.

---

## 1. Official Dataset Manifest

| Dataset Name | Intended Task | Official URL / Source | Annotation Format | Licensing & Access | Status in System |
|---|---|---|---|---|---|
| **COCO 2017** | Person & Object Detection (Bags, Phones) | [cocodataset.org](https://cocodataset.org) | Bounding boxes (80 COCO classes) | Creative Commons Attribution 4.0 | Pretrained weights (`yolov8n.pt`) |
| **COCO Keypoints** | Pose Estimation & Keypoint Geometry | [cocodataset.org](https://cocodataset.org/#keypoints-2017) | 17 human keypoints (x, y, v) | Creative Commons Attribution 4.0 | Pretrained weights (`yolov8n-pose.pt`) |
| **RWF-2000** | Video Violence & Physical Fight Detection | [GitHub - RWF-2000](https://github.com/mivoi/RWF-2000) | 2,000 video clips (5s each), binary (Fight / Non-Fight) | Academic Research License | Baseline zero-shot CLIP (`ViT-B/32`) + temporal sliding window |
| **Hockey Fights Dataset** | Close-contact Physical Combat | [Kaggle / Hockey Fights](https://www.kaggle.com/datasets/yasserhessein/hockey-fight-vidoes) | 1,000 video clips (50 frames each), binary | Open Academic Use | Evaluation benchmark |
| **CCTV-Fights** | Real-world Surveillance Altercations | [GitHub - CCTV-Fights](https://github.com/cctv-fights) | 1,000 real CCTV videos with temporal boundaries | Academic Access on Request | Validation reference for surveillance angles |
| **UR Fall Detection (URFD)** | Human Fall & Posture Transitions | [UR Fall Detection](http://fenix.univ.orleans.fr/urfd/) | 70 sequences (30 falls, 40 ADL activities) with accelerometers | Open Research License | Geometric aspect ratio + state machine baseline |
| **Multiple Cameras Fall Dataset** | Multi-angle Falls vs Sitting / Lying | [Univ. of Rzeszów](http://www.iro.umontreal.ca/~labimage/Dataset/) | 192 video sequences across 8 camera angles | Academic Use | Tested against intentional sitting/lying down |
| **UCF-Crime (Theft / Robbery)** | Sudden Snatching & Robbery Actions | [UCF Center for Research in Computer Vision](https://www.crcv.ucf.edu/research/projects/real-world-anomaly-detection-in-surveillance-videos/) | 128 hours of surveillance video, temporal anomaly tags | Academic Research License | Spatial-temporal departure heuristic + object proximity |
| **HaGRID (Hand Gesture)** | Distress & SOS Hand Gestures | [GitHub - HaGRID](https://github.com/h-agrid/hagrid) | 552k images, 18 gesture classes | Apache 2.0 | Hand/Wrist elevation & MediaPipe distress signatures |

---

## 2. Train / Validation / Test Splitting Rules

To ensure scientific validity and eliminate optimistic bias:
1. **Zero Data Leakage Across Splits**:
   - Splitting must be performed at the **recording / sequence level**, NEVER randomly at the frame level.
   - Frames from the same camera clip or room setup must not appear in both training and test sets.
2. **Standard Split Ratios**:
   - **Training Set**: 70% of independent sequences.
   - **Validation Set**: 15% for threshold tuning and hyperparameter optimization.
   - **Test Set**: 15% held out for final reporting.
3. **Class Balance**:
   - Negative examples (Activities of Daily Living - ADL, normal walking, hugs, handshakes, sitting down, picking up dropped items) must equal or exceed positive incident examples to simulate realistic surveillance distributions.

---

## 3. Evaluation Metrics Standards

Every detection task is evaluated with task-appropriate quantitative metrics:

### A. Person Detection & Tracking
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.50.
- **MOTA (Multiple Object Tracking Accuracy)**: Measures missed detections, false positives, and ID switches.
- **IDF1**: ID F1-Score measuring trajectory continuity.
- **ID Switches per 1,000 frames**: Target < 5 switches.

### B. Fall & Violence Detection
- **Precision**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ (minimizing false alarms).
- **Recall (Sensitivity)**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ (ensuring real incidents are detected).
- **F1-Score**: Harmonic mean of Precision and Recall.
- **False Positives per Hour (FP/h)**: Primary metric for real-world surveillance deployment feasibility.
- **Detection Latency**: Time in seconds from event onset to system alert confirmation.

---

## 4. Reproducible Baseline vs Fine-Tuning Strategy

1. **Pretrained Baselines (Zero Download Friction)**:
   - Person & Objects: YOLOv8 Nano (`yolov8n.pt`, 6.5 MB).
   - Pose & Keypoints: YOLOv8 Pose Nano (`yolov8n-pose.pt`, 6.8 MB).
   - Violence: OpenAI CLIP (`ViT-B/32`) zero-shot scene matching with temporal multi-frame confirmation.
   - Gender: Hugging Face ViT (`rizvandwiki/gender-classification`).
2. **Fine-Tuning Roadmap (`training/`)**:
   - Provided reproducible script `training/train_violence_head.py` for training a lightweight temporal classifier on sequence features without requiring large GPU clusters.
