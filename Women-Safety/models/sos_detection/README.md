# SOS & Distress Detection

## Implementation Overview
SOS and distress detection in this system operates via geometric and temporal analysis of COCO 17-keypoint skeletons extracted by the YOLOv8-pose model (`models/fall_detection/yolov8n-pose.pt`):

1. **Overhead Distress High-Waving**: Evaluates vertical elevation of both wrists (COCO keypoints 9 & 10) significantly above shoulders and head sustained over a temporal streak ($\ge 5$ frames).
2. **Defensive Shielding Posture**: Evaluates wrists drawn close to head/face defensively while compressed.
3. **Manual Trigger**: Supports immediate operator or victim manual panic keypress (`'S'` key shortcut or external trigger button).

Universal emergency handling: Valid SOS signals trigger emergency response regardless of whether the person is classified as male, female, or unknown.
