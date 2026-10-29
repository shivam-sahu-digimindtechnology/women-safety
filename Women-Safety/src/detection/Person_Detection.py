"""
Person Detection and Tracking Module.
Uses Ultralytics YOLOv8 with ByteTrack tracking.
Configured via config.py.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO

import config

_yolo_model: Optional[YOLO] = None


def load_person_model() -> Optional[YOLO]:
    """Loads and caches the YOLO person detection model."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    model_path = config.PERSON_MODEL_PATH
    if not os.path.exists(model_path):
        # Check fallback in models/ or root
        alt_paths = [
            os.path.join(config.BASE_DIR, "models", "yolov8n.pt"),
            os.path.join(config.BASE_DIR, "yolov8n.pt"),
            "yolov8n.pt"
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                found = True
                break
        if not found:
            print(f"[Person Detection] Model weights not found at {config.PERSON_MODEL_PATH}. YOLO will attempt download.")
            model_path = "yolov8n.pt"

    try:
        _yolo_model = YOLO(model_path)
        return _yolo_model
    except Exception as e:
        print(f"[Person Detection Error] Failed to load YOLO model: {e}")
        return None


def detect_person(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Backward-compatible person detection function.
    Returns list of (x1, y1, x2, y2) bounding box tuples.
    """
    if frame is None or frame.size == 0:
        return []

    model = load_person_model()
    if model is None:
        return []

    results = model(
        frame,
        conf=config.PERSON_CONFIDENCE,
        iou=config.PERSON_IOU_THRESHOLD,
        classes=[0],  # 0 is 'person'
        verbose=False
    )

    person_boxes: List[Tuple[int, int, int, int]] = []
    if results and len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_boxes.append((x1, y1, x2, y2))

    return person_boxes


def detect_and_track_persons(frame: np.ndarray, persist: bool = True) -> List[Dict[str, Any]]:
    """
    Detects and tracks persons across frames.
    Returns a list of dicts:
    [
        {
            "id": int,
            "box": (x1, y1, x2, y2),
            "conf": float,
            "label": "Person"
        },
        ...
    ]
    """
    if frame is None or frame.size == 0 or not config.PERSON_DETECTION_ENABLED:
        return []

    model = load_person_model()
    if model is None:
        return []

    try:
        device = getattr(config, "INFERENCE_DEVICE", "cpu")
        if config.TRACKING_ENABLED:
            results = model.track(
                frame,
                persist=persist,
                conf=config.PERSON_CONFIDENCE,
                iou=config.PERSON_IOU_THRESHOLD,
                classes=[0],
                tracker="bytetrack.yaml",
                device=device,
                verbose=False
            )
        else:
            results = model(
                frame,
                conf=config.PERSON_CONFIDENCE,
                iou=config.PERSON_IOU_THRESHOLD,
                classes=[0],
                device=device,
                verbose=False
            )
    except Exception as e:
        # Fallback to standard detect if tracking fails
        device = getattr(config, "INFERENCE_DEVICE", "cpu")
        print(f"[Person Tracking Warning] Tracking failed: {e}. Falling back to standard detection.")
        results = model(
            frame,
            conf=config.PERSON_CONFIDENCE,
            iou=config.PERSON_IOU_THRESHOLD,
            classes=[0],
            device=device,
            verbose=False
        )

    detections: List[Dict[str, Any]] = []
    if not results or len(results) == 0:
        return detections

    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return detections

    for i in range(len(boxes_obj)):
        box_coords = boxes_obj.xyxy[i].tolist()
        x1, y1, x2, y2 = map(int, box_coords)
        conf = float(boxes_obj.conf[i].item())

        # Extract tracking ID if available
        track_id = i + 1
        if boxes_obj.id is not None:
            track_id = int(boxes_obj.id[i].item())

        detections.append({
            "id": track_id,
            "box": (x1, y1, x2, y2),
            "conf": round(conf, 2),
            "label": "Person"
        })

    return detections


_cached_carried_objects: List[Dict[str, Any]] = []
_carried_objects_frame_counter: int = 0

def detect_carried_objects(frame: np.ndarray, classes: Optional[List[int]] = None, force: bool = False) -> List[Dict[str, Any]]:
    """
    Detects carried objects (backpack, handbag, suitcase, cell phone) for snatching analysis.
    Uses COCO class IDs: 24 (backpack), 26 (handbag), 28 (suitcase), 67 (cell phone).
    Caches detections over CARRIED_OBJECTS_INTERVAL frames to avoid redundant YOLO forward passes.
    """
    global _cached_carried_objects, _carried_objects_frame_counter
    if frame is None or frame.size == 0:
        return []

    interval = getattr(config, "CARRIED_OBJECTS_INTERVAL", 8)
    _carried_objects_frame_counter += 1
    if not force and (_carried_objects_frame_counter % interval != 0) and _cached_carried_objects:
        return _cached_carried_objects

    if classes is None:
        classes = [24, 26, 28, 67]

    model = load_person_model()
    if model is None:
        return []

    device = getattr(config, "INFERENCE_DEVICE", "cpu")
    try:
        results = model(
            frame,
            conf=0.25,
            iou=0.45,
            classes=classes,
            device=device,
            verbose=False
        )
    except Exception:
        return _cached_carried_objects

    objects: List[Dict[str, Any]] = []
    if not results or len(results) == 0:
        _cached_carried_objects = objects
        return objects

    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        _cached_carried_objects = objects
        return objects

    for i in range(len(boxes_obj)):
        box_coords = boxes_obj.xyxy[i].tolist()
        x1, y1, x2, y2 = map(int, box_coords)
        conf = float(boxes_obj.conf[i].item())
        cls_id = int(boxes_obj.cls[i].item())
        cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") else str(cls_id)

        objects.append({
            "box": (x1, y1, x2, y2),
            "class_id": cls_id,
            "class_name": cls_name,
            "conf": round(conf, 2)
        })

    _cached_carried_objects = objects
    return objects

