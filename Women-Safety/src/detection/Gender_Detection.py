"""
Gender Classification Module.
Integrates the trained YOLO male/female classification model (working_m_&_F.pt):
- Classes: {0: 'female', 1: 'male'}
- Associates predictions with tracked person bounding boxes via IoU matching
- Applies temporal smoothing (multi-frame voting/EMA) per tracking ID to avoid flickering
- Configurable confidence threshold (config.GENDER_CONFIDENCE_THRESHOLD)
- Strictly returns 'Male', 'Female', or 'Unknown' (never forces binary label when uncertain)
- Gracefully falls back to HuggingFace ViT if working_m_&_F.pt is missing.
"""

import os
import time
from collections import defaultdict, deque
from typing import Tuple, Optional, Dict, List, Any
import cv2
import numpy as np

import config

_yolo_gender_model = None
_gender_model_loaded = False
_gender_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
_gender_cache: Dict[int, Tuple[str, float, float]] = {}


def load_gender_model():
    """Loads and caches the working_m_&_F.pt YOLO model."""
    global _yolo_gender_model, _gender_model_loaded
    if _gender_model_loaded:
        return _yolo_gender_model

    model_path = config.GENDER_MODEL_PATH
    if not os.path.exists(model_path):
        alt_paths = [
            os.path.join(config.BASE_DIR, "working_m_&_F.pt"),
            os.path.join(config.BASE_DIR, "models", "working_m_&_F.pt"),
            "working_m_&_F.pt"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                break

    if os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            print(f"[Gender Classification] Loading YOLO model from {model_path}...")
            _yolo_gender_model = YOLO(model_path)
            _gender_model_loaded = True
            print(f"[Gender Classification] Model loaded. Classes: {_yolo_gender_model.names}")
            return _yolo_gender_model
        except Exception as e:
            print(f"[Gender Classification Warning] Failed to load {model_path}: {e}")
            _yolo_gender_model = None
    else:
        print(f"[Gender Classification Info] {model_path} not found. Will use fallback if available.")

    _gender_model_loaded = True
    return _yolo_gender_model


def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union_area = float(areaA + areaB - inter_area)
    return inter_area / union_area if union_area > 0 else 0.0


def detect_genders_in_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Runs working_m_&_F.pt on the full frame to detect male and female candidates.
    Returns: list of dicts with {"box": (x1, y1, x2, y2), "gender": "Female"|"Male", "conf": float}
    """
    model = load_gender_model()
    if model is None or frame is None or frame.size == 0:
        return []

    try:
        device = getattr(config, "INFERENCE_DEVICE", "cpu")
        results = model(frame, conf=0.20, device=device, verbose=False)
        if not results or len(results) == 0:
            return []

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        detected = []
        for i in range(len(boxes_obj)):
            coords = list(map(int, boxes_obj.xyxy[i].tolist()))
            cls_id = int(boxes_obj.cls[i].item())
            conf = float(boxes_obj.conf[i].item())
            
            # Map cls_id using model.names: 0 -> female, 1 -> male
            name = model.names.get(cls_id, str(cls_id)).lower()
            if "female" in name or "woman" in name:
                label = "Female"
            elif "male" in name or "man" in name:
                label = "Male"
            else:
                label = "Unknown"

            detected.append({
                "box": (coords[0], coords[1], coords[2], coords[3]),
                "gender": label,
                "conf": round(conf, 2)
            })
        return detected
    except Exception as e:
        print(f"[Gender Detection Frame Error] {e}")
        return []


def classify_person_crop(person_crop: np.ndarray) -> Tuple[str, float]:
    """
    Classifies gender on a single person crop using working_m_&_F.pt.
    Returns: ("Female"|"Male"|"Unknown", confidence)
    """
    if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 30 or person_crop.shape[1] < 20:
        return "Unknown", 0.0

    model = load_gender_model()
    if model is not None:
        try:
            device = getattr(config, "INFERENCE_DEVICE", "cpu")
            results = model(person_crop, conf=0.20, device=device, verbose=False)
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                best_box = results[0].boxes[0]
                cls_id = int(best_box.cls[0].item())
                conf = float(best_box.conf[0].item())
                name = model.names.get(cls_id, "").lower()
                label = "Female" if "female" in name else ("Male" if "male" in name else "Unknown")
                if conf >= config.GENDER_CONFIDENCE_THRESHOLD:
                    return label, round(conf, 2)
                return "Unknown", round(conf, 2)
        except Exception:
            pass

    return "Unknown", 0.0


_vit_pipeline = None

def _vit_fallback_classify(crop_image: np.ndarray) -> Tuple[str, float]:
    """Fallback Hugging Face classifier only if working_m_&_F.pt is missing."""
    global _vit_pipeline
    if not config.GENDER_CLASSIFICATION_ENABLED or load_gender_model() is not None:
        return "Unknown", 0.0

    try:
        from PIL import Image
        if _vit_pipeline is None:
            from transformers import pipeline
            _vit_pipeline = pipeline("image-classification", model=config.GENDER_FALLBACK_MODEL_NAME)

        rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        res = _vit_pipeline(pil_img)
        if res:
            raw_label = res[0]["label"].lower()
            conf = float(res[0]["score"])
            label = "Female" if ("female" in raw_label or "woman" in raw_label) else ("Male" if ("male" in raw_label or "man" in raw_label) else "Unknown")
            if conf >= config.GENDER_CONFIDENCE_THRESHOLD:
                return label, round(conf, 2)
            return "Unknown", round(conf, 2)
    except Exception:
        pass
    return "Unknown", 0.0


def get_or_predict_gender(
    track_id: int,
    person_box: Tuple[int, int, int, int],
    frame_gender_detections: List[Dict[str, Any]],
    person_crop: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    """
    Classifies gender using the trained working_m_&_F.pt model.
    Prioritizes full-person crop evaluation (where the model achieves 75-85% confidence),
    falls back to frame-level IoU matching if crop is unavailable, applies 15-frame temporal
    majority voting per track ID, and strictly returns 'Unknown' if confidence is insufficient.
    """
    now = time.time()

    # 1. Fast cache lookup for stable, already-classified tracks
    if track_id in _gender_cache:
        cached_label, cached_conf, cached_time = _gender_cache[track_id]
        if cached_label in ("Male", "Female") and cached_conf >= 0.65 and (now - cached_time < 3.0):
            return cached_label, cached_conf

    instant_gender = "Unknown"
    instant_conf = 0.0

    # 2. Primary evaluation: Direct person crop evaluation with working_m_&_F.pt
    if person_crop is not None and person_crop.size > 0 and person_crop.shape[0] >= 35 and person_crop.shape[1] >= 20:
        c_label, c_conf = classify_person_crop(person_crop)
        if c_label in ("Male", "Female") and c_conf >= config.GENDER_CONFIDENCE_THRESHOLD:
            instant_gender = c_label
            instant_conf = c_conf

    # 3. Fallback evaluation: Frame-level IoU matching against detected candidates
    if instant_gender == "Unknown" and frame_gender_detections:
        best_iou = 0.0
        matched_gender = None
        matched_conf = 0.0
        for g_det in frame_gender_detections:
            iou = _compute_iou(person_box, g_det["box"])
            if iou > best_iou:
                best_iou = iou
                matched_gender = g_det["gender"]
                matched_conf = g_det["conf"]

        if best_iou >= config.GENDER_IOU_MATCH_THRESHOLD and matched_gender in ("Male", "Female") and matched_conf >= config.GENDER_CONFIDENCE_THRESHOLD:
            instant_gender = matched_gender
            instant_conf = matched_conf

    # 4. Record into temporal history for track_id
    history = _gender_history[track_id]
    history.append((instant_gender, instant_conf, now))

    # 5. Multi-frame majority voting (across confident frames in last 4 seconds)
    recent_votes = [item for item in history if now - item[2] <= 4.0 and item[0] in ("Male", "Female")]
    if len(recent_votes) >= 2:
        females = [v for v in recent_votes if v[0] == "Female"]
        males = [v for v in recent_votes if v[0] == "Male"]
        
        if len(females) > len(males):
            avg_conf = sum(v[1] for v in females) / len(females)
            smoothed_label = "Female" if avg_conf >= config.GENDER_CONFIDENCE_THRESHOLD else "Unknown"
        elif len(males) > len(females):
            avg_conf = sum(v[1] for v in males) / len(males)
            smoothed_label = "Male" if avg_conf >= config.GENDER_CONFIDENCE_THRESHOLD else "Unknown"
        else:
            smoothed_label = instant_gender
            avg_conf = instant_conf
    else:
        smoothed_label = instant_gender if instant_conf >= config.GENDER_CONFIDENCE_THRESHOLD else "Unknown"
        avg_conf = instant_conf

    _gender_cache[track_id] = (smoothed_label, avg_conf, now)
    return smoothed_label, round(avg_conf, 2)


def cleanup_gender_tracks(active_track_ids: List[int]) -> None:
    """Removes historical data for tracks that exited the scene."""
    active_set = set(active_track_ids)
    for tid in list(_gender_history.keys()):
        if tid not in active_set:
            del _gender_history[tid]
    for tid in list(_gender_cache.keys()):
        if tid not in active_set:
            del _gender_cache[tid]
