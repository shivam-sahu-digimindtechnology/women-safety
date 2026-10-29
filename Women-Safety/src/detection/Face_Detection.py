"""
Face Detection Module.
Uses MediaPipe Face Detection to locate human faces independently of full body crops.
Configured via config.py.
"""

from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp

import config

_face_detector: Optional[mp.solutions.face_detection.FaceDetection] = None


def get_face_detector() -> Optional[mp.solutions.face_detection.FaceDetection]:
    """Initializes and caches the MediaPipe FaceDetection instance."""
    global _face_detector
    if not config.FACE_DETECTION_ENABLED:
        return None

    if _face_detector is None:
        try:
            mp_face = mp.solutions.face_detection
            _face_detector = mp_face.FaceDetection(
                model_selection=config.FACE_MODEL_SELECTION,
                min_detection_confidence=config.FACE_MIN_CONFIDENCE
            )
        except Exception as e:
            print(f"[Face Detection Error] Failed to initialize MediaPipe FaceDetection: {e}")
            _face_detector = None

    return _face_detector


def detect_faces(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detects faces in an image or frame.
    Returns a list of dicts:
    [
        {
            "box": (x1, y1, x2, y2),
            "conf": float,
            "crop": np.ndarray
        },
        ...
    ]
    """
    if frame is None or frame.size == 0 or not config.FACE_DETECTION_ENABLED:
        return []

    detector = get_face_detector()
    if detector is None:
        return []

    h, w = frame.shape[:2]
    if h < config.FACE_MIN_SIZE_PX or w < config.FACE_MIN_SIZE_PX:
        return []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = detector.process(rgb)
    except Exception as e:
        print(f"[Face Detection Warning] Face detection process error: {e}")
        return []

    faces = []
    if results and results.detections:
        for detection in results.detections:
            conf = float(detection.score[0]) if detection.score else 0.0
            rbb = detection.location_data.relative_bounding_box
            x1 = max(0, int(rbb.xmin * w))
            y1 = max(0, int(rbb.ymin * h))
            x2 = min(w, int((rbb.xmin + rbb.width) * w))
            y2 = min(h, int((rbb.ymin + rbb.height) * h))

            fw, fh = x2 - x1, y2 - y1
            if fw < config.FACE_MIN_SIZE_PX or fh < config.FACE_MIN_SIZE_PX:
                continue

            crop = frame[y1:y2, x1:x2]
            faces.append({
                "box": (x1, y1, x2, y2),
                "conf": round(conf, 2),
                "crop": crop
            })

    return faces


def find_face_in_person_crop(person_crop: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Given a person bounding box crop, searches for a detectable, clear face within the upper portion.
    Returns face dict with face crop and coords relative to the person crop, or None if not found/unsuitable.
    """
    if person_crop is None or person_crop.size == 0:
        return None

    h, w = person_crop.shape[:2]
    # Faces are typically in the top 50% of an upright person's body
    upper_h = max(int(h * 0.55), config.FACE_MIN_SIZE_PX)
    upper_crop = person_crop[0:min(upper_h, h), :]

    faces = detect_faces(upper_crop)
    if not faces:
        # Also try full person crop as fallback if person is sitting/reclined
        faces = detect_faces(person_crop)

    if faces:
        # Return face with highest confidence
        faces.sort(key=lambda f: f["conf"], reverse=True)
        return faces[0]

    return None
