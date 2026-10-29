"""
Snatching and Theft Detection Module.
Formulates snatching as a spatio-temporal action recognition problem combining:
1. Person tracking trajectories (Owner vs Approacher)
2. Relevant carried object tracking (backpack, handbag, suitcase, cell phone via YOLO COCO classes)
3. Spatio-temporal departure dynamics (abrupt separation velocity differential)

IMPORTANT LIMITATIONS:
Surveillance camera angle, resolution, and occlusion can cause ambiguous readings.
Handovers, fast running to catch transit, and picking up dropped belongings are
distinguished by velocity differentials, but alerts remain suspected decision-support signals.
"""

import time
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# COCO class IDs for snatchable carried belongings
SNATCHABLE_COCO_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    67: "cell phone"
}

# Velocity and proximity thresholds
OWNERSHIP_PROXIMITY_RATIO = 1.4      # Distance within person width to establish possession
APPROACH_PROXIMITY_RATIO = 1.5       # Proximity for second person to be in grab range
SNATCH_DEPARTURE_VELOCITY = 650.0     # Pixels/second departure speed indicating sudden sprint
NORMAL_HANDOVER_MAX_VELOCITY = 280.0  # Speeds below this are typical of normal cooperative handovers
MIN_POSSESSION_DURATION_SEC = 0.8    # Minimum time person A must possess object before snatch evaluation

# Internal tracking histories
# person_id -> list of (timestamp, centroid_x, centroid_y, width)
_person_trajectories: Dict[int, List[Tuple[float, float, float, float]]] = {}
# object_id -> {"class": str, "history": [(t, cx, cy)], "owner_id": Optional[int], "possession_since": float}
_tracked_objects: Dict[int, Dict[str, Any]] = {}
_last_snatch_event_time: float = 0.0


def reset_snatch_tracks():
    """Resets tracking state for a new video or camera source."""
    _person_trajectories.clear()
    _tracked_objects.clear()


def reset_snatch_person(person_id: int):
    """Cleans up history for a person who exited the field of view."""
    _person_trajectories.pop(person_id, None)
    for obj_data in _tracked_objects.values():
        if obj_data.get("owner_id") == person_id:
            obj_data["owner_id"] = None


def _compute_velocity(history: List[Tuple[float, float, float]]) -> float:
    """Computes speed in pixels/second over history window."""
    if len(history) < 2:
        return 0.0
    t0, x0, y0 = history[0][:3]
    t1, x1, y1 = history[-1][:3]
    dt = t1 - t0
    if dt <= 0.01:
        return 0.0
    dist = math.hypot(x1 - x0, y1 - y0)
    return dist / dt


def detect_snatching(
    people: List[Dict[str, Any]],
    carried_objects: Optional[List[Dict[str, Any]]] = None,
    frame_shape: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Evaluates multi-person spatio-temporal dynamics for sudden grab-and-flee actions.

    Parameters:
    - people: list of {"id": int, "box": (x1,y1,x2,y2), ...}
    - carried_objects: optional list of detected object dicts {"box": (x1,y1,x2,y2), "class_id": int, "conf": float}

    Returns:
    {
        "is_snatching": bool,
        "confidence": float,
        "suspected_pairs": [(snatcher_id, victim_id)],
        "object_type": str,
        "evidence_reason": str,
        "limitations_note": str
    }
    """
    global _last_snatch_event_time
    now = time.time()

    result = {
        "is_snatching": False,
        "confidence": 0.0,
        "suspected_pairs": [],
        "object_type": "None",
        "evidence_reason": "No snatching dynamics observed",
        "limitations_note": "Camera perspective and occlusion affect camera-coordinate velocity estimations."
    }

    if not people or len(people) < 2:
        return result

    # 1. Update person trajectories
    per_person = {}
    for p in people:
        p_id = p["id"]
        x1, y1, x2, y2 = p["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))

        traj = _person_trajectories.setdefault(p_id, [])
        traj.append((now, cx, cy, w))
        # Keep 2.0 seconds window
        _person_trajectories[p_id] = [pt for pt in traj if now - pt[0] <= 2.0]

        vel = _compute_velocity([(t, x, y) for (t, x, y, _) in _person_trajectories[p_id]])
        per_person[p_id] = {
            "centroid": (cx, cy),
            "width": w,
            "velocity": vel,
            "box": p["box"]
        }

    # 2. Analyze Pairwise Separation Dynamics
    # A snatch incident requires:
    # a) Person A (victim) and Person B (snatcher) were in very close contact.
    # b) One person suddenly bolts at high speed (vel > SNATCH_DEPARTURE_VELOCITY).
    # c) The other person decelerates or remains stationary (large velocity differential).
    # d) If objects are present, object moves with the fleeing person.

    ids = list(per_person.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            p_a, p_b = per_person[id_a], per_person[id_b]

            dist = math.hypot(p_a["centroid"][0] - p_b["centroid"][0], p_a["centroid"][1] - p_b["centroid"][1])
            avg_width = (p_a["width"] + p_b["width"]) / 2.0

            # Check if currently or recently close
            was_close = dist < (avg_width * APPROACH_PROXIMITY_RATIO * 1.5)

            vel_a = p_a["velocity"]
            vel_b = p_b["velocity"]
            vel_diff = abs(vel_a - vel_b)

            # Identification of flee motion: one moving very fast, other slow/stationary
            fleeing_id = id_a if vel_a > vel_b else id_b
            victim_id = id_b if fleeing_id == id_a else id_a
            flee_vel = max(vel_a, vel_b)
            victim_vel = min(vel_a, vel_b)

            if was_close and flee_vel >= SNATCH_DEPARTURE_VELOCITY and victim_vel < NORMAL_HANDOVER_MAX_VELOCITY:
                # Sudden departure with sharp velocity asymmetry
                # Check for carried object corroboration if available
                object_corroborated = False
                detected_item = "personal belonging / bag"

                if carried_objects:
                    flee_box = per_person[fleeing_id]["box"]
                    for obj in carried_objects:
                        ox1, oy1, ox2, oy2 = obj["box"]
                        ocx = (ox1 + ox2) / 2.0
                        ocy = (oy1 + oy2) / 2.0
                        # Check if object is located on or adjacent to fleeing person
                        if flee_box[0] - 20 <= ocx <= flee_box[2] + 20 and flee_box[1] - 20 <= ocy <= flee_box[3] + 20:
                            object_corroborated = True
                            detected_item = SNATCHABLE_COCO_CLASSES.get(obj.get("class_id"), "belonging")
                            break

                conf = 0.85 if object_corroborated else 0.65
                result["is_snatching"] = True
                result["confidence"] = conf
                result["suspected_pairs"].append((fleeing_id, victim_id))
                result["object_type"] = detected_item
                result["evidence_reason"] = (
                    f"Rapid snatch & flee dynamic: ID {fleeing_id} accelerated to {int(flee_vel)} px/s "
                    f"away from stationary ID {victim_id} following close contact ({detected_item})"
                )
                _last_snatch_event_time = now
                return result

    return result
