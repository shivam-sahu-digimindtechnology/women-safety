"""
Interaction and Behavioral Analysis Module.
Provides explainable heuristics for:
- People Proximity
- Crowd Density
- Loitering Duration
- Following-like Movement Patterns
- Restricted-Zone Entry
- Fighting / Snatching Motion Heuristics

All metrics are camera-coordinate approximations and serve as decision-support signals,
not conclusive proof of malicious intent. Gender is never used to determine threat.
"""

import time
import math
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

import config

# COCO-17 keypoint indices
LEFT_WRIST, RIGHT_WRIST = 9, 10

HISTORY_WINDOW_SECONDS = 3.0

# Fighting / snatching parameters
PROXIMITY_RATIO = config.PROXIMITY_RATIO_THRESHOLD
FIGHT_WRIST_VELOCITY_THRESHOLD = 900
FIGHT_CONFIRM_SECONDS = 1.5
SEPARATION_VELOCITY_THRESHOLD = 700
SNATCH_CONFIRM_SECONDS = 0.5

# Tracking histories: ID -> list of entries
_wrist_history: Dict[int, List[Tuple[float, Tuple[float, float], Tuple[float, float]]]] = {}
_centroid_history: Dict[int, List[Tuple[float, Tuple[float, float], float]]] = {}
_first_seen_time: Dict[int, float] = {}
_loiter_anchor: Dict[int, Tuple[float, float, float]] = {}  # ID -> (x, y, start_time)

# Pair states
_pair_close_since: Dict[frozenset, float] = {}
_fight_since: Dict[frozenset, float] = {}
_snatch_since: Dict[frozenset, float] = {}
_following_since: Dict[Tuple[int, int], float] = {}


def reset_track(object_id: int):
    """Clear interaction and behavioral history for an ID that left the frame."""
    _wrist_history.pop(object_id, None)
    _centroid_history.pop(object_id, None)
    _first_seen_time.pop(object_id, None)
    _loiter_anchor.pop(object_id, None)

    for store in (_pair_close_since, _fight_since, _snatch_since):
        for key in [k for k in list(store.keys()) if object_id in k]:
            store.pop(key, None)

    for key in [k for k in list(_following_since.keys()) if object_id in k]:
        _following_since.pop(key, None)


def _velocity(history: List[Tuple[float, Tuple[float, float]]]) -> float:
    """Average speed in pixels/second between start and end of history window."""
    if len(history) < 2:
        return 0.0
    t0, p0 = history[0]
    t1, p1 = history[-1]
    dt = t1 - t0
    if dt <= 0:
        return 0.0
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt


def _direction_vector(history: List[Tuple[float, Tuple[float, float]]]) -> Tuple[float, float]:
    """Computes displacement vector (dx, dy) over history."""
    if len(history) < 2:
        return (0.0, 0.0)
    p0 = history[0][1]
    p1 = history[-1][1]
    return (p1[0] - p0[0], p1[1] - p0[1])


def _update_history(object_id: int, box: Tuple[int, int, int, int], keypoints: Optional[List[List[float]]]):
    now = time.time()
    x1, y1, x2, y2 = box
    centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    width = float(max(1, x2 - x1))

    if object_id not in _first_seen_time:
        _first_seen_time[object_id] = now
        _loiter_anchor[object_id] = (centroid[0], centroid[1], now)

    ch = _centroid_history.setdefault(object_id, [])
    ch.append((now, centroid, width))
    _centroid_history[object_id] = [h for h in ch if now - h[0] <= HISTORY_WINDOW_SECONDS]

    wrist_velocity = 0.0
    if keypoints and len(keypoints) > RIGHT_WRIST:
        lw = (keypoints[LEFT_WRIST][0], keypoints[LEFT_WRIST][1])
        rw = (keypoints[RIGHT_WRIST][0], keypoints[RIGHT_WRIST][1])
        wh = _wrist_history.setdefault(object_id, [])
        wh.append((now, lw, rw))
        _wrist_history[object_id] = [h for h in wh if now - h[0] <= 1.0]

        left_pts = [(t, l) for t, l, r in _wrist_history[object_id]]
        right_pts = [(t, r) for t, l, r in _wrist_history[object_id]]
        wrist_velocity = max(_velocity(left_pts), _velocity(right_pts))

    c_pts = [(t, c) for t, c, w in _centroid_history[object_id]]
    centroid_velocity = _velocity(c_pts)
    disp_vec = _direction_vector(c_pts)

    return centroid, width, wrist_velocity, centroid_velocity, disp_vec


def detect_interactions(people: List[Dict[str, Any]], frame_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Analyzes multiple tracked people in a scene.
    people: list of {"id": objectID, "box": (x1,y1,x2,y2), "keypoints": [...] (optional)}.
    Returns:
    {
        "fighting_pairs": [(id1, id2), ...],
        "snatching_pairs": [(id1, id2), ...],
        "following_pairs": [(follower_id, leader_id), ...],
        "loitering_ids": [id1, ...],
        "crowd_count": int,
        "is_crowd_dense": bool,
        "pair_scores": {frozenset: {"proximity_score": float, ...}}
    }
    """
    now = time.time()
    results = {
        "fighting_pairs": [],
        "snatching_pairs": [],
        "following_pairs": [],
        "loitering_ids": [],
        "crowd_count": len(people),
        "is_crowd_dense": len(people) >= config.CROWD_DENSITY_THRESHOLD,
        "pair_scores": {}
    }

    if not config.INTERACTION_ANALYSIS_ENABLED or not people:
        return results

    per_person = {}
    for p in people:
        p_id = p["id"]
        box = p["box"]
        kps = p.get("keypoints")
        centroid, width, wrist_vel, centroid_vel, disp_vec = _update_history(p_id, box, kps)
        per_person[p_id] = {
            "centroid": centroid,
            "width": width,
            "wrist_velocity": wrist_vel,
            "centroid_velocity": centroid_vel,
            "disp_vec": disp_vec,
            "box": box
        }


        # Check loitering
        if config.LOITERING_ENABLED and p_id in _loiter_anchor:
            ax, ay, a_time = _loiter_anchor[p_id]
            dist_from_anchor = math.hypot(centroid[0] - ax, centroid[1] - ay)
            if dist_from_anchor > config.LOITERING_RADIUS_PX:
                # Reset anchor to current location
                _loiter_anchor[p_id] = (centroid[0], centroid[1], now)
            else:
                elapsed = now - a_time
                if elapsed >= config.LOITERING_TIME_THRESHOLD_SEC:
                    results["loitering_ids"].append(p_id)

    # Analyze pairwise interactions
    ids = list(per_person.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            pair_key = frozenset((id_a, id_b))
            a, b = per_person[id_a], per_person[id_b]

            avg_width = (a["width"] + b["width"]) / 2.0 or 1.0
            distance = math.hypot(a["centroid"][0] - b["centroid"][0], a["centroid"][1] - b["centroid"][1])

            proximity_score = (avg_width * PROXIMITY_RATIO) / (distance + 1e-6)
            is_close = proximity_score >= 1.0
            was_close = pair_key in _pair_close_since

            if is_close:
                _pair_close_since.setdefault(pair_key, now)
            else:
                _pair_close_since.pop(pair_key, None)

            # Fighting heuristic
            fight_score = max(a["wrist_velocity"], b["wrist_velocity"]) / FIGHT_WRIST_VELOCITY_THRESHOLD
            aggressive_motion = fight_score >= 1.0
            if is_close and aggressive_motion:
                _fight_since.setdefault(pair_key, now)
            else:
                _fight_since.pop(pair_key, None)

            if pair_key in _fight_since and (now - _fight_since[pair_key]) >= FIGHT_CONFIRM_SECONDS:
                results["fighting_pairs"].append((id_a, id_b))

            # Snatching heuristic
            departure_score = max(a["centroid_velocity"], b["centroid_velocity"]) / SEPARATION_VELOCITY_THRESHOLD
            sudden_separation = (not is_close) and was_close and departure_score >= 1.0
            if sudden_separation:
                _snatch_since.setdefault(pair_key, now)
            elif is_close:
                _snatch_since.pop(pair_key, None)

            if pair_key in _snatch_since and (now - _snatch_since[pair_key]) >= SNATCH_CONFIRM_SECONDS:
                results["snatching_pairs"].append((id_a, id_b))
                _snatch_since.pop(pair_key, None)

            # Following heuristic
            if config.FOLLOWING_ENABLED:
                # Sustained distance within FOLLOWING_DISTANCE_PX
                if distance <= config.FOLLOWING_DISTANCE_PX:
                    # Check if both are moving in similar direction
                    va, vb = a["disp_vec"], b["disp_vec"]
                    speed_a = math.hypot(va[0], va[1])
                    speed_b = math.hypot(vb[0], vb[1])
                    if speed_a > 15 and speed_b > 15:  # Moving noticeably
                        dot = (va[0] * vb[0] + va[1] * vb[1]) / (speed_a * speed_b + 1e-6)
                        if dot > 0.6:  # Heading in roughly same direction
                            following_key = (id_a, id_b)
                            _following_since.setdefault(following_key, now)
                            if now - _following_since[following_key] >= config.FOLLOWING_MIN_DURATION_SEC:
                                results["following_pairs"].append((id_a, id_b))
                        else:
                            _following_since.pop((id_a, id_b), None)
                    else:
                        _following_since.pop((id_a, id_b), None)
                else:
                    _following_since.pop((id_a, id_b), None)

            results["pair_scores"][pair_key] = {
                "proximity_score": round(proximity_score, 2),
                "fight_score": round(fight_score, 2),
                "distance_px": round(distance, 1)
            }

    return results
