"""
Fall detection using YOLOv8-pose keypoints, ported from the temporal
state-machine approach in https://github.com/vjn6805/Fall-Detection-Model
(fall_detector.py / state.py), adapted to this project's simpler pipeline.

SCORING FORMULA
---------------
fall_score = 0.35 * posture_score      # current body orientation
           + 0.30 * transition_score   # upright -> horizontal transition evidence
           + 0.20 * persistence_score  # how long low posture has been held
           + 0.15 * motion_score       # post-fall stillness

Each sub-score is itself a normalized 0..1 value/threshold ratio computed
purely from body keypoints -- gender is never part of this calculation, so
detection applies identically to every tracked person.

STATE MACHINE
--------------
NORMAL -> POSSIBLE_FALL -> FALL_CONFIRMED -> POST_FALL_MONITORING -> RECOVERED -> NORMAL
       -> UNKNOWN_LOW_POSTURE (horizontal but no observed upright->horizontal transition,
          e.g. person entered frame already lying down -- not treated as a fall)

A person simply lying/sitting still does NOT trigger a fall on its own: the
transition_score requires having observed them upright beforehand, and the
fall must persist for MIN_FALL_DURATION_SECONDS before FALL_CONFIRMED.

This is a prototype heuristic based on visual signals only, not a trained or
validated model, and not a medical diagnosis tool.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os
from ultralytics import YOLO
import config

_pose_model = None

def get_pose_model():
    """Loads and caches the YOLOv8-pose model."""
    global _pose_model
    if not config.POSE_DETECTION_ENABLED:
        return None
    if _pose_model is not None:
        return _pose_model

    model_path = config.POSE_MODEL_PATH
    if not os.path.exists(model_path):
        alt_paths = [
            os.path.join(config.BASE_DIR, "models", "yolov8n-pose.pt"),
            os.path.join(config.BASE_DIR, "yolov8n-pose.pt"),
            "yolov8n-pose.pt"
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                found = True
                break
        if not found:
            print(f"[Pose Detection] Pose model not found at {config.POSE_MODEL_PATH}. Will attempt download.")
            model_path = "yolov8n-pose.pt"

    try:
        _pose_model = YOLO(model_path)
        return _pose_model
    except Exception as e:
        print(f"[Pose Detection Error] Failed to load pose model: {e}")
        return None

# COCO-17 keypoint indices used by YOLOv8-pose
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# --- Confidence / observability gates ---
MIN_POSE_CONFIDENCE = config.MIN_POSE_CONFIDENCE
MIN_TRACK_AGE_SECONDS = 1.0         # a track must exist this long before fall scoring begins
OBSERVABILITY_HIGH_THRESHOLD = 0.55 # mean keypoint confidence for HIGH observability
OBSERVABILITY_LOW_THRESHOLD = 0.30  # below this, observability is LOW
MIN_VISIBLE_KEYPOINTS = 5           # required (with HIGH conf) for HIGH observability
KEYPOINT_CONF_THRESHOLD = 0.3       # a keypoint below this confidence is treated as not visible

# --- Posture signal ---
UPRIGHT_TORSO_ANGLE = 30.0     # degrees from vertical; below this = upright
HORIZONTAL_TORSO_ANGLE = 50.0  # degrees from vertical; above this = horizontal/low
HORIZONTAL_ASPECT_RATIO = config.FALL_ASPECT_RATIO_THRESHOLD  # bbox width/height above this = wide/horizontal

# --- Velocity / transition signal ---
FALL_VELOCITY_THRESHOLD = 0.08         # downward speed (box-heights/sec) counted as rapid descent
TRANSITION_WINDOW_SECONDS = 2.5        # recent time scanned for an upright->horizontal transition
CONTROLLED_TRANSITION_SECONDS = 1.2    # slower than this = controlled sit/lie-down, score penalised
MIN_UPRIGHT_SECONDS_BEFORE_FALL = 0.75 # must have been upright for this much of the window to count

# --- State machine timing ---
POSSIBLE_FALL_SECONDS = 0.5             # sustained high score before entering POSSIBLE_FALL
MIN_FALL_DURATION_SECONDS = config.FALL_CONFIRMATION_SECONDS  # time in POSSIBLE_FALL before FALL_CONFIRMED
RECOVERY_UPRIGHT_SECONDS = 2.0          # upright time required to leave POST_FALL_MONITORING
UNKNOWN_LOW_POSTURE_UPRIGHT_SECONDS = 1.5

# --- Motion (post-fall stillness) signal ---
STILLNESS_THRESHOLD = 6.0  # px/frame movement; used to score "not moving" after a fall

FALL_SCORE_ENTER_THRESHOLD = config.FALL_SCORE_THRESHOLD  # fall_score >= this (+ upright history) -> possible fall
FALL_SCORE_EXIT_THRESHOLD = config.FALL_SCORE_THRESHOLD - 0.10   # fall_score below this while POSSIBLE_FALL -> back to NORMAL



class FallState(Enum):
    NORMAL = "NORMAL"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    FALL_CONFIRMED = "FALL_CONFIRMED"
    POST_FALL_MONITORING = "POST_FALL_MONITORING"
    RECOVERED = "RECOVERED"
    UNKNOWN_LOW_POSTURE = "UNKNOWN_LOW_POSTURE"


@dataclass
class _Snapshot:
    timestamp: float
    center: tuple
    height: float
    aspect_ratio: float
    torso_angle: Optional[float]
    pose_confidence: float
    observability: str
    position_delta: Optional[float]


@dataclass
class _Track:
    history: deque = field(default_factory=lambda: deque(maxlen=300))
    track_started_at: float = field(default_factory=time.time)
    state: FallState = FallState.NORMAL
    state_entered_at: float = field(default_factory=time.time)
    high_score_since: Optional[float] = None  # when the fall-score-crossing streak began
    recovery_start_time: Optional[float] = None


_tracks = {}  # objectID -> _Track


def reset_track(object_id):
    """Drop all history/state for a tracked person that has disappeared, so a
    later new detection reusing the ID doesn't inherit stale state."""
    _tracks.pop(object_id, None)


def detect_poses(frame):
    """Runs YOLOv8-pose once per frame. Returns a list of
    {"box": (x1,y1,x2,y2), "keypoints": [[x,y], ...17], "kp_conf": [c, ...17]}
    for each detected person."""
    if frame is None or frame.size == 0 or not config.POSE_DETECTION_ENABLED:
        return []

    model = get_pose_model()
    if model is None:
        return []

    try:
        device = getattr(config, "INFERENCE_DEVICE", "cpu")
        conf_thresh = getattr(config, "MIN_POSE_CONFIDENCE", 0.25)
        results = model(frame, conf=conf_thresh, device=device, verbose=False)
    except Exception as e:
        print(f"[Pose Detection Warning] Pose inference error: {e}")
        return []

    detections = []

    for result in results:
        if result.keypoints is None or result.boxes is None:
            continue
        boxes = result.boxes
        keypoints = result.keypoints.xy
        kp_conf = result.keypoints.conf
        for i in range(len(boxes)):
            if int(boxes.cls[i]) != 0:  # 'person' class
                continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            detections.append({
                "box": (x1, y1, x2, y2),
                "keypoints": keypoints[i].tolist(),
                "kp_conf": kp_conf[i].tolist() if kp_conf is not None else [1.0] * 17,
            })
    return detections


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _extract_pose_features(keypoints, kp_conf):
    """Mirrors the reference repo's state._extract_pose_features: computes mean
    pose confidence, visible keypoint count, torso angle, and an HIGH/MEDIUM/LOW
    observability rating that gates how much the state machine trusts this frame."""
    visible = [c > KEYPOINT_CONF_THRESHOLD for c in kp_conf]
    visible_count = sum(visible)
    confs = [c for c, v in zip(kp_conf, visible) if v]
    pose_confidence = sum(confs) / len(confs) if confs else 0.0

    if pose_confidence >= OBSERVABILITY_HIGH_THRESHOLD and visible_count >= MIN_VISIBLE_KEYPOINTS:
        observability = "HIGH"
    elif pose_confidence >= OBSERVABILITY_LOW_THRESHOLD and visible_count >= 3:
        observability = "MEDIUM"
    else:
        observability = "LOW"

    torso_angle = None
    if visible[LEFT_SHOULDER] and visible[RIGHT_SHOULDER] and visible[LEFT_HIP] and visible[RIGHT_HIP]:
        shoulder_mid = _midpoint(keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER])
        hip_mid = _midpoint(keypoints[LEFT_HIP], keypoints[RIGHT_HIP])
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        torso_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

    return pose_confidence, observability, torso_angle


# ----------------------------------------------------------------------
# Sub-scores (each a 0..1 value/threshold ratio)
# ----------------------------------------------------------------------

def _posture_score(snap: _Snapshot) -> float:
    """0.0 = clearly upright, 1.0 = clearly horizontal. Torso angle weighted
    more heavily than aspect ratio, which is a supporting signal only."""
    angle_score = 0.0
    if snap.torso_angle is not None:
        lo, hi = UPRIGHT_TORSO_ANGLE, HORIZONTAL_TORSO_ANGLE
        angle_score = max(0.0, min(1.0, (snap.torso_angle - lo) / (hi - lo + 1e-6)))

    ar_score = 0.0
    if snap.aspect_ratio > 0:
        ar_score = max(0.0, min(1.0, (snap.aspect_ratio - 0.8) / (HORIZONTAL_ASPECT_RATIO - 0.8 + 1e-6)))

    return 0.65 * angle_score + 0.35 * ar_score


def _upright_duration(window) -> float:
    """Real elapsed time (seconds) covered by upright samples in the window,
    computed from timestamps so it stays correct regardless of frame rate."""
    duration = 0.0
    for i in range(1, len(window)):
        prev, curr = window[i - 1], window[i]
        if prev.torso_angle is not None and prev.torso_angle < UPRIGHT_TORSO_ANGLE + 10:
            duration += curr.timestamp - prev.timestamp
    return duration


def _had_upright_history(window) -> bool:
    """Full-window gate used by the state machine (NORMAL/UNKNOWN_LOW_POSTURE
    transitions): was the person upright for a meaningful part of the window?"""
    return _upright_duration(window) >= MIN_UPRIGHT_SECONDS_BEFORE_FALL


def _had_some_upright_history(window) -> bool:
    """Lighter check used inside _transition_score's early-half: only needs a
    brief upright presence to count as "came from standing", since by
    definition the early half is already a smaller slice of the window."""
    return _upright_duration(window) >= MIN_UPRIGHT_SECONDS_BEFORE_FALL / 2


def _velocity_score(window) -> float:
    """Rapid downward movement score, normalized by box height so it's
    resolution/distance independent. Only the peak velocity in the window matters."""
    if len(window) < 3:
        return 0.0
    velocities = []
    for i in range(1, len(window)):
        prev, curr = window[i - 1], window[i]
        if curr.height == 0:
            continue
        dy = curr.center[1] - prev.center[1]
        dt = max(curr.timestamp - prev.timestamp, 1e-3)
        velocities.append((dy / curr.height) / dt)
    if not velocities:
        return 0.0
    return max(0.0, min(1.0, max(velocities) / FALL_VELOCITY_THRESHOLD))


def _transition_score(window) -> float:
    """Upright->horizontal transition score, penalised when the transition is
    slow/controlled (sitting/lying down deliberately) rather than a rapid,
    uncontrolled fall. Gated on having actually observed the person upright
    before the transition, so someone who enters frame already lying down
    never accumulates a transition score."""
    if len(window) < 4:
        return 0.0

    t_mid = (window[0].timestamp + window[-1].timestamp) / 2
    early = [s for s in window if s.timestamp <= t_mid]
    late = [s for s in window if s.timestamp > t_mid]
    early_angles = [s.torso_angle for s in early if s.torso_angle is not None]
    late_angles = [s.torso_angle for s in late if s.torso_angle is not None]
    if not early_angles or not late_angles:
        return 0.0

    if not _had_some_upright_history(early):
        return 0.0

    delta = max(late_angles) - min(early_angles)
    angle_score = max(0.0, min(1.0, delta / 40.0))
    vel_score = _velocity_score(window)
    raw_score = max(angle_score, vel_score)
    if raw_score < 0.1:
        return 0.0

    transition_start = transition_end = None
    for s in window:
        if s.torso_angle is not None:
            if s.torso_angle < HORIZONTAL_TORSO_ANGLE and transition_start is None:
                transition_start = s.timestamp
            if s.torso_angle >= HORIZONTAL_TORSO_ANGLE:
                transition_end = s.timestamp

    if transition_start is not None and transition_end is not None:
        duration = max(transition_end - transition_start, 1e-3)
        speed_factor = max(0.1, min(1.0, CONTROLLED_TRANSITION_SECONDS / duration))
        raw_score *= speed_factor

    return raw_score


def _persistence_score(window) -> float:
    """Fraction of the recent window showing horizontal/low posture. Builds
    slowly, so a brief bend scores low while prolonged low posture scores high."""
    if not window:
        return 0.0
    low_count = sum(1 for s in window if s.torso_angle is not None and s.torso_angle > HORIZONTAL_TORSO_ANGLE)
    return low_count / len(window)


def _motion_score(window) -> float:
    """Post-event stillness score: high = not moving (consistent with a real
    collapse), low = still moving (consistent with intentional activity)."""
    now = window[-1].timestamp if window else 0.0
    recent = [s for s in window if now - s.timestamp <= 1.0]
    if len(recent) < 3:
        return 0.0
    deltas = [s.position_delta for s in recent if s.position_delta is not None]
    if not deltas:
        return 0.0
    avg_delta = sum(deltas) / len(deltas)
    return max(0.0, min(1.0, 1.0 - avg_delta / (STILLNESS_THRESHOLD * 2)))


def _is_currently_low(snap: _Snapshot) -> bool:
    return (snap.torso_angle is not None and snap.torso_angle > HORIZONTAL_TORSO_ANGLE) \
        or snap.aspect_ratio > HORIZONTAL_ASPECT_RATIO


def _is_currently_upright(snap: _Snapshot) -> bool:
    return snap.torso_angle is not None and snap.torso_angle < UPRIGHT_TORSO_ANGLE


# ----------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------

def _advance_state(track: _Track, snap: _Snapshot, fall_score: float, window):
    now = time.time()

    if track.state == FallState.NORMAL:
        if _is_currently_low(snap):
            if _had_upright_history(window) and fall_score >= FALL_SCORE_ENTER_THRESHOLD:
                if track.high_score_since is None:
                    track.high_score_since = now
                elif (now - track.high_score_since) >= POSSIBLE_FALL_SECONDS:
                    track.state, track.state_entered_at = FallState.POSSIBLE_FALL, now
            elif not _had_upright_history(window):
                track.state, track.state_entered_at = FallState.UNKNOWN_LOW_POSTURE, now
                track.high_score_since = None
            else:
                track.high_score_since = None
        else:
            track.high_score_since = None

    elif track.state == FallState.POSSIBLE_FALL:
        if fall_score < FALL_SCORE_EXIT_THRESHOLD:
            track.state, track.state_entered_at = FallState.NORMAL, now
            track.high_score_since = None
        elif snap.observability == "LOW":
            pass  # can't confirm with poor visibility -- hold in POSSIBLE_FALL
        elif (now - track.state_entered_at) >= MIN_FALL_DURATION_SECONDS:
            track.state, track.state_entered_at = FallState.FALL_CONFIRMED, now

    elif track.state == FallState.FALL_CONFIRMED:
        track.state, track.state_entered_at = FallState.POST_FALL_MONITORING, now

    elif track.state == FallState.POST_FALL_MONITORING:
        if _is_currently_upright(snap):
            if track.recovery_start_time is None:
                track.recovery_start_time = now
            elif (now - track.recovery_start_time) >= RECOVERY_UPRIGHT_SECONDS:
                track.state, track.state_entered_at = FallState.RECOVERED, now
        else:
            track.recovery_start_time = None

    elif track.state == FallState.RECOVERED:
        if (now - track.state_entered_at) >= RECOVERY_UPRIGHT_SECONDS:
            track.state, track.state_entered_at = FallState.NORMAL, now
            track.high_score_since = None
            track.recovery_start_time = None

    elif track.state == FallState.UNKNOWN_LOW_POSTURE:
        if _is_currently_upright(snap):
            if track.recovery_start_time is None:
                track.recovery_start_time = now
            elif (now - track.recovery_start_time) >= UNKNOWN_LOW_POSTURE_UPRIGHT_SECONDS:
                track.state, track.state_entered_at = FallState.NORMAL, now
                track.recovery_start_time = None
        else:
            track.recovery_start_time = None
            if _had_upright_history(window) and fall_score >= FALL_SCORE_ENTER_THRESHOLD:
                if track.high_score_since is None:
                    track.high_score_since = now
                elif (now - track.high_score_since) >= POSSIBLE_FALL_SECONDS:
                    track.state, track.state_entered_at = FallState.POSSIBLE_FALL, now


# ----------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------

_STATE_LABELS = {
    FallState.NORMAL: None,               # resolved to Walking/Standing below
    FallState.POSSIBLE_FALL: "Possible Fall",
    FallState.FALL_CONFIRMED: "Fall Detected",
    FallState.POST_FALL_MONITORING: "Fall Detected",
    FallState.RECOVERED: "Recovered",
    FallState.UNKNOWN_LOW_POSTURE: "Lying Down (unconfirmed)",
}


def detect_action(object_id, box, keypoints, kp_conf=None):
    """Runs the fall state machine for one tracked person for this frame.
    Returns (label, debug) where debug exposes the live sub-scores/state so
    they can be displayed and the thresholds tuned against real values.
    Scoring uses only body geometry/keypoints -- never gender."""
    debug = {"fall_score": 0.0, "posture_score": 0.0, "transition_score": 0.0,
             "persistence_score": 0.0, "motion_score": 0.0, "observability": "LOW",
             "state": FallState.NORMAL.value}

    if kp_conf is None:
        kp_conf = [1.0] * 17
    if not keypoints or len(keypoints) < 17:
        return "Unknown", debug

    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if height <= 0 or width <= 0:
        return "Unknown", debug

    track = _tracks.setdefault(object_id, _Track())

    pose_confidence, observability, torso_angle = _extract_pose_features(keypoints, kp_conf)
    debug["observability"] = observability

    now = time.time()
    prev = track.history[-1] if track.history else None
    position_delta = None
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    if prev is not None:
        position_delta = math.hypot(center[0] - prev.center[0], center[1] - prev.center[1])

    snap = _Snapshot(
        timestamp=now, center=center, height=height,
        aspect_ratio=width / height, torso_angle=torso_angle,
        pose_confidence=pose_confidence, observability=observability,
        position_delta=position_delta,
    )
    track.history.append(snap)

    # Gates: not enough track history yet, or pose too unreliable to trust.
    if (now - track.track_started_at) < MIN_TRACK_AGE_SECONDS or pose_confidence < MIN_POSE_CONFIDENCE:
        debug["state"] = track.state.value
        return _resolve_label(track.state, keypoints), debug

    window = [s for s in track.history if now - s.timestamp <= TRANSITION_WINDOW_SECONDS]
    posture_score = _posture_score(snap)
    transition_score = _transition_score(window)
    persistence_score = _persistence_score(window)
    motion_score = _motion_score(window)

    obs_weight = 1.0 if observability == "HIGH" else (0.6 if observability == "MEDIUM" else 0.2)
    fall_score = (0.35 * posture_score + 0.30 * (transition_score * obs_weight)
                  + 0.20 * persistence_score + 0.15 * motion_score)

    debug.update(fall_score=round(fall_score, 3), posture_score=round(posture_score, 3),
                 transition_score=round(transition_score, 3), persistence_score=round(persistence_score, 3),
                 motion_score=round(motion_score, 3))

    _advance_state(track, snap, fall_score, window)
    debug["state"] = track.state.value

    return _resolve_label(track.state, keypoints), debug


def _resolve_label(state, keypoints):
    label = _STATE_LABELS.get(state)
    if label is not None:
        return label

    # NORMAL state: fall back to a simple walking/standing read from ankle spread.
    left_hip, right_hip = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    left_ankle, right_ankle = keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE]
    hip_width = abs(left_hip[0] - right_hip[0]) + 1e-6
    ankle_spread = abs(left_ankle[0] - right_ankle[0])
    return "Walking" if ankle_spread > hip_width * 1.5 else "Standing"
