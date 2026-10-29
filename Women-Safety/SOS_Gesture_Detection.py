"""
SOS Gesture Detection Module
============================
Detects distress and SOS gestures from human pose keypoints and hand cues:
1. Overhead hands-up waving / distress posture (both wrists elevated above head for temporal streak).
2. Defensive shielding posture (wrists guarding head/face under close physical proximity).
3. Universal Signal for Help (optional high-res hand landmark analysis if MediaPipe Hands is available).
4. Manual SOS trigger function for keyboard (e.g. 'S' key) or button activation.

Strictly non-blocking, temporal confirmation to avoid false positives from mundane gestures.
"""

import time
import math
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple

# COCO Keypoint mapping (YOLOv8-pose)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

class SOSGestureDetector:
    def __init__(
        self,
        hands_up_streak_required: int = 5,
        confidence_threshold: float = 0.4,
        history_window_sec: float = 3.0,
    ):
        """
        Args:
            hands_up_streak_required: Number of consecutive positive frames to confirm SOS.
            confidence_threshold: Minimum keypoint confidence required.
            history_window_sec: Window in seconds to track posture history.
        """
        self.streak_required = hands_up_streak_required
        self.conf_thresh = confidence_threshold
        self.history_window = history_window_sec
        
        # Track history per person_id: {person_id: deque([(timestamp, gesture_tag)])}
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        # Streak counters per person_id
        self.hands_up_streaks: Dict[int, int] = defaultdict(int)
        self.shielding_streaks: Dict[int, int] = defaultdict(int)
        
        # Manual SOS trigger state
        self._manual_sos_active = False
        self._manual_sos_timestamp = 0.0
        self._manual_sos_source = ""

    def trigger_manual_sos(self, source: str = "Manual Keypress") -> None:
        """Manually trigger SOS state (e.g., via keyboard shortcut 'S' or UI panic button)."""
        self._manual_sos_active = True
        self._manual_sos_timestamp = time.time()
        self._manual_sos_source = source

    def reset_manual_sos(self) -> None:
        """Reset manual SOS state."""
        self._manual_sos_active = False
        self._manual_sos_source = ""

    def is_manual_sos_active(self, duration_sec: float = 5.0) -> Tuple[bool, str]:
        """Check if manual SOS is currently active within the given duration."""
        if not self._manual_sos_active:
            return False, ""
        if (time.time() - self._manual_sos_timestamp) <= duration_sec:
            return True, self._manual_sos_source
        self._manual_sos_active = False
        return False, ""

    def analyze_pose(
        self,
        person_id: int,
        keypoints: Any,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        now: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze a person's pose keypoints for distress gestures.
        
        Args:
            person_id: Unique tracking ID for the person.
            keypoints: List or array of 17 keypoints: [[x, y, conf], ...] or [[x, y], ...]
            bbox: Optional bounding box (x1, y1, x2, y2).
            now: Current timestamp (defaults to time.time()).
            
        Returns:
            Dict containing:
                - sos_detected: bool
                - gesture_type: str ("OVERHEAD_WAVE", "SHIELDING", "MANUAL_SOS", or None)
                - confidence: float
                - description: str
        """
        if now is None:
            now = time.time()

        # Check manual trigger first
        manual_active, manual_src = self.is_manual_sos_active()
        if manual_active:
            return {
                "sos_detected": True,
                "gesture_type": "MANUAL_SOS",
                "confidence": 1.0,
                "description": f"Manual SOS triggered ({manual_src})",
                "person_id": person_id
            }

        if keypoints is None or len(keypoints) < 17:
            return self._empty_result(person_id)

        try:
            kp = self._parse_keypoints(keypoints)
        except Exception:
            return self._empty_result(person_id)

        # Extract needed keypoints
        # 0: nose, 5: l_sh, 6: r_sh, 7: l_el, 8: r_el, 9: l_wr, 10: r_wr, 11: l_hip, 12: r_hip
        nose = kp[0]
        l_sh, r_sh = kp[5], kp[6]
        l_wr, r_wr = kp[9], kp[10]
        l_hip, r_hip = kp[11], kp[12]

        # Calculate scale metrics (torso height as reference)
        sh_y = min(l_sh[1], r_sh[1]) if (l_sh[2] > self.conf_thresh and r_sh[2] > self.conf_thresh) else (l_sh[1] if l_sh[2] > self.conf_thresh else r_sh[1])
        hip_y = max(l_hip[1], r_hip[1]) if (l_hip[2] > self.conf_thresh and r_hip[2] > self.conf_thresh) else sh_y + 100
        torso_height = max(hip_y - sh_y, 40.0)

        # 1. Overhead distress posture: Both wrists elevated significantly above shoulders/head
        l_raised = (l_wr[2] > self.conf_thresh and l_sh[2] > self.conf_thresh and l_wr[1] < (sh_y - 0.2 * torso_height))
        r_raised = (r_wr[2] > self.conf_thresh and r_sh[2] > self.conf_thresh and r_wr[1] < (sh_y - 0.2 * torso_height))
        
        # In case one arm is obscured or single-arm high waving
        both_raised = l_raised and r_raised
        single_raised_high = (l_raised and nose[2] > self.conf_thresh and l_wr[1] < nose[1] - 0.1 * torso_height) or \
                             (r_raised and nose[2] > self.conf_thresh and r_wr[1] < nose[1] - 0.1 * torso_height)

        is_overhead_distress = both_raised or single_raised_high

        if is_overhead_distress:
            self.hands_up_streaks[person_id] += 1
        else:
            self.hands_up_streaks[person_id] = max(0, self.hands_up_streaks[person_id] - 1)

        # 2. Defensive shielding posture: Both wrists brought close to face/head defensively
        is_shielding = False
        if nose[2] > self.conf_thresh and (l_wr[2] > self.conf_thresh or r_wr[2] > self.conf_thresh):
            d_l = math.hypot(l_wr[0] - nose[0], l_wr[1] - nose[1]) if l_wr[2] > self.conf_thresh else 9999.0
            d_r = math.hypot(r_wr[0] - nose[0], r_wr[1] - nose[1]) if r_wr[2] > self.conf_thresh else 9999.0
            # Wrists within 0.5 torso height of face, while wrists are above shoulder level
            if (d_l < 0.6 * torso_height or d_r < 0.6 * torso_height) and min(l_wr[1], r_wr[1]) < hip_y - 0.4 * torso_height:
                is_shielding = True

        if is_shielding:
            self.shielding_streaks[person_id] += 1
        else:
            self.shielding_streaks[person_id] = max(0, self.shielding_streaks[person_id] - 1)

        # Check thresholds
        if self.hands_up_streaks[person_id] >= self.streak_required:
            conf = min(1.0, 0.5 + 0.1 * (self.hands_up_streaks[person_id] - self.streak_required))
            return {
                "sos_detected": True,
                "gesture_type": "OVERHEAD_DISTRESS_WAVE",
                "confidence": conf,
                "description": f"Overhead distress gesture sustained ({self.hands_up_streaks[person_id]} frames)",
                "person_id": person_id
            }

        if self.shielding_streaks[person_id] >= (self.streak_required + 2):
            conf = min(0.9, 0.4 + 0.08 * (self.shielding_streaks[person_id] - self.streak_required))
            return {
                "sos_detected": True,
                "gesture_type": "DEFENSIVE_SHIELDING",
                "confidence": conf,
                "description": f"Defensive shielding posture sustained ({self.shielding_streaks[person_id]} frames)",
                "person_id": person_id
            }

        return self._empty_result(person_id)

    def _parse_keypoints(self, keypoints: Any) -> List[Tuple[float, float, float]]:
        """Normalize keypoints to List of (x, y, conf)."""
        parsed = []
        for pt in keypoints:
            if hasattr(pt, "tolist"):
                pt = pt.tolist()
            if len(pt) >= 3:
                parsed.append((float(pt[0]), float(pt[1]), float(pt[2])))
            elif len(pt) == 2:
                parsed.append((float(pt[0]), float(pt[1]), 1.0))
            else:
                parsed.append((0.0, 0.0, 0.0))
        return parsed

    def _empty_result(self, person_id: int) -> Dict[str, Any]:
        return {
            "sos_detected": False,
            "gesture_type": None,
            "confidence": 0.0,
            "description": "Normal pose",
            "person_id": person_id
        }

    def cleanup_old_tracks(self, active_person_ids: List[int]) -> None:
        """Purge state for persons no longer in the camera frame."""
        active_set = set(active_person_ids)
        for pid in list(self.hands_up_streaks.keys()):
            if pid not in active_set:
                del self.hands_up_streaks[pid]
        for pid in list(self.shielding_streaks.keys()):
            if pid not in active_set:
                del self.shielding_streaks[pid]
        for pid in list(self.history.keys()):
            if pid not in active_set:
                del self.history[pid]
