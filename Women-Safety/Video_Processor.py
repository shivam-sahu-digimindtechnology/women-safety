"""
Video Processor and Pipeline Controller.
Unified pipeline with Person Counting, Female Victim Highlighting, and 6-Stage Incident Monitoring:
- 6-Stage Incident State Machine:
  NORMAL -> POSSIBLE_INCIDENT -> CONFIRMED_BY_CONFIGURED_RULES -> ALERT_QUEUED -> ALERT_SENT -> COOLDOWN
- Counters Dashboard:
  TOTAL PERSONS = MALE + FEMALE + UNKNOWN (Strict mathematical identity per visible tracked person)
- Model Integrations:
  * Person Tracking: Ultralytics YOLOv8 + ByteTrack
  * Gender Classification: working_m_&_F.pt (YOLO classes {0: female, 1: male}) with multi-frame EMA smoothing
  * Violence Detection: best.pt (ResNet18-GRU on RWF-2000, 8 sampled frames, 160x160)
  * Fall Detection: YOLOv8-pose temporal state machine
  * SOS Gestures: Overhead waving, defensive shielding, and keyboard 'S' trigger
  * Snatching Detection: Carried objects + departure velocity
  * Police Station & Helplines: OpenStreetMap verified directory lookup (112, 1091)
  * Calling Provider: Omnidimension Voice AI / safe simulation
- Scenario Highlights:
  * Scenario A: Woman attacked -> Highlight ONLY the female victim in BOLD RED with label 'WOMAN MAY BE UNSAFE'
  * Scenario B: Male-only fight -> Status 'FIGHT SUSPECTED' (No red box on anyone)
  * Scenario C: Normal activity -> Status 'NORMAL' (No red box)
  * Scenario D: Fight with unconfirmed female involvement -> Status 'VIOLENCE DETECTED — WOMAN INVOLVEMENT UNCONFIRMED'
  * Scenario E: Gender uncertain -> Status 'GENDER UNCERTAIN'
"""

import os
import sys
import time
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

import config
from Person_Detection import detect_and_track_persons, detect_carried_objects
from Gender_Detection import detect_genders_in_frame, get_or_predict_gender, cleanup_gender_tracks, _gender_cache
from Pose_Detection import detect_poses, detect_action
from Violence_Detection import classify_scene, reset_violence_state
from Interaction_Detection import detect_interactions
from Snatching_Detection import detect_snatching, reset_snatch_tracks
from SOS_Gesture_Detection import SOSGestureDetector
from Risk_Assessment import RiskAssessmentEngine, IncidentState
from Alert_Manager import dispatch_incident_alert
from Incident_Manager import record_evidence_frame

_sos_detector = SOSGestureDetector()
_cached_pose_results: List[Dict[str, Any]] = []


def rotate_frame(frame: np.ndarray, rotation_angle: int) -> np.ndarray:
    """Rotates frame if needed (e.g. mobile streaming horizontally)."""
    if rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes for accurate matching."""
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


def draw_visual_overlay(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    risk_info: Dict[str, Any],
    scene_info: Dict[str, Any],
    interactions: Dict[str, Any],
    female_ids: List[int],
    male_ids: List[int],
    unknown_ids: List[int],
    victim_female_ids: List[int],
    sos_active: bool = False,
    snatch_active: bool = False,
    fps: float = 0.0
) -> np.ndarray:
    """Renders the comprehensive Incident Monitoring HUD overlay with Female Victim Highlighting."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw Person Bounding Boxes with Specific Female Victim Highlighting
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        p_id = det["id"]
        gender = det.get("gender", "Unknown")
        action = det.get("action", "Normal")
        sos_info = det.get("sos_info", {})

        is_victim = (p_id in victim_female_ids)
        is_female = (p_id in female_ids)
        is_male = (p_id in male_ids)

        if is_victim:
            # SCENARIO A: BOLD RED BOX AROUND SUSPECTED FEMALE VICTIM
            box_color = config.VICTIM_BOX_COLOR  # (0, 0, 255) Red
            box_thickness = config.VICTIM_BOX_THICKNESS # 3
        elif is_female:
            box_color = (200, 30, 200)   # Purple for normal Female
            box_thickness = 2
        elif is_male:
            box_color = (255, 170, 0)    # Blue for Male
            box_thickness = 2
        else:
            box_color = (160, 160, 160)  # Slate Gray for Unknown
            box_thickness = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)

        # Corner reticles
        clen = min(14, (x2 - x1) // 4, (y2 - y1) // 4)
        if clen > 3:
            reticle_thick = 4 if is_victim else 3
            cv2.line(annotated, (x1, y1), (x1 + clen, y1), box_color, reticle_thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + clen), box_color, reticle_thick)
            cv2.line(annotated, (x2, y1), (x2 - clen, y1), box_color, reticle_thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + clen), box_color, reticle_thick)
            cv2.line(annotated, (x1, y2), (x1 + clen, y2), box_color, reticle_thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - clen), box_color, reticle_thick)
            cv2.line(annotated, (x2, y2), (x2 - clen, y2), box_color, reticle_thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - clen), box_color, reticle_thick)

        # Label Construction
        tag_parts = [f"ID {p_id}"]
        tag_parts.append(gender.upper())

        if is_victim:
            tag_parts.append(config.VICTIM_LABEL_TEXT)
        elif sos_info.get("sos_detected"):
            tag_parts.append("SOS DETECTED")
        elif action and action not in ("Normal", "Standing", "Walking"):
            tag_parts.append(action)

        tag_str = " | ".join(tag_parts)
        tsize, _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx2 = x1 + tsize[0] + 8
        ty1 = max(0, y1 - 22)
        ty2 = max(18, y1)
        cv2.rectangle(annotated, (x1, ty1), (tx2, ty2), box_color, -1)
        cv2.putText(annotated, tag_str, (x1 + 4, ty2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # 3. Top Header Bar: Counters Dashboard & Incident State
    top_bar_h = 75
    cv2.rectangle(annotated, (0, 0), (w, top_bar_h), (18, 18, 18), -1)

    # Status & State Badge
    current_status = risk_info.get("primary_incident", "SAFE / NO INCIDENT EVIDENCE")
    badge_color = (46, 204, 113) # Default Green
    if "UNCONFIRMED" in current_status or "GENDER" in current_status or "INSUFFICIENT" in current_status:
        badge_color = (0, 165, 255) # Amber for unconfirmed
    elif "FIGHT" in current_status:
        badge_color = (255, 140, 0) # Blue for male-only fight
    elif "SOS" in current_status or (("WOMAN" in current_status or "UNSAFE" in current_status) and len(victim_female_ids) > 0):
        badge_color = (0, 0, 255) # Red for confirmed woman safety incident
    elif risk_info.get("state") == "COOLDOWN":
        badge_color = (255, 170, 0) # Amber-blue for cooldown

    badge_w = 340
    cv2.rectangle(annotated, (10, 6), (badge_w, 34), badge_color, -1)
    status_display = current_status[:32]
    cv2.putText(annotated, status_display, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

    # Counters: TOTAL PERSONS = MALE + FEMALE + UNKNOWN
    n_females = len(female_ids)
    n_males = len(male_ids)
    n_unknown = len(unknown_ids)
    n_total = len(detections)
    counter_text = f"TOTAL PERSONS: {n_total} | MALE: {n_males} | FEMALE: {n_females} | UNKNOWN: {n_unknown}"
    cv2.putText(annotated, counter_text, (badge_w + 14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1)

    # Performance & Status Indicators (Right-aligned)
    status_x = w - 10
    
    # Live Performance Pill (FPS + Latency + Hardware Device)
    dev_name = getattr(config, "INFERENCE_DEVICE", "cpu").upper()
    if fps > 0:
        perf_text = f"{fps:.1f} FPS ({1000.0/fps:.0f}ms) | {dev_name}"
    else:
        perf_text = f"-- FPS | {dev_name}"
    pts, _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    status_x -= (pts[0] + 12)
    cv2.putText(annotated, perf_text, (status_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)

    if config.TEST_MODE:
        test_badge = "[DEMO]"
        ts, _ = cv2.getTextSize(test_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        status_x -= (ts[0] + 10)
        cv2.putText(annotated, test_badge, (status_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    if sos_active:
        sos_badge = "🚨 SOS ACTIVE"
        ss, _ = cv2.getTextSize(sos_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 2)
        status_x -= (ss[0] + 12)
        cv2.rectangle(annotated, (status_x - 4, 8), (status_x + ss[0] + 4, 30), (0, 0, 255), -1)
        cv2.putText(annotated, sos_badge, (status_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2)

    if snatch_active:
        snatch_badge = "⚠️ SNATCH"
        sns, _ = cv2.getTextSize(snatch_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 2)
        status_x -= (sns[0] + 12)
        cv2.rectangle(annotated, (status_x - 4, 8), (status_x + sns[0] + 4, 30), (0, 100, 255), -1)
        cv2.putText(annotated, snatch_badge, (status_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2)

    # Sub-bar (Row 2): Location, Police Station, and Emergency Call status
    police_status_str = f"Police: {config.POLICE_STATION_NAME} (112)" if config.POLICE_NOTIFICATION_ENABLED else "POLICE NOTIFICATION NOT CONFIGURED"
    call_status_str = f"Call: {config.TRUSTED_CONTACT_PHONE}" if (config.CALL_ALERTS_ENABLED and config.OMNIDIM_API_KEY) else "CALLING NOT CONFIGURED"
    sub_text = f"Loc: {config.LOCATION_NAME[:28]}  |  {police_status_str}  |  {call_status_str}"
    cv2.putText(annotated, sub_text[:115], (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

    # 4. Bottom Footer Bar: Explainable Situation Ticker & Shortcut hints
    cv2.rectangle(annotated, (0, h - 30), (w, h), (18, 18, 18), -1)
    reasons = risk_info.get("reasons", ["Monitoring active - no threats detected"])
    explanation = reasons[0][:80] if reasons else "All clear"
    footer_color = (0, 0, 255) if len(victim_female_ids) > 0 else ((0, 140, 255) if "FIGHT" in current_status else (220, 220, 220))
    cv2.putText(annotated, f" SITUATION: {explanation}", (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, footer_color, 1)

    hint_text = "[S: Manual SOS | R: Reset | Q: Quit]"
    hs, _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(annotated, hint_text, (w - hs[0] - 12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    return annotated


def process_frame(
    frame: np.ndarray,
    risk_engine: RiskAssessmentEngine,
    frame_idx: int = 0,
    fps: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Core frame processing pipeline with Female Victim Highlighting and Person Counting."""
    if frame is None or frame.size == 0:
        return frame, {}

    if config.CAMERA_ROTATION != 0:
        frame = rotate_frame(frame, config.CAMERA_ROTATION)

    h, w = frame.shape[:2]

    # 1. Person Detection & Tracking (YOLOv8 + ByteTrack)
    detections = detect_and_track_persons(frame, persist=True)

    # 2. Gender Classification (working_m_&_F.pt) with interval decimation
    gender_interval = getattr(config, "GENDER_INFERENCE_INTERVAL", 4)
    run_gender = (frame_idx % gender_interval == 0) or any(d["id"] not in _gender_cache for d in detections)
    if config.GENDER_CLASSIFICATION_ENABLED and run_gender:
        frame_gender_detections = detect_genders_in_frame(frame)
    else:
        frame_gender_detections = []

    female_ids = []
    male_ids = []
    unknown_ids = []

    for det in detections:
        p_id = det["id"]
        x1, y1, x2, y2 = det["box"]
        p_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if config.GENDER_CLASSIFICATION_ENABLED:
            gender, conf = get_or_predict_gender(p_id, det["box"], frame_gender_detections, p_crop)
            det["gender"] = gender
            det["gender_conf"] = conf
        else:
            det["gender"] = "Unknown"
            det["gender_conf"] = 0.0

        if det["gender"] == "Female":
            female_ids.append(p_id)
        elif det["gender"] == "Male":
            male_ids.append(p_id)
        else:
            unknown_ids.append(p_id)

    # Cleanup exited tracks in gender module
    cleanup_gender_tracks([d["id"] for d in detections])

    # 3. Pose, Fall Detection & SOS Gesture Analysis with interval decimation
    fall_confirmed = False
    fall_score = 0.0
    fall_person_ids = []
    sos_active = False
    sos_active_info = {}

    global _cached_pose_results
    if config.POSE_DETECTION_ENABLED:
        pose_interval = getattr(config, "POSE_INFERENCE_INTERVAL", 3)
        if (frame_idx % pose_interval == 0) or not _cached_pose_results:
            pose_results = detect_poses(frame)
            _cached_pose_results = pose_results
        else:
            pose_results = _cached_pose_results

        matched_pose_indices = set()

        for det in detections:
            p_id = det["id"]
            box = det["box"]
            best_iou = 0.0
            best_idx = -1

            for p_idx, p_res in enumerate(pose_results):
                if p_idx in matched_pose_indices:
                    continue
                iou = compute_iou(box, p_res["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = p_idx

            if best_idx >= 0 and best_iou >= 0.20:
                matched_pose_indices.add(best_idx)
                kps = pose_results[best_idx].get("keypoints", [])
                kp_conf = pose_results[best_idx].get("kp_conf", [])
                det["keypoints"] = kps

                # Fall Detection
                action, debug_info = detect_action(p_id, box, kps, kp_conf)
                det["action"] = action
                det["fall_score"] = debug_info.get("fall_score", 0.0)

                if action == "Fall Detected":
                    fall_confirmed = True
                    fall_score = max(fall_score, det["fall_score"])
                    fall_person_ids.append(p_id)

                # SOS Gesture Analysis
                sos_res = _sos_detector.analyze_pose(p_id, kps, box)
                det["sos_info"] = sos_res
                if sos_res.get("sos_detected"):
                    sos_active = True
                    sos_active_info = sos_res
            else:
                det["action"] = "Normal"
                det["fall_score"] = 0.0
                det["sos_info"] = {"sos_detected": False}

        _sos_detector.cleanup_old_tracks([d["id"] for d in detections])

    # Check manual SOS trigger
    manual_active, manual_src = _sos_detector.is_manual_sos_active()
    if manual_active:
        sos_active = True
        sos_active_info = {"sos_detected": True, "description": f"Manual SOS ({manual_src})", "confidence": 1.0}

    # 4. Spatio-Temporal Snatching Detection
    carried_objects = detect_carried_objects(frame)
    snatch_res = detect_snatching(detections, carried_objects=carried_objects, frame_shape=(h, w))
    snatch_active = snatch_res.get("is_snatching", False)

    # 5. Interaction Analysis
    interactions = detect_interactions(detections, frame_shape=(h, w))

    # 6. Violence Classification with CNN-GRU (best.pt)
    scene_info = classify_scene(frame)
    is_fight = scene_info.get("is_violence", False)

    # 7. Female Victim Attribution Logic (Scenarios A, B, C, D, E)
    female_attacked_ids = []
    female_surrounded = False
    surrounded_female_ids = []
    female_followed = False
    followed_female_ids = []

    # Check if a female is in direct physical struggle/fight
    # Positive evidence requires direct fighting pair OR physical bounding box overlap (IoU >= 0.08) with a combatant during violence
    if is_fight or interactions.get("fighting_pairs"):
        for f_id in female_ids:
            f_det = next((d for d in detections if d["id"] == f_id), None)
            if not f_det:
                continue

            for other_det in detections:
                if other_det["id"] == f_id:
                    continue

                in_fight_pair = (f_id, other_det["id"]) in interactions.get("fighting_pairs", []) or \
                                (other_det["id"], f_id) in interactions.get("fighting_pairs", [])

                # Direct physical collision/overlap during violence
                box_iou = compute_iou(f_det["box"], other_det["box"])
                is_physical_combat = is_fight and (box_iou >= 0.08)

                if in_fight_pair or is_physical_combat:
                    female_attacked_ids.append(f_id)
                    break

    # Hostile encirclement check: requires AT LEAST 3 individuals closely crowding (<= 80px)
    for f_id in female_ids:
        f_det = next((d for d in detections if d["id"] == f_id), None)
        if not f_det:
            continue
        fx1, fy1, fx2, fy2 = f_det["box"]
        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        close_others = 0
        for other_det in detections:
            if other_det["id"] == f_id:
                continue
            ox1, oy1, ox2, oy2 = other_det["box"]
            ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            if np.hypot(fcx - ocx, fcy - ocy) <= 80:
                close_others += 1
        if close_others >= 3:
            female_surrounded = True
            surrounded_female_ids.append(f_id)

    # Following check
    for pair in interactions.get("following_pairs", []):
        if pair[0] in female_ids:
            female_followed = True
            followed_female_ids.append(pair[0])
        elif pair[1] in female_ids:
            female_followed = True
            followed_female_ids.append(pair[1])

    # 8. Aggregate Signals into Risk Engine
    signals = {
        "female_ids": female_ids,
        "male_ids": male_ids,
        "unknown_ids": unknown_ids,
        "sos_detected": sos_active,
        "sos_details": sos_active_info,
        "snatching_detected": snatch_active,
        "snatching_details": snatch_res,
        "female_attacked": (len(female_attacked_ids) > 0),
        "female_attacked_ids": female_attacked_ids,
        "female_surrounded": female_surrounded,
        "surrounded_female_ids": surrounded_female_ids,
        "female_followed": female_followed,
        "followed_female_ids": followed_female_ids,
        "fall_confirmed": fall_confirmed,
        "fall_person_ids": fall_person_ids,
        "fall_score": fall_score,
        "violence_detected": is_fight,
        "violence_confidence": scene_info.get("confidence", 0.0),
        "violence_label": scene_info.get("label", "NonFight"),
        "fighting_pairs": interactions.get("fighting_pairs", []),
        "snatching_pairs": interactions.get("snatching_pairs", []),
        "following_pairs": interactions.get("following_pairs", []),
        "loitering_ids": interactions.get("loitering_ids", []),
        "is_crowd_dense": interactions.get("is_crowd_dense", False),
    }

    risk_info = risk_engine.evaluate_frame(signals)
    victim_female_ids = risk_info.get("victim_female_ids", [])

    # 9. Render HUD Visual Overlay (Highlights female victim with bold RED box)
    annotated = draw_visual_overlay(
        frame=frame,
        detections=detections,
        risk_info=risk_info,
        scene_info=scene_info,
        interactions=interactions,
        female_ids=female_ids,
        male_ids=male_ids,
        unknown_ids=unknown_ids,
        victim_female_ids=victim_female_ids,
        sos_active=sos_active,
        snatch_active=snatch_active,
        fps=fps
    )

    # 10. DISPATCH INCIDENT ALERT WHEN CONFIRMED
    if risk_info.get("alert_worthy", False):
        risk_engine.mark_alert_queued()
        dispatch_incident_alert(
            raw_frame=frame,
            annotated_frame=annotated,
            incident_type=risk_info.get("primary_incident", "Suspected Threat"),
            risk_score=risk_info.get("score", 0),
            reasons=risk_info.get("reasons", []),
            confidence=0.85,
            victim_ids=victim_female_ids,
            is_qualifying_incident=risk_info.get("is_qualifying_incident", False),
            async_mode=True
        )

    # 11. Record evidence frame into rolling video ring-buffer
    record_evidence_frame(frame)

    metadata = {
        "person_count": len(detections),
        "female_ids": female_ids,
        "male_ids": male_ids,
        "unknown_ids": unknown_ids,
        "victim_female_ids": victim_female_ids,
        "risk_info": risk_info,
        "scene_info": scene_info,
        "interactions": interactions,
        "sos_active": sos_active,
        "snatch_active": snatch_active,
    }

    return annotated, metadata


def process_image(image_path: str) -> bool:
    """Processes a single input image and saves the annotated result."""
    print(f"\n[Video Processor] Loading Image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Failed to read image: {image_path}")
        return False

    risk_engine = RiskAssessmentEngine()
    annotated, meta = process_frame(image, risk_engine, frame_idx=0)

    out_file = os.path.join(config.OUTPUT_DIR, "annotated_image.jpg")
    cv2.imwrite(out_file, annotated)
    print(f"[Video Processor] Annotated image saved -> {out_file}")
    print(f"[Summary] {meta['risk_info']['summary']}")

    if config.DISPLAY_WINDOW:
        display_frame = _resize_for_display(annotated)
        cv2.imshow(config.WINDOW_TITLE, display_frame)
        print("\n[Info] Displaying result window. Press any key in the window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def process_video(video_path: str) -> bool:
    """Processes an input video file and writes an annotated video to output dir."""
    print(f"\n[Video Processor] Loading Video: {video_path}")
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if config.CAMERA_ROTATION in (90, 270):
        width, height = height, width

    out_file = os.path.join(config.OUTPUT_DIR, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    risk_engine = RiskAssessmentEngine()
    reset_violence_state()
    reset_snatch_tracks()
    frame_idx = 0
    t_start = time.time()

    print(f"[Video Processor] Processing {total_frames} frames -> {out_file}")
    if config.DISPLAY_WINDOW:
        print(" [Tip] Press 's' for manual SOS | 'r' to reset risk state | 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            t_now = time.time()
            current_fps = frame_idx / (t_now - t_start) if (t_now - t_start) > 0 else 0.0

            annotated, _ = process_frame(frame, risk_engine, frame_idx, fps=current_fps)
            writer.write(annotated)

            if config.DISPLAY_WINDOW:
                display_frame = _resize_for_display(annotated)
                cv2.imshow(config.WINDOW_TITLE, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Info] Processing interrupted by user.")
                    break
                elif key == ord('s'):
                    print("\n[Manual SOS] Key 's' pressed! Triggering emergency alert...")
                    _sos_detector.trigger_manual_sos("Keyboard 'S' Shortcut")
                elif key == ord('r'):
                    print("\n[State Machine] Key 'r' pressed! Resetting risk engine...")
                    risk_engine.reset()

            if frame_idx % 30 == 0:
                print(f"  Processed frame {frame_idx}/{total_frames} ({current_fps:.1f} FPS)...")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\n[Video Processor] Video processing complete -> {out_file}")
    return True


def process_stream(source: Any, is_rtsp: bool = False) -> bool:
    """Processes live webcam or RTSP stream feed with interactive OpenCV display."""
    mode_name = "RTSP Stream" if is_rtsp else "Webcam"
    print(f"\n[Video Processor] Opening {mode_name} source: {source}")

    if getattr(config, "USE_THREADED_CAMERA", True):
        try:
            from VideoStream import VideoStream
            cap = VideoStream(source)
            print(f"[Video Processor] Threaded VideoStream active with latest-frame buffer (lag-free).")
        except Exception as e:
            print(f"[Video Processor Warning] VideoStream fallback to standard capture: {e}")
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Error] Failed to open {mode_name} source: {source}")
        if not is_rtsp:
            print("[Tip] Ensure your webcam is plugged in and not in use by another app.")
        return False

    risk_engine = RiskAssessmentEngine()
    reset_violence_state()
    reset_snatch_tracks()
    frame_idx = 0
    prev_time = time.time()
    fps = 0.0

    print(f"[Video Processor] {mode_name} active. [S: Manual SOS | R: Reset | Q: Quit]")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_rtsp:
                    print("[Warning] Stream disconnected. Reconnecting...")
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(source)
                    continue
                else:
                    print("[Warning] Failed to read frame from webcam.")
                    break

            frame_idx += 1
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            prev_time = now

            if config.FRAME_INTERVAL > 1 and frame_idx % config.FRAME_INTERVAL != 0:
                continue

            annotated, _ = process_frame(frame, risk_engine, frame_idx, fps=fps)

            if config.DISPLAY_WINDOW:
                display_frame = _resize_for_display(annotated)
                cv2.imshow(config.WINDOW_TITLE, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Info] Closing video feed.")
                    break
                elif key == ord('s'):
                    print("\n[Manual SOS] Key 's' pressed! Triggering emergency alert...")
                    _sos_detector.trigger_manual_sos("Keyboard 'S' Shortcut")
                elif key == ord('r'):
                    print("\n[State Machine] Key 'r' pressed! Resetting risk engine...")
                    risk_engine.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


def _resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Scales frame for display window if DISPLAY_WIDTH is specified in config."""
    if not config.DISPLAY_WIDTH or frame.shape[1] <= config.DISPLAY_WIDTH:
        return frame
    scale = config.DISPLAY_WIDTH / frame.shape[1]
    new_h = int(frame.shape[0] * scale)
    return cv2.resize(frame, (config.DISPLAY_WIDTH, new_h))


def run_pipeline(mode: Optional[str] = None):
    """Entry point to launch the configured processing pipeline."""
    selected_mode = (mode or config.INPUT_MODE).lower()

    if selected_mode == "image":
        process_image(config.INPUT_IMAGE_PATH)
    elif selected_mode == "video":
        process_video(config.INPUT_VIDEO_PATH)
    elif selected_mode == "webcam":
        process_stream(config.CAMERA_INDEX, is_rtsp=False)
    elif selected_mode == "rtsp":
        process_stream(config.RTSP_URL, is_rtsp=True)
    else:
        print(f"[Error] Unknown INPUT_MODE '{selected_mode}'. Valid options: image, video, webcam, rtsp")
