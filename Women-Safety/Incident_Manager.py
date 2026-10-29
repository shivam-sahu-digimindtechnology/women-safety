"""
Incident Evidence and Record Manager.
Handles creation of unique incident IDs, organized dated storage (outputs/incidents/YYYY-MM-DD/),
timestamped snapshots with red bounding boxes and incident headers, and structured JSON incident records.

Strict Rules:
- Saves snapshots ONLY for qualifying women safety incidents or verified emergency SOS events.
- Never saves snapshots for routine walking, normal proximity, or unconfirmed fights.
- Enforces duplicate snapshot prevention using incident cooldown.
- Files stored as: INCIDENT_YYYYMMDD_HHMMSS_XXX.jpg and INCIDENT_YYYYMMDD_HHMMSS_XXX.json.
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

import config
from Evidence_Recorder import add_frame as record_evidence_frame

# Cooldown tracking: incident_key -> last_saved_timestamp
_recent_incident_saves: Dict[str, float] = {}
_incident_counter = 0


def generate_incident_id() -> str:
    """Generates unique incident identifier in format INCIDENT_YYYYMMDD_HHMMSS_XXX."""
    global _incident_counter
    _incident_counter = (_incident_counter + 1) % 1000
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"INCIDENT_{now_str}_{_incident_counter:03d}"


def get_dated_incident_dir() -> str:
    """
    Creates and returns directory for the current date:
    outputs/incidents/YYYY-MM-DD/
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    base_dir = config.INCIDENT_SAVE_DIR
    target_dir = os.path.join(base_dir, today_str)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def is_in_save_cooldown(incident_type: str, victim_ids: List[int]) -> bool:
    """Checks if an identical incident was already saved within the cooldown period."""
    now = time.time()
    v_key = ",".join(map(str, sorted(victim_ids))) if victim_ids else "general"
    key = f"{incident_type}:{v_key}"

    last_time = _recent_incident_saves.get(key, 0.0)
    cooldown = getattr(config, "QUALIFYING_INCIDENT_COOLDOWN_SECONDS", 30.0)

    if (now - last_time) < cooldown:
        return True
    return False


def record_save_time(incident_type: str, victim_ids: List[int]) -> None:
    now = time.time()
    v_key = ",".join(map(str, sorted(victim_ids))) if victim_ids else "general"
    key = f"{incident_type}:{v_key}"
    _recent_incident_saves[key] = now


def create_annotated_incident_snapshot(
    frame: np.ndarray,
    incident_id: str,
    incident_type: str,
    risk_score: int,
    reasons: List[str],
    victim_ids: Optional[List[int]] = None
) -> np.ndarray:
    """Renders high-visibility incident header banner onto snapshot."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    banner_h = 70
    cv2.rectangle(annotated, (0, 0), (w, banner_h), (20, 20, 85), -1)
    cv2.line(annotated, (0, banner_h), (w, banner_h), (0, 0, 255), 3)

    test_tag = " [TEST MODE - SIMULATED]" if config.TEST_MODE else ""
    title = f"INCIDENT DETECTED: {incident_id}{test_tag}"
    cv2.putText(annotated, title, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    v_str = f"VICTIM ID(s): {victim_ids}" if victim_ids else "GENERAL ALERT"
    meta_line = (
        f"LOC: {config.LOCATION_NAME} | TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{v_str} | RISK: {risk_score}/100"
    )
    cv2.putText(annotated, meta_line[:95], (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 255), 1)

    reason_str = reasons[0] if reasons else incident_type
    cv2.putText(annotated, f"TYPE: {incident_type} -- {reason_str}"[:95], (15, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    # Bottom disclaimer banner
    cv2.rectangle(annotated, (0, h - 22), (w, h), (15, 15, 15), -1)
    disclaimer = "AUTOMATED WOMEN SAFETY AI DETECTION - REQUIRES AUTHORIZED VERIFICATION"
    cv2.putText(annotated, disclaimer, (15, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return annotated


def save_incident_evidence(
    raw_frame: np.ndarray,
    annotated_frame: Optional[np.ndarray],
    incident_type: str,
    risk_score: int,
    reasons: List[str],
    confidence: float = 0.80,
    victim_ids: Optional[List[int]] = None,
    delivery_status: Optional[Dict[str, Any]] = None,
    is_qualifying_incident: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Saves snapshot and structured JSON incident record to outputs/incidents/YYYY-MM-DD/.
    Only saves if is_qualifying_incident is True and cooldown has elapsed.
    """
    if not is_qualifying_incident:
        if config.DEBUG_MODE:
            print(f"[Incident Manager] Event '{incident_type}' does not meet qualifying woman-safety criteria; snapshot skipped.")
        return None

    # Check duplicate save cooldown
    if is_in_save_cooldown(incident_type, victim_ids or []):
        if config.DEBUG_MODE:
            print(f"[Incident Manager] Snapshot suppressed by cooldown for {incident_type}.")
        return None

    incident_id = generate_incident_id()
    dated_dir = get_dated_incident_dir()
    timestamp_iso = datetime.now().isoformat()
    datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    evidence_files = {}

    try:
        # Save annotated evidence frame with red boxes and metadata banner
        frame_to_save = annotated_frame if annotated_frame is not None else raw_frame
        if frame_to_save is not None and frame_to_save.size > 0:
            evidence_img = create_annotated_incident_snapshot(
                frame_to_save, incident_id, incident_type, risk_score, reasons, victim_ids
            )
            snapshot_filename = f"{incident_id}.jpg"
            snapshot_path = os.path.join(dated_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, evidence_img)
            evidence_files["snapshot"] = snapshot_filename
            evidence_files["snapshot_path"] = snapshot_path

        # Compile Structured JSON Incident Record
        incident_record = {
            "incident_id": incident_id,
            "timestamp": timestamp_iso,
            "datetime": datetime_str,
            "camera_id": f"Camera-{config.CAMERA_INDEX}",
            "location": {
                "name": config.LOCATION_NAME,
                "latitude": config.LOCATION_LATITUDE,
                "longitude": config.LOCATION_LONGITUDE,
                "source": "configured_installation"
            },
            "incident_type": incident_type,
            "risk_score": risk_score,
            "model_confidence": round(confidence, 2),
            "victim_tracking_ids": victim_ids or [],
            "observed_signals": reasons,
            "verification_status": "SUSPECTED_AUTOMATED_DETECTION",
            "is_human_confirmed": False,
            "evidence_files": evidence_files,
            "delivery_status": delivery_status or {
                "channel": config.ALERT_DELIVERY_CHANNEL,
                "test_mode": config.TEST_MODE,
                "delivered": False,
                "simulated": config.TEST_MODE,
                "status": "SIMULATED_TEST_MODE" if config.TEST_MODE else "PENDING"
            }
        }

        # Write individual JSON record
        json_filename = f"{incident_id}.json"
        json_path = os.path.join(dated_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(incident_record, f, indent=2)

        # Record cooldown timestamp
        record_save_time(incident_type, victim_ids or [])

        # Append to rolling history index
        _append_to_incident_history(incident_record, dated_dir)

        print(f"\n[Incident Manager] >>> Qualifying Safety Incident Snapshot Saved <<<")
        print(f"  Incident ID : {incident_id}")
        print(f"  Category    : {incident_type} (Score: {risk_score}/100)")
        print(f"  Victim ID(s): {victim_ids or 'N/A'}")
        print(f"  Directory   : {dated_dir}")
        print(f"  Snapshot    : {snapshot_filename}")
        print(f"  Record      : {json_filename}\n")

        return incident_record

    except Exception as e:
        print(f"[Incident Manager Error] Failed to save incident evidence: {e}")
        return {
            "incident_id": incident_id,
            "error": str(e)
        }


def _append_to_incident_history(record: Dict[str, Any], dated_dir: str):
    """Maintains a rolling JSON history index of recent incidents."""
    history_file = os.path.join(dated_dir, "incidents_history.json")
    history = []

    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    summary_entry = {
        "incident_id": record["incident_id"],
        "datetime": record["datetime"],
        "incident_type": record["incident_type"],
        "risk_score": record["risk_score"],
        "victim_tracking_ids": record["victim_tracking_ids"],
        "snapshot": record["evidence_files"].get("snapshot")
    }

    history.insert(0, summary_entry)
    history = history[:100]

    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def load_recent_incidents(limit: int = 10) -> List[Dict[str, Any]]:
    """Loads the most recent incidents from today's dated directory."""
    dated_dir = get_dated_incident_dir()
    history_file = os.path.join(dated_dir, "incidents_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)[:limit]
        except Exception:
            return []
    return []
