"""
Alert Manager Module.
Coordinates safe alert delivery across channels (Telegram, Call, Local Evidence)
with an asynchronous, thread-safe background dispatch queue, strict test-mode simulation,
delivery receipts, and cooldown gates.

Features:
- Thread-safe background worker queue (queue.Queue): video processing loop never drops FPS due to network latency.
- Strict cooldown enforcement.
- Local incident evidence recording (raw frame, annotated frame, JSON record) in outputs/incidents/YYYY-MM-DD/.
- Safe test-mode simulation by default (TEST_MODE = True, ALERTS_ENABLED = False).
- Status introspection for the 6-stage risk assessment engine.
"""

import os
import time
import queue
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

import config
from Telebot_Alert import send_telegram_alert
from Call_Alert import send_call_alert
from Incident_Manager import save_incident_evidence

_last_incident_alert_time: float = 0.0
_alert_queue: queue.Queue = queue.Queue(maxsize=10)
_worker_thread: Optional[threading.Thread] = None
_last_delivery_result: Dict[str, Any] = {"status": "IDLE", "timestamp": ""}
_status_lock = threading.Lock()


def format_incident_message(incident_record: Dict[str, Any]) -> str:
    """Formats a structured, human-readable emergency dispatch notification message."""
    inc_id = incident_record.get("incident_id", "N/A")
    dt = incident_record.get("datetime", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    loc = incident_record.get("location", {}).get("name", config.LOCATION_NAME)
    inc_type = incident_record.get("incident_type", "WOMAN SAFETY INCIDENT — EVIDENCE SUPPORTED")
    score = incident_record.get("risk_score", 0)
    signals = incident_record.get("observed_signals", [])
    conf = incident_record.get("model_confidence", 0.0)
    victims = incident_record.get("victim_tracking_ids", [])
    cam_id = incident_record.get("camera_id", f"Camera-{config.CAMERA_INDEX}")

    signals_text = "\n".join(f"  • {s}" for s in signals[:3])

    test_badge = "⚠️ [TEST MODE - SIMULATED INCIDENT] ⚠️\n" if config.TEST_MODE else "🚨 [OFFICIAL SAFETY ALERT] 🚨\n"
    victim_line = f"Suspected Victim Track: {victims}\n" if victims else ""

    msg = (
        f"{test_badge}"
        f"WOMEN SAFETY ALERT — POTENTIAL INCIDENT\n\n"
        f"Incident ID   : {inc_id}\n"
        f"Date & Time   : {dt}\n"
        f"Location      : {loc}\n"
        f"Detected Event: {inc_type}\n"
        f"Camera ID     : {cam_id}\n"
        f"{victim_line}"
        f"Risk Score    : {score}/100 (Confidence: {int(conf * 100)}%)\n\n"
        f"Observed Visual Evidence:\n{signals_text}\n\n"
        f"Nearest Police Station: {config.POLICE_STATION_NAME}\n"
        f"Emergency Helpline    : {config.POLICE_DISPATCH_HOTLINE}\n\n"
        f"⚠️ DISCLAIMER: Automated machine vision detection. Requires human verification before emergency escalation."
    )
    return msg


def _worker_loop():
    """Background daemon worker consuming alert tasks from queue without blocking video pipeline."""
    global _last_delivery_result
    while True:
        try:
            task = _alert_queue.get()
            if task is None:
                break
            
            raw_frame = task["raw_frame"]
            annotated_frame = task["annotated_frame"]
            incident_type = task["incident_type"]
            risk_score = task["risk_score"]
            reasons = task["reasons"]
            confidence = task["confidence"]
            victim_ids = task.get("victim_ids", [])
            is_qualifying = task.get("is_qualifying_incident", True)

            result = _execute_dispatch(
                raw_frame=raw_frame,
                annotated_frame=annotated_frame,
                incident_type=incident_type,
                risk_score=risk_score,
                reasons=reasons,
                confidence=confidence,
                victim_ids=victim_ids,
                is_qualifying_incident=is_qualifying
            )

            with _status_lock:
                _last_delivery_result = result

            _alert_queue.task_done()
        except Exception as e:
            print(f"[Alert Manager Worker Error] {e}")


def _ensure_worker_running():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="AlertDispatchWorker")
        _worker_thread.start()


def get_last_delivery_result() -> Dict[str, Any]:
    """Retrieve the latest alert delivery result safely."""
    with _status_lock:
        return dict(_last_delivery_result)


def is_queue_busy() -> bool:
    """Return True if background alert queue has pending dispatches."""
    return not _alert_queue.empty()


def dispatch_incident_alert(
    raw_frame: np.ndarray,
    annotated_frame: Optional[np.ndarray],
    incident_type: str,
    risk_score: int,
    reasons: List[str],
    confidence: float = 0.80,
    victim_ids: Optional[List[int]] = None,
    is_qualifying_incident: bool = True,
    async_mode: bool = True
) -> Dict[str, Any]:
    """
    Main dispatch entry point for confirmed qualifying incidents.
    1. Enforces cooldown.
    2. Enqueues dispatch in async worker thread (or runs synchronously if async_mode=False).
    3. Returns immediately with queue status to avoid dropping camera FPS.
    """
    global _last_incident_alert_time
    now = time.time()

    # Enforce cooldown gate
    if now - _last_incident_alert_time < config.ALERT_COOLDOWN_SECONDS:
        remaining = int(config.ALERT_COOLDOWN_SECONDS - (now - _last_incident_alert_time))
        if config.DEBUG_MODE:
            print(f"[Alert Manager] Incident suppressed by cooldown ({remaining}s remaining): {incident_type}")
        return {
            "delivered": False,
            "suppressed_by_cooldown": True,
            "remaining_seconds": remaining,
            "status": "COOLDOWN_ACTIVE"
        }

    _last_incident_alert_time = now

    task = {
        "raw_frame": raw_frame.copy() if raw_frame is not None else None,
        "annotated_frame": annotated_frame.copy() if annotated_frame is not None else None,
        "incident_type": incident_type,
        "risk_score": risk_score,
        "reasons": reasons,
        "confidence": confidence,
        "victim_ids": victim_ids or [],
        "is_qualifying_incident": is_qualifying_incident,
        "queued_at": now
    }

    if async_mode:
        _ensure_worker_running()
        try:
            _alert_queue.put_nowait(task)
            with _status_lock:
                _last_delivery_result["status"] = "ALERT_QUEUED"
            return {
                "delivered": False,
                "status": "ALERT_QUEUED",
                "message": "Incident alert placed in non-blocking background queue."
            }
        except queue.Full:
            return {
                "delivered": False,
                "status": "QUEUE_FULL",
                "message": "Background alert queue is full, dropping duplicate trigger."
            }
    else:
        # Synchronous execution
        return _execute_dispatch(
            raw_frame=raw_frame,
            annotated_frame=annotated_frame,
            incident_type=incident_type,
            risk_score=risk_score,
            reasons=reasons,
            confidence=confidence,
            victim_ids=victim_ids,
            is_qualifying_incident=is_qualifying_incident
        )


def _execute_dispatch(
    raw_frame: np.ndarray,
    annotated_frame: Optional[np.ndarray],
    incident_type: str,
    risk_score: int,
    reasons: List[str],
    confidence: float = 0.80,
    victim_ids: Optional[List[int]] = None,
    is_qualifying_incident: bool = True
) -> Dict[str, Any]:
    """Internal execution logic for incident persistence and notification."""
    delivery_status: Dict[str, Any] = {
        "channel": config.ALERT_DELIVERY_CHANNEL,
        "test_mode": config.TEST_MODE,
        "delivered": False,
        "simulated": config.TEST_MODE,
        "status": "PROCESSING",
        "timestamp": datetime.now().isoformat()
    }

    # 1. Save incident evidence & write JSON record into outputs/incidents/YYYY-MM-DD/
    incident_record = save_incident_evidence(
        raw_frame=raw_frame,
        annotated_frame=annotated_frame,
        incident_type=incident_type,
        risk_score=risk_score,
        reasons=reasons,
        confidence=confidence,
        victim_ids=victim_ids,
        delivery_status=delivery_status,
        is_qualifying_incident=is_qualifying_incident
    )

    if incident_record is None or "incident_id" not in incident_record:
        delivery_status["status"] = "SKIPPED_NOT_QUALIFYING"
        return delivery_status

    formatted_msg = format_incident_message(incident_record)

    # 2. Check Safety Master Switch
    if not config.ALERTS_ENABLED:
        print("\n" + "=" * 65)
        print(f" [Alert Manager] ALERTS_ENABLED = False in config.py")
        print(f" Evidence recorded in outputs: {incident_record.get('incident_id')}")
        print(f" External notifications suppressed.")
        print("=" * 65 + "\n")
        delivery_status["status"] = "SUPPRESSED_BY_CONFIG"
        return delivery_status

    # 3. Handle TEST MODE simulation
    if config.TEST_MODE:
        print("\n" + "=" * 70)
        print(" 🧪 [TEST MODE ACTIVE] Simulated Incident Alert Dispatch")
        print("=" * 70)
        print(f" Incident ID        : {incident_record.get('incident_id')}")
        print(f" Target Channel     : {config.ALERT_DELIVERY_CHANNEL.upper()} (Simulated)")
        print(f" Victim Track ID(s) : {victim_ids or 'N/A'}")
        print(f" Trusted Contact    : {config.TRUSTED_CONTACT_PHONE}")
        print(f" Emergency Contact  : {config.EMERGENCY_CONTACT_PHONE}")
        print(f" Nearest Station    : {config.POLICE_STATION_NAME} (Directory lookup only)")
        print(f" Dispatch Summary   :\n{formatted_msg}")
        print("=" * 70)
        print(" [SAFEGUARD] No real phone calls or external police reports were dispatched.")
        print(" Set TEST_MODE = False in config.py when ready for live delivery.")
        print("=" * 70 + "\n")

        delivery_status["status"] = "SIMULATED_SUCCESS"
        delivery_status["delivered"] = True
        return delivery_status

    # 4. LIVE DELIVERY MODE (TEST_MODE = False, ALERTS_ENABLED = True)
    print(f"\n[Alert Manager] >>> INITIATING LIVE INCIDENT DISPATCH: {incident_record.get('incident_id')} <<<")

    evidence_img = annotated_frame if annotated_frame is not None else raw_frame

    # Telegram Delivery
    if config.TELEGRAM_ALERTS_ENABLED:
        try:
            tg_success = send_telegram_alert(evidence_img, formatted_msg)
            if tg_success:
                delivery_status["telegram_delivered"] = True
                delivery_status["delivered"] = True
                print("[Alert Manager] Telegram incident snapshot delivered successfully.")
            else:
                delivery_status["telegram_error"] = "Telegram transmission failed (check token / chat_id)."
        except Exception as e:
            delivery_status["telegram_error"] = str(e)
            print(f"[Alert Manager Error] Telegram delivery failed: {e}")

    # Trusted Contact Calling (if configured)
    if config.CALL_ALERTS_ENABLED:
        try:
            call_msg = f"Safety incident at {config.LOCATION_NAME}: {incident_type}. Please check your messages immediately."
            recipient_phone = config.TRUSTED_CONTACT_PHONE or os.getenv("ALERT_PHONE_NUMBER")
            call_result = send_call_alert(call_msg, recipient_phone=recipient_phone)
            delivery_status["call_dispatch"] = call_result
            if call_result.get("status") == "CALL_REQUESTED":
                delivery_status["call_requested"] = True
                delivery_status["delivered"] = True
                print(f"[Alert Manager] Emergency call dispatched to contact: {recipient_phone}")
            elif call_result.get("status") == "COOLDOWN":
                delivery_status["call_suppressed_cooldown"] = True
                print(f"[Alert Manager] Call alert in cooldown ({call_result.get('remaining_seconds')}s remaining).")
            else:
                delivery_status["call_error"] = call_result.get("message") or call_result.get("error")
        except Exception as e:
            delivery_status["call_error"] = str(e)
            print(f"[Alert Manager Warning] Emergency call dispatch failed: {e}")

    delivery_status["status"] = "DELIVERED" if delivery_status["delivered"] else "DELIVERY_FAILED"
    return delivery_status


def test_alert_subsystem() -> bool:
    """Utility command to test the alert subsystem safely."""
    print("\n[Alert Manager] Running Alert Subsystem Self-Test...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "ALERT SUBSYSTEM TEST FRAME", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    res = dispatch_incident_alert(
        raw_frame=dummy_frame,
        annotated_frame=dummy_frame,
        incident_type="Subsystem Diagnostics Test",
        risk_score=95,
        reasons=["User-initiated diagnostic alert test"],
        confidence=1.0,
        victim_ids=[1],
        is_qualifying_incident=True,
        async_mode=False
    )
    print(f"[Alert Manager] Self-test complete. Status: {res.get('status')}")
    return res.get("status") in ("SIMULATED_SUCCESS", "DELIVERED", "SUPPRESSED_BY_CONFIG")


if __name__ == "__main__":
    test_alert_subsystem()
