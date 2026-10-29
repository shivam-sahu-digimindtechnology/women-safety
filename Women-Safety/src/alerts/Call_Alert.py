"""
Outbound AI Voice Call Alert Module.
Triggers an automated emergency voice call via Omnidimension Dispatch API.
Supports dynamic recipient phone number resolution (TRUSTED_CONTACT_PHONE / ALERT_PHONE_NUMBER)
and structured status reporting with cooldown gating.
"""

import os
import time
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Ensure environment is refreshed
load_dotenv()

OMNIDIM_DISPATCH_URL = "https://omnidim.io/api/v1/calls/dispatch"
CALL_COOLDOWN_SECONDS = 120  # Avoid rapid duplicate call placement
last_call_time: float = 0.0


def send_call_alert(message: str, recipient_phone: Optional[str] = None) -> Dict[str, Any]:
    """
    Triggers an outbound AI voice call via Omnidimension.
    
    Args:
        message: Alert description passed into the call context [alert_message].
        recipient_phone: Optional phone override in E.164 format (e.g. +919876543210).
    
    Returns:
        Structured result dict:
        {
            "status": "CALL_REQUESTED" | "COOLDOWN" | "CONFIG_MISSING" | "FAILED",
            "call_id": str,
            "recipient": str,
            "message": str
        }
    """
    global last_call_time
    load_dotenv()

    api_key = os.getenv("OMNIDIM_API_KEY", "").strip()
    agent_id = os.getenv("OMNIDIM_AGENT_ID", "").strip()
    from_number_id = os.getenv("OMNIDIM_FROM_NUMBER_ID", "").strip()
    
    # Resolve recipient phone dynamically
    phone = (
        recipient_phone or
        os.getenv("ALERT_PHONE_NUMBER") or
        os.getenv("TRUSTED_CONTACT_PHONE") or
        ""
    ).strip()

    # Credential validation
    if not api_key or not agent_id or not phone or "your_" in api_key or "your_" in agent_id:
        msg = "Omnidimension calling credentials or destination phone number not configured in .env."
        print(f"[Call Alert Notice] {msg}")
        return {
            "status": "CONFIG_MISSING",
            "recipient": phone or "NOT_CONFIGURED",
            "message": msg
        }

    if not agent_id.isdigit():
        msg = "OMNIDIM_AGENT_ID must be a numeric identifier from your Omnidimension dashboard."
        print(f"[Call Alert Warning] {msg}")
        return {
            "status": "CONFIG_MISSING",
            "recipient": phone,
            "message": msg
        }

    # Cooldown enforcement
    current_time = time.time()
    elapsed = current_time - last_call_time
    if elapsed < CALL_COOLDOWN_SECONDS:
        rem = int(CALL_COOLDOWN_SECONDS - elapsed)
        msg = f"Call alert in cooldown ({rem}s remaining)."
        print(f"[Call Alert] {msg}")
        return {
            "status": "COOLDOWN",
            "remaining_seconds": rem,
            "recipient": phone,
            "message": msg
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "agent_id": int(agent_id),
        "to_number": phone,
        "call_context": {
            "alert_message": message,
        },
    }
    if from_number_id and from_number_id.isdigit():
        payload["from_number_id"] = int(from_number_id)

    try:
        response = requests.post(OMNIDIM_DISPATCH_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        last_call_time = current_time

        res_json = {}
        try:
            res_json = response.json()
        except Exception:
            pass

        call_id = str(res_json.get("call_id") or res_json.get("id") or f"DISPATCHED_{int(current_time)}")
        print(f"[Call Alert Dispatched] Outbound call successfully requested (Call ID: {call_id})")
        return {
            "status": "CALL_REQUESTED",
            "call_id": call_id,
            "recipient": phone,
            "message": "Call requested successfully via Omnidimension.",
            "response": res_json
        }
    except requests.exceptions.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            err_msg += f" (Status: {e.response.status_code})"
        print(f"[Call Alert Error] Failed to place outbound call: {err_msg}")
        return {
            "status": "FAILED",
            "recipient": phone,
            "error": err_msg,
            "message": "Omnidimension call dispatch API request failed."
        }
