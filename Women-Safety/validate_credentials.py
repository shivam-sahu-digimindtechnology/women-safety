"""
Safe Credential and System Configuration Validator.
Audits configuration and external provider integrations locally WITHOUT exposing or printing secrets.
"""

import os
import re
import sys
from typing import Tuple
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

import config


def mask_secret(secret: str) -> str:
    """Masks secret values, showing only first 3 and last 3 characters or length."""
    if not secret:
        return "[NOT SET]"
    s = secret.strip()
    if len(s) <= 8:
        return f"[SET, {len(s)} chars]"
    return f"{s[:3]}...{s[-3:]} ({len(s)} chars)"


def validate_phone(phone: str) -> Tuple[bool, str]:
    if not phone or "9876543210" in phone or "your_" in phone:
        return False, "Placeholder or default number"
    # E.164 format: + followed by 8 to 15 digits
    match = re.match(r"^\+[1-9]\d{7,14}$", phone.strip())
    if match:
        return True, "Valid E.164 format"
    return False, "Not in standard E.164 format (e.g., +91XXXXXXXXXX)"


def audit_telegram() -> dict:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    result = {
        "channel": "Telegram Bot Alerts",
        "token_status": "MISSING",
        "chat_id_status": "MISSING",
        "overall": "MISSING",
        "details": ""
    }

    if not token or "your_" in token or ":" not in token:
        result["details"] = "TELEGRAM_BOT_TOKEN is missing or has placeholder value"
        return result

    result["token_status"] = "CONFIGURED"

    if not chat_id or "your_" in chat_id:
        result["details"] = "TELEGRAM_CHAT_ID is missing or has placeholder value"
        return result

    # Check if chat_id is the bot's own ID
    bot_id_part = token.split(":")[0]
    if chat_id == bot_id_part:
        result["chat_id_status"] = "INVALID"
        result["overall"] = "INVALID"
        result["details"] = f"TELEGRAM_CHAT_ID matches Bot ID ({bot_id_part}). Must be YOUR personal user ID from @userinfobot!"
        return result

    result["chat_id_status"] = "CONFIGURED"

    # Test authentication via telebot without exposing token
    try:
        import telebot
        bot = telebot.TeleBot(token)
        me = bot.get_me()
        bot_username = getattr(me, "username", "UnknownBot")
        result["token_status"] = "WORKING"

        # Try to verify chat access
        try:
            chat_obj = bot.get_chat(chat_id)
            chat_title = getattr(chat_obj, "title", None) or getattr(chat_obj, "first_name", "User Chat")
            result["chat_id_status"] = "WORKING"
            result["overall"] = "WORKING"
            result["details"] = f"Connected to @{bot_username}, destination chat verified: '{chat_title}'"
        except Exception as ce:
            err_str = str(ce)
            if "chat not found" in err_str.lower():
                result["chat_id_status"] = "INVALID"
                result["overall"] = "INVALID"
                result["details"] = f"Bot @{bot_username} authenticated, but Chat ID not found. Ensure user has messaged or started the bot."
            elif "bot can't send messages to the bot" in err_str.lower():
                result["chat_id_status"] = "INVALID"
                result["overall"] = "INVALID"
                result["details"] = "Chat ID is a bot ID. Must be your personal user ID from @userinfobot."
            else:
                result["chat_id_status"] = "INSUFFICIENT_PERMISSIONS"
                result["overall"] = "INSUFFICIENT_PERMISSIONS"
                result["details"] = f"Bot @{bot_username} authenticated. Chat check note: {err_str}"

    except Exception as te:
        err_msg = str(te)
        if "Unauthorized" in err_msg or "401" in err_msg:
            result["token_status"] = "INVALID"
            result["overall"] = "INVALID"
            result["details"] = "Telegram API 401 Unauthorized: token is invalid or revoked."
        else:
            result["overall"] = "NETWORK_ERROR"
            result["details"] = f"Connection error testing Telegram: {err_msg}"

    return result


def audit_omnidimension() -> dict:
    api_key = os.getenv("OMNIDIM_API_KEY", "").strip()
    agent_id = os.getenv("OMNIDIM_AGENT_ID", "").strip()
    recipient = os.getenv("TRUSTED_CONTACT_PHONE", "").strip()

    result = {
        "channel": "OmniDimension Voice Calling",
        "api_key_status": "MISSING",
        "agent_id_status": "MISSING",
        "recipient_status": "MISSING",
        "overall": "MISSING",
        "details": ""
    }

    has_key = api_key and "your_" not in api_key and len(api_key) > 10
    has_agent = agent_id and "your_" not in agent_id and len(agent_id) > 1
    valid_rcp, rcp_msg = validate_phone(recipient)

    result["api_key_status"] = "CONFIGURED" if has_key else "MISSING"
    result["agent_id_status"] = "CONFIGURED" if has_agent else "MISSING"
    result["recipient_status"] = "WORKING" if valid_rcp else "INVALID"

    if not has_key:
        result["details"] = "OMNIDIM_API_KEY is not configured or has placeholder value"
        return result
    if not has_agent:
        result["details"] = "OMNIDIM_AGENT_ID is not configured"
        return result
    if not valid_rcp:
        result["details"] = f"TRUSTED_CONTACT_PHONE format issue: {rcp_msg}"
        return result

    # Test API key validity with Omnidimension API endpoint if available
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            f"https://api.omnidim.io/agents/{agent_id}",
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "GuardianEye/2.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status in (200, 201):
                    result["overall"] = "WORKING"
                    result["details"] = f"Omnidimension API authenticated, Agent ID {agent_id} verified, recipient configured."
                else:
                    result["overall"] = "CONFIGURED"
                    result["details"] = f"API returned status {resp.status}."
        except urllib.error.HTTPError as he:
            if he.code == 401 or he.code == 403:
                result["api_key_status"] = "INVALID"
                result["overall"] = "INVALID"
                result["details"] = f"HTTP {he.code} Unauthorized: Omnidimension API key is invalid."
            elif he.code == 404:
                result["agent_id_status"] = "INVALID"
                result["overall"] = "INVALID"
                result["details"] = f"HTTP 404: Agent ID '{agent_id}' not found in your account."
            else:
                result["overall"] = "CONFIGURED"
                result["details"] = f"Endpoint responded with HTTP {he.code} (credentials set)."
        except Exception as e:
            result["overall"] = "CONFIGURED"
            result["details"] = f"Configured properly (Network probe info: {e})"
    except Exception as e:
        result["overall"] = "CONFIGURED"
        result["details"] = f"Credentials formatted and configured. ({e})"

    return result


def audit_models() -> dict:
    models = {
        "Person Detector (YOLOv8n)": (config.PERSON_MODEL_PATH, 6_000_000),
        "Gender Classifier (working_m_&_F)": (config.GENDER_MODEL_PATH, 18_000_000),
        "Violence Classifier (ResNet18-GRU)": (config.VIOLENCE_MODEL_PATH, 44_000_000),
        "Pose / Fall Detector (YOLOv8n-pose)": (config.POSE_MODEL_PATH, 6_000_000)
    }

    report = {}
    for name, (path, min_sz) in models.items():
        if not os.path.exists(path):
            report[name] = {"status": "MISSING", "path": path, "size_bytes": 0}
        else:
            sz = os.path.getsize(path)
            if sz < 1000:
                report[name] = {"status": "INVALID", "path": path, "size_bytes": sz}
            else:
                report[name] = {"status": "WORKING", "path": path, "size_bytes": sz}
    return report


def run_audit():
    print("\n" + "=" * 70)
    print("      GUARDIAN EYE: COMPREHENSIVE CONFIGURATION & CREDENTIAL AUDIT")
    print("=" * 70)

    # 1. Mode and Safety Gates
    print("\n[1] SAFETY GATES & OPERATION MODES:")
    print(f"  • TEST_MODE        : {config.TEST_MODE} ({'Simulated local alerts' if config.TEST_MODE else 'LIVE PRODUCTION DISPATCH'})")
    print(f"  • ALERTS_ENABLED   : {config.ALERTS_ENABLED} ({'External notifications active' if config.ALERTS_ENABLED else 'Disabled - local records only'})")
    print(f"  • INFERENCE_DEVICE : {getattr(config, 'INFERENCE_DEVICE', 'cpu').upper()}")

    # 2. Model Integrity
    print("\n[2] LOCAL TRAINED MODEL CHECKPOINTS:")
    model_res = audit_models()
    for name, info in model_res.items():
        sz_mb = info['size_bytes'] / (1024 * 1024)
        print(f"  • {name:<35}: [{info['status']}] ({sz_mb:.1f} MB) -> {Path(info['path']).name}")

    # 3. Telegram Integration
    print("\n[3] TELEGRAM BOT ALERT CHANNEL:")
    tg = audit_telegram()
    print(f"  • Status           : [{tg['overall']}]")
    print(f"  • Token Status     : [{tg['token_status']}]")
    print(f"  • Chat ID Status   : [{tg['chat_id_status']}]")
    print(f"  • Details          : {tg['details']}")

    # 4. Calling Integration
    print("\n[4] VOICE AI CALLING CHANNEL:")
    call = audit_omnidimension()
    print(f"  • Status           : [{call['overall']}]")
    print(f"  • API Key Status   : [{call['api_key_status']}]")
    print(f"  • Agent ID Status  : [{call['agent_id_status']}]")
    print(f"  • Recipient Status : [{call['recipient_status']}]")
    print(f"  • Details          : {call['details']}")

    # 5. Contacts & Location
    print("\n[5] METADATA & EMERGENCY CONTACTS:")
    valid_trusted, t_msg = validate_phone(config.TRUSTED_CONTACT_PHONE)
    valid_emerg, e_msg = validate_phone(config.EMERGENCY_CONTACT_PHONE)
    print(f"  • Trusted Contact Phone   : [{'VALID' if valid_trusted else 'INVALID'}] ({t_msg})")
    print(f"  • Emergency Contact Phone : [{'VALID' if valid_emerg else 'INVALID'}] ({e_msg})")
    print(f"  • Location Name           : {config.LOCATION_NAME}")
    print(f"  • Nearest Police Station  : {config.POLICE_STATION_NAME} ({config.POLICE_DISPATCH_HOTLINE})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    from typing import Tuple
    run_audit()
