"""
Telegram Emergency Alert Module.
Transmits alert messages and situation snapshot photos to Telegram.
Configured via .env credentials.
"""

import os
import time
import cv2
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

_bot = None
last_alert_time = 0


def get_bot():
    """Initializes or retrieves the Telegram bot instance."""
    global _bot
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if token and ":" in token and "your_" not in token and _bot is None:
        try:
            import telebot
            _bot = telebot.TeleBot(token)
        except Exception as e:
            print(f"[Telegram Alert Error] Failed to initialize bot: {e}")
            _bot = None
    return _bot



def send_telegram_alert(frame, message):
    """Sends photo evidence and alert description to the configured Telegram chat."""
    global last_alert_time
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id or "your_" in token or ":" not in token:
        print("\n[Telegram Alert Notice] To send live messages to your phone/police group:")
        print("  1. Open the '.env' file in Notepad.")
        print("  2. Set your real TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        print("  3. Save the file. (Alerts are currently logged to console and saved locally).\n")
        return


    bot = get_bot()
    if not bot:
        print("[Telegram Alert Warning] Telegram bot could not be created with provided token.")
        return

    current_time = time.time()
    if current_time - last_alert_time < 20:
        return

    try:
        temp_img_path = os.path.join("evidence", "temp_telegram_alert.jpg")
        os.makedirs("evidence", exist_ok=True)

        if frame is not None and frame.size > 0:
            cv2.imwrite(temp_img_path, frame)
            with open(temp_img_path, 'rb') as photo:
                bot.send_photo(chat_id, photo, caption=f"🚨 EMERGENCY ALERT 🚨\n\n{message}")
        else:
            bot.send_message(chat_id, f"🚨 EMERGENCY ALERT 🚨\n\n{message}")

        last_alert_time = current_time
        print(f"[Telegram Alert Sent] Snapshot and alert successfully transmitted to Chat ID: {chat_id}")
        return True

    except Exception as e:
        err_msg = str(e)
        print(f"[Telegram Alert Error] Failed to send alert: {err_msg}")
        if "bot can't send messages to the bot" in err_msg.lower():
            print("\n  👉 HINT: In your .env file, TELEGRAM_CHAT_ID must be YOUR personal Telegram ID, not the bot's ID!")
            print("     1. Open Telegram and search for @userinfobot")
            print("     2. Press Start to get your personal 'Id' (e.g., 987654321)")
            print("     3. Put that number in .env as TELEGRAM_CHAT_ID=<your_user_id>")
            print("     4. Make sure to also open YOUR bot in Telegram and press 'Start'!\n")
        return False
