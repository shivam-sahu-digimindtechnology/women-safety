"""
Central Configuration File for Women Safety AI Incident Detection & Alert System.
All adjustable application settings, model paths, thresholds, and safety toggles are managed here.
Provider secrets and tokens must be stored in .env and never hardcoded in this file.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load secrets from .env if present
load_dotenv()

# ==========================================
# 1. CORE APPLICATION & PATHS
# ==========================================
APP_NAME = "Women Safety AI - Incident Monitoring System"
DEBUG_MODE = False
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Base directory & src package resolution
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
for sub in ["detection", "tracking", "incident_detection", "alerts", "utilities"]:
    sub_path = str(SRC_DIR / sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# SAFETY GATES:
# TEST_MODE = True simulates alerts without sending live external messages or placing real phone calls.
# ALERTS_ENABLED = False prevents unexpected external notifications until explicitly enabled.
TEST_MODE = False
ALERTS_ENABLED = True

# Location settings (for CCTV context and nearest police station lookup)
LOCATION_NAME = os.getenv("LOCATION_NAME", "Sector 17 Central Plaza, North Gate")
LOCATION_LATITUDE = float(os.getenv("LOCATION_LATITUDE", "30.7415"))
LOCATION_LONGITUDE = float(os.getenv("LOCATION_LONGITUDE", "76.7794"))

# Directory for storing qualifying incident snapshots and JSON records: outputs/incidents/YYYY-MM-DD/
OUTPUTS_DIR = str(BASE_DIR / "outputs")
INCIDENT_SAVE_DIR = str(BASE_DIR / "outputs" / "incidents")
OUTPUT_DIR = INCIDENT_SAVE_DIR
LOGS_DIR = str(BASE_DIR / "logs")

os.makedirs(INCIDENT_SAVE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Display window settings
DISPLAY_WINDOW = True
WINDOW_TITLE = "Women Safety AI - Incident Feed"
DISPLAY_WIDTH = 960  # None to keep original size, or width in pixels
FRAME_INTERVAL = 1   # 1 = process every frame, 2 = process every other frame for performance

# ==========================================
# 1B. HARDWARE & PERFORMANCE OPTIMIZATIONS
# ==========================================
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    INFERENCE_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
    if not CUDA_AVAILABLE:
        # Optimize CPU intra-op thread allocation
        torch.set_num_threads(max(1, min(os.cpu_count() or 2, 4)))
except Exception:
    CUDA_AVAILABLE = False
    INFERENCE_DEVICE = "cpu"

# Decimated Inference Intervals (Avoid running every expensive deep-learning model on every frame)
PERSON_DETECTION_INTERVAL = 1      # Run person detector & ByteTrack every N frames
GENDER_INFERENCE_INTERVAL = 4      # Re-classify gender every N frames (cached per track ID)
POSE_INFERENCE_INTERVAL = 3        # Run YOLOv8-pose keypoints every N frames (cached per track ID)
CARRIED_OBJECTS_INTERVAL = 8       # Run carried object detection every N frames (snatching)
USE_THREADED_CAMERA = True         # Use VideoStream with latest-frame buffer to eliminate stream lag
INFERENCE_MAX_WIDTH = 640          # Downscale frame for inference if larger, preserving display frame



# ==========================================
# 2. INPUT SOURCE CONFIGURATION
# ==========================================
# Options: "webcam", "image", "video", "rtsp"
INPUT_SOURCE = "webcam"
INPUT_MODE = INPUT_SOURCE  # Alias for backward compatibility

# Webcam settings
CAMERA_INDEX = int(os.getenv("CAMERA_SOURCE", "0")) if os.getenv("CAMERA_SOURCE", "0").isdigit() else 0

# File path settings
INPUT_IMAGE_PATH = str(BASE_DIR / "test_image.png")
INPUT_VIDEO_PATH = str(BASE_DIR / "test_video.mp4")

# RTSP / CCTV network stream
RTSP_URL = os.getenv("CAMERA_SOURCE", "rtsp://127.0.0.1:8554/stream")
CAMERA_ROTATION = int(os.getenv("CAMERA_ROTATION", "0"))


# ==========================================
# 3. PERSON DETECTION & TRACKING
# ==========================================
PERSON_DETECTION_ENABLED = True
PERSON_MODEL_PATH = str(BASE_DIR / "models" / "person_detection" / "yolov8n.pt")
PERSON_CONFIDENCE = 0.40
PERSON_IOU_THRESHOLD = 0.45
PERSON_IMG_SIZE = 640

# Tracking settings (ByteTrack via Ultralytics)
TRACKING_ENABLED = True
TRACKER_TYPE = "bytetrack"


# ==========================================
# 4. INCIDENT DETECTION MODULES
# ==========================================
# Physical violence and struggle detection (RWF-2000 CNN-GRU model: best.pt)
VIOLENCE_DETECTION_ENABLED = True
VIOLENCE_MODEL_PATH = str(BASE_DIR / "models" / "violence_detection" / "best.pt")
VIOLENCE_NUM_FRAMES = 8
VIOLENCE_IMAGE_SIZE = 160
VIOLENCE_CONFIDENCE_THRESHOLD = 0.55
VIOLENCE_CHECK_INTERVAL_SEC = 0.4    # Run CNN-GRU inference on rolling buffer
VIOLENCE_CONFIRM_COUNT = 2           # Consecutive windows to confirm fight

# Fallback violence classifier (OpenAI CLIP) if best.pt is unavailable
CLIP_FALLBACK_ENABLED = True
VIOLENCE_MODEL_NAME = "ViT-B/32"

# Pose and fall detection
FALL_DETECTION_ENABLED = True
POSE_DETECTION_ENABLED = True
POSE_MODEL_PATH = str(BASE_DIR / "models" / "fall_detection" / "yolov8n-pose.pt")
FALL_SCORE_THRESHOLD = 0.55
FALL_CONFIRMATION_SECONDS = 3.0
FALL_ASPECT_RATIO_THRESHOLD = 1.3
MIN_POSE_CONFIDENCE = 0.25

# Interaction & behavioral analysis
INTERACTION_DETECTION_ENABLED = True
INTERACTION_ANALYSIS_ENABLED = INTERACTION_DETECTION_ENABLED
PROXIMITY_RATIO_THRESHOLD = 1.3
CROWD_DENSITY_THRESHOLD = 5

LOITERING_ENABLED = True
LOITERING_TIME_THRESHOLD_SEC = 15.0
LOITERING_RADIUS_PX = 60

FOLLOWING_ENABLED = True
FOLLOWING_MIN_DURATION_SEC = 4.0
FOLLOWING_DISTANCE_PX = 130

# Gender classification (working_m_&_F.pt YOLO model)
GENDER_CLASSIFICATION_ENABLED = True
GENDER_MODEL_PATH = str(BASE_DIR / "models" / "gender_classification" / "working_m_&_F.pt")
GENDER_CONFIDENCE_THRESHOLD = 0.35
GENDER_IOU_MATCH_THRESHOLD = 0.20
FACE_DETECTION_ENABLED = True
FACE_MIN_CONFIDENCE = 0.50
FACE_MIN_SIZE_PX = 30
FACE_MODEL_SELECTION = 1
GENDER_FALLBACK_MODEL_NAME = "rizvandwiki/gender-classification"

# Female Victim Highlighting Configuration
VICTIM_BOX_COLOR = (0, 0, 255)       # Red in BGR
VICTIM_BOX_THICKNESS = 3             # Bold box around identified female victim
VICTIM_LABEL_TEXT = "POTENTIAL WOMAN SAFETY INCIDENT"


# ==========================================
# 5. RISK ASSESSMENT & INCIDENT STATE MACHINE
# ==========================================
# Weights contributing to total incident risk score (0-100)
# (Restricted zone weight removed completely)
RISK_WEIGHTS = {
    "violence_detected": 85,        # Physical aggression / fight
    "female_attacked": 90,          # Direct physical struggle or assault
    "fall_confirmed": 80,           # Collapse followed by immobility
    "female_surrounded": 75,        # Multi-person encirclement
    "following_detected": 45,       # Sustained tailing
    "fighting_interaction": 60,     # Rapid hand striking motion
    "loitering_detected": 25,       # Prolonged lingering
    "crowd_dense": 20,              # High crowd density
}

RISK_THRESHOLD_LOW = 25
RISK_THRESHOLD_MEDIUM = 50
RISK_THRESHOLD_HIGH = 75

# Minimum time between re-raising incident state for ongoing events
ALERT_COOLDOWN_SECONDS = 30.0
EVENT_COOLDOWN_SECONDS = ALERT_COOLDOWN_SECONDS
QUALIFYING_INCIDENT_COOLDOWN_SECONDS = 30.0


# ==========================================
# 6. INCIDENT EVIDENCE & RECORDING
# ==========================================
SAVE_INCIDENT_SNAPSHOT = True
SAVE_INCIDENT_VIDEO = True
SAVE_INCIDENT_IMAGES = SAVE_INCIDENT_SNAPSHOT
EVIDENCE_RECORDING_ENABLED = SAVE_INCIDENT_VIDEO
EVIDENCE_PRE_EVENT_SEC = 8
EVIDENCE_POST_EVENT_SEC = 8


# ==========================================
# 7. ALERT DELIVERY CHANNELS & EMERGENCY CONTACTS
# ==========================================
# Selected primary delivery channel: "telegram", "sms", or "console"
ALERT_DELIVERY_CHANNEL = "telegram"

# Telegram settings (credentials read from .env)
TELEGRAM_ALERTS_ENABLED = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Emergency & trusted contacts
TRUSTED_CONTACT_PHONE = os.getenv("TRUSTED_CONTACT_PHONE")
EMERGENCY_CONTACT_PHONE = os.getenv("EMERGENCY_CONTACT_PHONE", "+917415297115")

# Emergency calling integration (Omnidimension / Voice AI)
CALL_ALERTS_ENABLED = True
OMNIDIM_API_KEY = os.getenv("OMNIDIM_API_KEY", "")
OMNIDIM_AGENT_ID = os.getenv("OMNIDIM_AGENT_ID", "")
ALERT_PHONE_NUMBER = TRUSTED_CONTACT_PHONE
CALL_ALERT_COOLDOWN_SEC = 120.0

# Official emergency service notification policy
# IMPORTANT: Direct 112 dialing is not automated by ordinary commercial messaging APIs.
# POLICE_NOTIFICATION_ENABLED provides verified public station lookup and human escalation contact info.
POLICE_NOTIFICATION_ENABLED = True
POLICE_STATION_NAME = os.getenv("POLICE_STATION_NAME", "Sector 17 Central Police Station")
POLICE_DISPATCH_HOTLINE = "112 / 1091 (Women Helpline)"
