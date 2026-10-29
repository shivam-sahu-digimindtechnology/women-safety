"""Saves a local video clip around a detected event: PRE_EVENT_SECONDS of
footage from before the trigger (pulled from a continuously maintained rolling
buffer) plus POST_EVENT_SECONDS recorded after it. Clips are saved locally
under EVIDENCE_DIR only -- never sent over Telegram or any other channel."""

import os
import time
import threading
from collections import deque
import cv2

PRE_EVENT_SECONDS = 10
POST_EVENT_SECONDS = 10
EVIDENCE_DIR = "evidence"
# Avoid overlapping/duplicate clips if multiple alerts fire in quick succession.
RECORD_COOLDOWN_SECONDS = 15

os.makedirs(EVIDENCE_DIR, exist_ok=True)

_buffer_lock = threading.Lock()
_buffer = deque()  # (timestamp, frame)

_recording_lock = threading.Lock()
_is_recording = False
_last_capture_time = 0.0


def add_frame(frame):
    """Call this on every displayed frame to keep the rolling pre-event buffer
    current. Cheap: just appends and trims, no encoding happens here."""
    now = time.time()
    with _buffer_lock:
        _buffer.append((now, frame.copy()))
        cutoff = now - PRE_EVENT_SECONDS
        while _buffer and _buffer[0][0] < cutoff:
            _buffer.popleft()


def capture_event(reason: str):
    """Triggers saving a clip covering PRE_EVENT_SECONDS before this call and
    POST_EVENT_SECONDS after. Non-blocking: recording/encoding happens on a
    background thread so it never stalls the detection loop that calls this."""
    global _is_recording, _last_capture_time
    now = time.time()
    with _recording_lock:
        if _is_recording or (now - _last_capture_time) < RECORD_COOLDOWN_SECONDS:
            return
        _is_recording = True
        _last_capture_time = now

    with _buffer_lock:
        pre_frames = list(_buffer)

    threading.Thread(target=_record_post_and_save, args=(reason, pre_frames), daemon=True).start()


def _record_post_and_save(reason: str, pre_frames: list):
    global _is_recording
    try:
        post_frames = []
        last_ts = pre_frames[-1][0] if pre_frames else time.time()
        deadline = time.time() + POST_EVENT_SECONDS

        while time.time() < deadline:
            with _buffer_lock:
                new_frames = [f for f in _buffer if f[0] > last_ts]
            if new_frames:
                post_frames.extend(new_frames)
                last_ts = new_frames[-1][0]
            time.sleep(0.05)

        _save_clip(reason, pre_frames + post_frames)
    finally:
        _is_recording = False


def _save_clip(reason: str, frames: list):
    if not frames:
        return

    timestamps = [t for t, _ in frames]
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
    fps = max(1.0, len(frames) / duration) if duration > 0 else 10.0

    height, width = frames[0][1].shape[:2]
    safe_reason = "".join(c if c.isalnum() else "_" for c in reason)[:40]
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_reason}.mp4"
    path = os.path.join(EVIDENCE_DIR, filename)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for _, frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Evidence clip saved: {path}")
