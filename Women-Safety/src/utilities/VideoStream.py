"""
High-Performance Threaded Video Stream.
Continuously reads frames from camera / IP stream in a background thread and keeps
strictly the latest frame. Prevents frame buffer lag and keeps live display responsive.
"""

import time
import threading
from typing import Tuple, Optional, Union
import cv2
import numpy as np


class VideoStream:
    """
    Threaded video capture stream.
    Exposes standard OpenCV VideoCapture interface while decoupling frame grabbing
    from the main AI processing pipeline.
    """

    def __init__(self, source: Union[int, str], queue_size: int = 1):
        self.source = source
        self.capture = cv2.VideoCapture(source)
        
        # Optimize internal OpenCV buffer size to 1 where supported
        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.status: bool = False
        self.stopped: bool = False
        self.fps: float = 30.0

        if self.capture.isOpened():
            self.status, self.frame = self.capture.read()
            src_fps = self.capture.get(cv2.CAP_PROP_FPS)
            if src_fps and src_fps > 0 and src_fps < 120:
                self.fps = src_fps

        self.thread = threading.Thread(target=self._grab_loop, daemon=True, name="VideoStreamWorker")
        self.thread.start()

    def _grab_loop(self) -> None:
        """Background thread constantly updating the latest frame."""
        while not self.stopped:
            if not self.capture.isOpened():
                break
            status, frame = self.capture.read()
            if not status or frame is None:
                with self.lock:
                    self.status = False
                break
            with self.lock:
                self.status = status
                self.frame = frame
            # Tiny sleep to yield GIL and prevent 100% CPU core pinning on frame grabbing
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns (success, latest_frame). Returns a copy or reference under lock."""
        with self.lock:
            if self.frame is not None:
                return self.status, self.frame.copy()
            return self.status, None

    def isOpened(self) -> bool:
        return self.capture.isOpened() and not self.stopped

    def get(self, propId: int) -> float:
        return self.capture.get(propId)

    def set(self, propId: int, value: float) -> bool:
        return self.capture.set(propId, value)

    def release(self) -> None:
        """Gracefully stops worker thread and releases OpenCV camera handle."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()
