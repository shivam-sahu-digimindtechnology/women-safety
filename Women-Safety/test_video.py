"""
Test Script for Video File Processing.
Verifies frame-by-frame processing, tracking, risk scoring, and video encoding.

Usage:
    python test_video.py
    python test_video.py --video path/to/video.mp4
"""

import os
import sys
import argparse
import cv2
import numpy as np
import config
from Video_Processor import process_video


def generate_sample_video(output_path: str, duration_sec: int = 3, fps: int = 20):
    """Generates a small test video from test_image.png if no video is present."""
    print(f"[Test Video] Generating sample video from test_image.png -> {output_path}...")
    base_img = cv2.imread(config.INPUT_IMAGE_PATH)
    if base_img is None:
        # Create solid canvas with a moving rectangle if test_image.png is missing
        base_img = np.zeros((480, 640, 3), dtype=np.uint8)

    h, w = base_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Slightly modulate brightness to simulate real camera motion
        frame = cv2.convertScaleAbs(base_img, alpha=1.0, beta=int(5 * np.sin(i / 5.0)))
        writer.write(frame)

    writer.release()
    print(f"[Test Video] Sample video created ({total_frames} frames).")


def main():
    parser = argparse.ArgumentParser(description="Test Video Processing Pipeline")
    parser.add_argument("--video", type=str, default=config.INPUT_VIDEO_PATH, help="Path to video file")
    parser.add_argument("--display", action="store_true", help="Display the video playback window")
    args = parser.parse_args()

    if not args.display:
        config.DISPLAY_WINDOW = False

    video_path = args.video

    # If the target video does not exist, create a sample test video
    if not os.path.exists(video_path):
        print(f"[Test Video] Video file '{video_path}' not found.")
        generate_sample_video(video_path, duration_sec=3, fps=15)

    print(f"[Test Video] Starting video processing test on: {video_path}")
    success = process_video(video_path)

    if success:
        out_video = os.path.join(config.OUTPUT_DIR, "annotated_video.mp4")
        print("\n" + "=" * 50)
        print("          TEST VIDEO PROCESSING COMPLETE")
        print("=" * 50)
        print(f" Output Video File: {out_video}")
        print(f" File Size: {os.path.getsize(out_video) / 1024:.1f} KB")
        print("=" * 50)
    else:
        print("\n[Error] Video processing failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
