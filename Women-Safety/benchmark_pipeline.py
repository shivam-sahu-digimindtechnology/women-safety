"""
Benchmark script to measure inference times and FPS of the pipeline.
"""
import time
import os
import sys
import cv2
import numpy as np

import config
config.DISPLAY_WINDOW = False

from test_video import generate_sample_video
from Risk_Assessment import RiskAssessmentEngine
from Video_Processor import process_frame

def run_benchmark():
    sample_video = "benchmark_sample.mp4"
    if not os.path.exists(sample_video):
        generate_sample_video(sample_video, duration_sec=3, fps=15)

    cap = cv2.VideoCapture(sample_video)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    print(f"[Benchmark] Loaded {total_frames} frames for evaluation.", flush=True)

    risk_engine = RiskAssessmentEngine()

    # Warmup first frame
    print("[Benchmark] Warming up models...", flush=True)
    _ = process_frame(frames[0].copy(), risk_engine)
    print("[Benchmark] Models warmed up.", flush=True)

    start_time = time.time()
    for idx, f in enumerate(frames):
        _ = process_frame(f.copy(), risk_engine)
    end_time = time.time()

    total_time = end_time - start_time
    fps = total_frames / total_time if total_time > 0 else 0
    print("\n" + "="*50, flush=True)
    print(f"BENCHMARK RESULTS ({total_frames} frames):", flush=True)
    print(f"Total Processing Time : {total_time:.2f} seconds", flush=True)
    print(f"Average Processing FPS: {fps:.2f} FPS", flush=True)
    print(f"Average Latency/Frame : {(total_time/total_frames)*1000:.1f} ms", flush=True)
    print("="*50 + "\n", flush=True)

if __name__ == "__main__":
    run_benchmark()
