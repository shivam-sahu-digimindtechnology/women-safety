"""
Main Entry Point for Women Safety AI System.
Interactive Console Launcher with Input Source Selection:
- Option to select Live Webcam (Built-in or USB)
- Option to select Video File (Browse via Windows Explorer or sample)
- Option to select Image File (Browse via Windows Explorer or sample)
- Option to connect to RTSP / IP Camera stream

Usage:
    python main.py                  (Interactive menu to choose Video, Image, or Camera)
    python main.py --mode webcam    (Direct webcam execution)
    python main.py --mode video     (Direct video execution)
    python main.py --mode image     (Direct image execution)
    python main.py --mode rtsp      (Direct network stream execution)
"""

import os
import sys
import argparse
from pathlib import Path

import config
from Video_Processor import run_pipeline


def print_banner():
    print("\n" + "=" * 65)
    print(f"      {config.APP_NAME} - Real-Time Threat & Safety Monitoring")
    print("=" * 65)
    mode_name = config.INPUT_MODE.upper()
    print(f" [Active Mode]           : {mode_name}")
    if config.INPUT_MODE == "webcam":
        print(f" [Camera Index]          : {config.CAMERA_INDEX}")
    elif config.INPUT_MODE == "image":
        print(f" [Image Path]            : {config.INPUT_IMAGE_PATH}")
    elif config.INPUT_MODE == "video":
        print(f" [Video Path]            : {config.INPUT_VIDEO_PATH}")
    elif config.INPUT_MODE == "rtsp":
        print(f" [Stream URL]            : {config.RTSP_URL}")

    print(f" [Test Mode]             : {'SIMULATED (TEST_MODE=True)' if config.TEST_MODE else 'LIVE PRODUCTION'}")
    print(f" [Alerts Enabled]        : {'ENABLED' if config.ALERTS_ENABLED else 'DISABLED (Local logs only)'}")
    print(f" [Person Detection]      : {'ENABLED' if config.PERSON_DETECTION_ENABLED else 'DISABLED'}")
    print(f" [ByteTrack Tracking]    : {'ENABLED' if config.TRACKING_ENABLED else 'DISABLED'}")
    print(f" [Pose & Fall Detection] : {'ENABLED' if config.POSE_DETECTION_ENABLED else 'DISABLED'}")
    print(f" [Violence Detection]    : {'ENABLED' if config.VIOLENCE_DETECTION_ENABLED else 'DISABLED'}")
    print(f" [Evidence Directory]    : {config.OUTPUT_DIR}")
    print("=" * 65)
    print(" Press 'q' in the video window at any time to quit.")
    print("=" * 65 + "\n")


def browse_file_dialog(file_types, title="Select File") -> str:
    """Opens a native Windows file browser dialog so the user can easily click their file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        selected_path = filedialog.askopenfilename(
            title=title,
            filetypes=file_types,
            initialdir=str(config.BASE_DIR)
        )
        root.destroy()
        return selected_path.strip()
    except Exception as e:
        print(f"[Notice] File dialog not available: {e}. You can type path manually.")
        return ""


def prompt_user_input_source():
    """
    Presents an interactive menu to choose between Webcam, Video, Image, and RTSP stream.
    Includes one-click file browsing via Windows File Explorer.
    """
    print("\n" + "=" * 65)
    print("       WOMEN SAFETY AI — CHOOSE INPUT SOURCE")
    print("=" * 65)
    print(" Select what you want to run:")
    print("   [1] Live Webcam (Laptop or USB Camera)")
    print("   [2] Video File (Upload / browse MP4, AVI, MOV)")
    print("   [3] Image File (Upload / browse JPG, PNG)")
    print("   [4] RTSP / IP Camera Network Stream (URL)")
    print("   [5] Exit")
    print("-" * 65)

    choice = input(" Enter your choice [1-5] (Press Enter for Webcam): ").strip()

    # --- 1. WEBCAM ---
    if choice in ("", "1"):
        config.INPUT_MODE = "webcam"
        config.INPUT_SOURCE = "webcam"
        print("\n [Camera Selection]")
        print("   [0] Default Laptop Webcam")
        print("   [1] External USB Camera")
        cam_input = input(f" Select camera [0 or 1] (Default: {config.CAMERA_INDEX}): ").strip()
        if cam_input.isdigit():
            config.CAMERA_INDEX = int(cam_input)
        print(f" -> Selected Camera: {config.CAMERA_INDEX}")

    # --- 2. VIDEO FILE ---
    elif choice == "2":
        config.INPUT_MODE = "video"
        config.INPUT_SOURCE = "video"
        print("\n [Video File Selection]")
        print("   [1] Browse from Computer (Opens Windows File Explorer)")
        print("   [2] Use Sample Video (test_video.mp4)")
        print("   [3] Type Path Manually")
        sub_choice = input(" Select option [1-3] (Default: 1): ").strip()

        video_path = ""
        if sub_choice in ("", "1"):
            print(" Opening file browser... Please click on your video file.")
            video_path = browse_file_dialog(
                [("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")],
                title="Select Video File for Women Safety AI"
            )
            if video_path:
                print(f" -> Selected Video: {video_path}")

        elif sub_choice == "2":
            video_path = config.INPUT_VIDEO_PATH
            print(f" -> Using sample video: {video_path}")

        if not video_path:
            # Fallback to manual prompt if dialog was canceled or option 3 was picked
            manual_in = input(f" Enter video path (Press Enter for '{os.path.basename(config.INPUT_VIDEO_PATH)}'): ").strip()
            video_path = manual_in.strip("\"'") if manual_in else config.INPUT_VIDEO_PATH

        # Auto-generate sample video if the target does not exist
        if not os.path.exists(video_path):
            if video_path == config.INPUT_VIDEO_PATH:
                from test_video import generate_sample_video
                print(f"[Notice] Sample video '{video_path}' not found. Generating a 3-second sample...")
                generate_sample_video(video_path, duration_sec=3, fps=15)
            else:
                print(f"[Warning] File not found: '{video_path}'. Falling back to sample video.")
                video_path = config.INPUT_VIDEO_PATH

        config.INPUT_VIDEO_PATH = video_path

    # --- 3. IMAGE FILE ---
    elif choice == "3":
        config.INPUT_MODE = "image"
        config.INPUT_SOURCE = "image"
        print("\n [Image File Selection]")
        print("   [1] Browse from Computer (Opens Windows File Explorer)")
        print("   [2] Use Sample Image (test_image.png)")
        print("   [3] Type Path Manually")
        sub_choice = input(" Select option [1-3] (Default: 1): ").strip()

        image_path = ""
        if sub_choice in ("", "1"):
            print(" Opening file browser... Please click on your image file.")
            image_path = browse_file_dialog(
                [("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")],
                title="Select Image File for Women Safety AI"
            )
            if image_path:
                print(f" -> Selected Image: {image_path}")

        elif sub_choice == "2":
            image_path = config.INPUT_IMAGE_PATH
            print(f" -> Using sample image: {image_path}")

        if not image_path:
            manual_in = input(f" Enter image path (Press Enter for '{os.path.basename(config.INPUT_IMAGE_PATH)}'): ").strip()
            image_path = manual_in.strip("\"'") if manual_in else config.INPUT_IMAGE_PATH

        if not os.path.exists(image_path):
            print(f"[Warning] File not found: '{image_path}'. Falling back to default '{config.INPUT_IMAGE_PATH}'.")
            image_path = config.INPUT_IMAGE_PATH

        config.INPUT_IMAGE_PATH = image_path

    # --- 4. RTSP STREAM ---
    elif choice == "4":
        config.INPUT_MODE = "rtsp"
        config.INPUT_SOURCE = "rtsp"
        default_url = config.RTSP_URL
        print(f"\n [RTSP / IP Camera Stream]")
        stream_url = input(f" Enter stream URL (Default: {default_url}): ").strip()
        if stream_url:
            config.RTSP_URL = stream_url.strip("\"'")

    # --- 5. EXIT ---
    elif choice == "5":
        print("\nExiting. Goodbye!")
        sys.exit(0)

    else:
        print(f"\n[Notice] Unrecognized choice '{choice}'. Defaulting to Webcam mode.")
        config.INPUT_MODE = "webcam"
        config.INPUT_SOURCE = "webcam"


def main():
    parser = argparse.ArgumentParser(description=f"{config.APP_NAME} Pipeline Launcher")
    parser.add_argument(
        "--mode",
        choices=["webcam", "image", "video", "rtsp"],
        default=None,
        help="Input mode to run directly without interactive menu"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source: camera index (0, 1), video/image path, or stream URL"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI window (useful for headless servers or automated scripts)"
    )

    args = parser.parse_args()

    if args.no_display:
        config.DISPLAY_WINDOW = False

    # If no mode flag was passed in CLI, prompt user interactively
    if args.mode is None:
        try:
            prompt_user_input_source()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
    else:
        config.INPUT_MODE = args.mode
        config.INPUT_SOURCE = args.mode

    # Handle --source override
    if args.source:
        source_clean = args.source.strip("\"'")
        if config.INPUT_MODE == "webcam" and source_clean.isdigit():
            config.CAMERA_INDEX = int(source_clean)
        elif config.INPUT_MODE == "image":
            config.INPUT_IMAGE_PATH = source_clean
        elif config.INPUT_MODE == "video":
            config.INPUT_VIDEO_PATH = source_clean
        elif config.INPUT_MODE == "rtsp":
            config.RTSP_URL = source_clean

    print_banner()

    try:
        run_pipeline(mode=config.INPUT_MODE)
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user. Exiting cleanly.")
    except Exception as e:
        print(f"\n[Error] Pipeline execution error: {e}")
        if config.DEBUG_MODE:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
