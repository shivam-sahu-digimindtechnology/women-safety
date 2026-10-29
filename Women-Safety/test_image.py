"""
Test Script for Single Image Processing.
Tests Person Detection, Face Detection, and Pose Estimation on an input image.

Usage:
    python test_image.py
    python test_image.py --image path/to/image.jpg --display
"""

import os
import sys
import argparse
import cv2
import config
from Person_Detection import detect_and_track_persons
from Face_Detection import find_face_in_person_crop
from Pose_Detection import detect_poses, detect_action
from Risk_Assessment import RiskAssessmentEngine
from Video_Processor import process_frame


def main():
    parser = argparse.ArgumentParser(description="Test Image Detection Pipeline")
    parser.add_argument("--image", type=str, default=config.INPUT_IMAGE_PATH, help="Path to test image")
    parser.add_argument("--display", action="store_true", help="Display the result window")
    args = parser.parse_args()

    image_path = args.image
    print(f"[Test Image] Testing image pipeline on: {image_path}")

    if not os.path.exists(image_path):
        print(f"[Error] Image not found at: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] cv2 failed to read image at: {image_path}")
        sys.exit(1)

    print(f"[Test Image] Image dimensions: {image.shape[1]}x{image.shape[0]}")

    # Process frame through full integrated pipeline
    risk_engine = RiskAssessmentEngine()
    annotated_output, metadata = process_frame(image, risk_engine, frame_idx=0)

    person_count = metadata.get("person_count", 0)
    risk_info = metadata.get("risk_info", {})

    print("\n" + "=" * 50)
    print("           TEST IMAGE DETECTION RESULTS")
    print("=" * 50)
    print(f" Total People Detected : {person_count}")
    print(f" Overall Risk Level    : {risk_info.get('level')} ({risk_info.get('score')}/100)")
    print(f" Primary Explanation   : {risk_info.get('summary')}")
    print("=" * 50)

    # Save output
    output_path = os.path.join(config.OUTPUT_DIR, "test_image_result.jpg")
    cv2.imwrite(output_path, annotated_output)
    # Also save to root for backward compatibility
    cv2.imwrite("test_image_result.jpg", annotated_output)
    print(f"\n[Test Image] Output saved successfully to:\n  - {output_path}\n  - test_image_result.jpg")

    if args.display:
        print("\n[Test Image] Showing output window. Press any key to close...")
        cv2.imshow("Test Image Result", annotated_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
