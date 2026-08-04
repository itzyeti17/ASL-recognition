"""Small webcam preview used as the input for the ASL recognizer."""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
except ImportError:
    print(
        "A project dependency is missing. Run ./setup.sh from WSL.",
        file=sys.stderr,
    )
    raise SystemExit(1)


MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# MediaPipe's 21 hand points, connected in the same groups as the hand bones.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Preview a webcam for ASL recognition.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="camera index to open (default: 0)",
    )
    return parser.parse_args()


def open_camera(index):
    # Camera drivers differ across Windows systems, so try the two native
    # backends and finally let OpenCV choose. Release failed handles before
    # moving to the next backend.
    if os.name == "nt":
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            camera = cv2.VideoCapture(index, backend)
            if camera.isOpened():
                return camera
            camera.release()
        return camera
    return cv2.VideoCapture(index)


def create_hand_landmarker():
    if not MODEL_PATH.exists():
        print("Hand model is missing. Run ./setup.sh from WSL.", file=sys.stderr)
        return None

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def draw_hands(frame, result, mirrored=False):
    height, width = frame.shape[:2]
    for hand_number, landmarks in enumerate(result.hand_landmarks):
        points = [
            (max(0, min(width - 1, int((1.0 - point.x if mirrored else point.x) * width))),
             max(0, min(height - 1, int(point.y * height))))
            for point in landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (80, 220, 80), 2)
        for point in points:
            cv2.circle(frame, point, 4, (40, 80, 255), -1)

        if hand_number < len(result.handedness) and result.handedness[hand_number]:
            label = result.handedness[hand_number][0].category_name
            x, y = points[0]
            cv2.putText(
                frame, label, (x, max(25, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )


def main():
    args = parse_args()
    landmarker = create_hand_landmarker()
    if landmarker is None:
        return 1
    camera = open_camera(args.camera)

    if not camera.isOpened():
        camera.release()
        landmarker.close()
        print(f"Error: camera {args.camera} could not be opened.", file=sys.stderr)
        if sys.platform.startswith("linux") and "microsoft" in os.uname().release.lower():
            print("In WSL, start the app with ./run.sh.", file=sys.stderr)
        return 1

    print("Webcam started. Press Q or Esc to close it.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: could not read a frame from the webcam.", file=sys.stderr)
                return 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Analyze the real camera orientation so handedness is correct,
            # then mirror only what the user sees.
            frame = cv2.flip(frame, 1)
            draw_hands(frame, result, mirrored=True)

            cv2.putText(
                frame, "Hand tracking", (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("ASL Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
