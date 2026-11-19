"""Small webcam preview used as the input for the ASL recognizer."""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

try:
    import cv2
    import joblib
    import mediapipe as mp
    import numpy as np
except ImportError:
    print(
        "A project dependency is missing. Run ./setup.sh from WSL.",
        file=sys.stderr,
    )
    raise SystemExit(1)

from data_pipeline import normalize_landmarks


PROJECT_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = PROJECT_DIR / "models" / "asl_classifier.joblib"

BACKGROUND = (24, 20, 18)
PANEL = (38, 32, 28)
ACCENT = (255, 190, 70)
PREDICTION_COLOR = (80, 230, 255)
TEXT = (245, 245, 245)
MUTED_TEXT = (180, 180, 180)

# MediaPipe's 21 hand points, connected in the same groups as the hand bones.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview a webcam for ASL recognition."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="camera index to open (default: 0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="minimum confidence for displaying a prediction (default: 0.65)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    return args


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
    if not HAND_MODEL_PATH.exists():
        print("Hand model is missing. Run ./setup.sh from WSL.", file=sys.stderr)
        return None

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def load_classifier():
    if not CLASSIFIER_PATH.exists():
        print(
            "ASL classifier is missing; showing hand tracking only. "
            "Run: python train_model.py",
            file=sys.stderr,
        )
        return None

    try:
        artifact = joblib.load(CLASSIFIER_PATH)
        classifier = artifact["model"] if isinstance(artifact, dict) else artifact
        if not hasattr(classifier, "predict_proba"):
            raise ValueError("saved classifier does not support probability estimates")
        return classifier
    except Exception as error:
        print(
            f"Could not load ASL classifier ({error}); showing hand tracking only.",
            file=sys.stderr,
        )
        return None


class PredictionSmoother:
    """Stabilize predictions by voting over each hand's recent frames."""

    def __init__(self, window_size=7, minimum_frames=3):
        self.window_size = window_size
        self.minimum_frames = minimum_frames
        self.histories = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(self, hand_key, prediction, confidence):
        history = self.histories[hand_key]
        history.append((prediction, confidence))
        if len(history) < self.minimum_frames:
            return None

        winner, votes = Counter(label for label, _ in history).most_common(1)[0]
        required_votes = len(history) // 2 + 1
        if votes < required_votes:
            return None

        winner_confidences = [
            score for label, score in history if label == winner
        ]
        return winner, float(np.mean(winner_confidences))

    def retain(self, active_hand_keys):
        for hand_key in set(self.histories) - active_hand_keys:
            del self.histories[hand_key]


def classify_hand(classifier, landmarks):
    coordinates = [(point.x, point.y, point.z) for point in landmarks]
    features = normalize_landmarks(coordinates).reshape(1, -1)
    probabilities = classifier.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    return str(classifier.classes_[best_index]), float(probabilities[best_index])


def draw_translucent_panel(frame, top_left, bottom_right, color=PANEL, alpha=0.82):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def fit_text(text, max_width, font, scale, thickness):
    """Keep the newest part of a line visible within the available width."""
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text

    shortened = text
    while shortened:
        candidate = f"...{shortened}"
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            return candidate
        shortened = shortened[1:]
    return "..."


def draw_interface(
    frame,
    classifier_ready,
    captured_text,
    active_prediction,
    status_message="",
):
    height, width = frame.shape[:2]
    header_height = min(76, height // 4)
    footer_height = min(118, height // 3)

    draw_translucent_panel(frame, (0, 0), (width, header_height), BACKGROUND)
    cv2.putText(
        frame,
        "ASL RECOGNIZER",
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        TEXT,
        2,
        cv2.LINE_AA,
    )
    subtitle = (
        f"Current: {active_prediction[0]}  {active_prediction[1]:.0%}"
        if active_prediction is not None
        else "Hold a static ASL letter in view"
    )
    cv2.putText(
        frame,
        subtitle,
        (18, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        PREDICTION_COLOR if active_prediction is not None else MUTED_TEXT,
        1,
        cv2.LINE_AA,
    )

    model_status = "MODEL READY" if classifier_ready else "TRACKING ONLY"
    status_size = cv2.getTextSize(
        model_status, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1
    )[0]
    cv2.putText(
        frame,
        model_status,
        (max(18, width - status_size[0] - 18), 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        ACCENT if classifier_ready else (100, 150, 255),
        1,
        cv2.LINE_AA,
    )

    footer_top = height - footer_height
    draw_translucent_panel(frame, (0, footer_top), (width, height), BACKGROUND)
    cv2.putText(
        frame,
        "CAPTURED TEXT",
        (18, footer_top + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        ACCENT,
        1,
        cv2.LINE_AA,
    )
    displayed_text = fit_text(
        captured_text or "_",
        max(20, width - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        2,
    )
    cv2.putText(
        frame,
        displayed_text,
        (18, footer_top + 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        TEXT if captured_text else MUTED_TEXT,
        2,
        cv2.LINE_AA,
    )
    controls = "ENTER capture   SPACE word gap   BACKSPACE undo   C clear   Q quit"
    controls = fit_text(
        controls, max(20, width - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    )
    cv2.putText(
        frame,
        controls,
        (18, footer_top + 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        MUTED_TEXT,
        1,
        cv2.LINE_AA,
    )

    if status_message:
        message_size = cv2.getTextSize(
            status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )[0]
        message_x = max(12, (width - message_size[0]) // 2)
        message_y = max(header_height + 35, footer_top - 18)
        draw_translucent_panel(
            frame,
            (max(4, message_x - 10), message_y - 25),
            (min(width - 4, message_x + message_size[0] + 10), message_y + 8),
            PANEL,
            0.9,
        )
        cv2.putText(
            frame,
            status_message,
            (message_x, message_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            PREDICTION_COLOR,
            1,
            cv2.LINE_AA,
        )


def handle_capture_key(key, captured_text, active_prediction):
    """Apply one keyboard action and return text, feedback, and quit state."""
    if key in (ord("q"), ord("Q"), 27):
        return captured_text, "", True
    if key in (10, 13):
        if active_prediction is None:
            return captured_text, "No confident letter to capture", False
        letter = active_prediction[0]
        return captured_text + letter, f"Captured {letter}", False
    if key == 32:
        if captured_text and not captured_text.endswith(" "):
            return captured_text + " ", "Added word gap", False
        return captured_text, "", False
    if key in (8, 127):
        if captured_text:
            removed = captured_text[-1]
            return captured_text[:-1], f"Removed {removed!r}", False
        return captured_text, "", False
    if key in (ord("c"), ord("C")):
        return "", "Captured text cleared", False
    return captured_text, "", False


def draw_hands(
    frame,
    result,
    mirrored=False,
    classifier=None,
    smoother=None,
    confidence_threshold=0.65,
):
    height, width = frame.shape[:2]
    active_hand_keys = set()
    visible_predictions = []
    for hand_number, landmarks in enumerate(result.hand_landmarks):
        points = [
            (
                max(
                    0,
                    min(
                        width - 1,
                        int((1.0 - point.x if mirrored else point.x) * width),
                    ),
                ),
                max(0, min(height - 1, int(point.y * height))),
            )
            for point in landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (80, 220, 80), 2)
        for point in points:
            cv2.circle(frame, point, 4, (40, 80, 255), -1)

        min_x = max(0, min(point[0] for point in points) - 14)
        max_x = min(width - 1, max(point[0] for point in points) + 14)
        min_y = max(0, min(point[1] for point in points) - 14)
        max_y = min(height - 1, max(point[1] for point in points) + 14)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), ACCENT, 2)

        handedness = "Hand"
        if hand_number < len(result.handedness) and result.handedness[hand_number]:
            handedness = result.handedness[hand_number][0].category_name

        badge_width = max(76, 14 + len(handedness) * 12)
        badge_top = max(0, min_y - 27)
        cv2.rectangle(
            frame,
            (min_x, badge_top),
            (min(width - 1, min_x + badge_width), min_y),
            ACCENT,
            -1,
        )
        cv2.putText(
            frame,
            handedness,
            (min_x + 7, max(17, min_y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            BACKGROUND,
            1,
            cv2.LINE_AA,
        )

        if classifier is not None and smoother is not None:
            hand_key = f"{handedness}-{hand_number}"
            active_hand_keys.add(hand_key)
            prediction, confidence = classify_hand(classifier, landmarks)
            smoothed = smoother.update(hand_key, prediction, confidence)
            if smoothed is not None:
                prediction, confidence = smoothed
                if confidence >= confidence_threshold:
                    visible_predictions.append((prediction, confidence))
                    cv2.putText(
                        frame,
                        f"{prediction} {confidence:.0%}",
                        (min_x, min(height - 10, max_y + 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

    if smoother is not None:
        smoother.retain(active_hand_keys)

    if not visible_predictions:
        return None
    return max(visible_predictions, key=lambda item: item[1])


def main():
    args = parse_args()
    landmarker = create_hand_landmarker()
    if landmarker is None:
        return 1
    classifier = load_classifier()
    smoother = PredictionSmoother() if classifier is not None else None
    camera = open_camera(args.camera)
    captured_text = ""
    status_message = ""
    status_until = 0.0

    if not camera.isOpened():
        camera.release()
        landmarker.close()
        print(f"Error: camera {args.camera} could not be opened.", file=sys.stderr)
        if (
            sys.platform.startswith("linux")
            and "microsoft" in os.uname().release.lower()
        ):
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
            active_prediction = draw_hands(
                frame,
                result,
                mirrored=True,
                classifier=classifier,
                smoother=smoother,
                confidence_threshold=args.confidence,
            )

            if time.monotonic() >= status_until:
                status_message = ""
            draw_interface(
                frame,
                classifier is not None,
                captured_text,
                active_prediction,
                status_message,
            )
            cv2.imshow("ASL Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            captured_text, feedback, should_quit = handle_capture_key(
                key, captured_text, active_prediction
            )
            if should_quit:
                return 0
            if feedback:
                status_message = feedback
                status_until = time.monotonic() + 1.5
    except KeyboardInterrupt:
        return 0
    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
