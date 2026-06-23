import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time ~29 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.")


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            # VIDEO mode uses timestamps for temporal tracking — far more stable
            # than IMAGE mode which re-detects from scratch every frame.
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self.landmarks = None

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        self.landmarks = None

        if result.hand_landmarks:
            self.landmarks = result.hand_landmarks[0]
            if draw:
                self._draw_landmarks(frame)

        return frame

    def _draw_landmarks(self, frame):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in self.landmarks]
        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (80, 200, 80), 1)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 128, 255), -1)

    def get_landmark_position(self, frame, landmark_id):
        if not self.landmarks:
            return None
        h, w = frame.shape[:2]
        lm = self.landmarks[landmark_id]
        return int(lm.x * w), int(lm.y * h)

    def _is_finger_extended(self, tip_id, pip_id):
        if not self.landmarks:
            return False
        return self.landmarks[tip_id].y < self.landmarks[pip_id].y

    def is_index_drawing(self):
        index_up = self._is_finger_extended(8, 6)
        middle_up = self._is_finger_extended(12, 10)
        return index_up and not middle_up

    def is_two_fingers_up(self):
        index_up = self._is_finger_extended(8, 6)
        middle_up = self._is_finger_extended(12, 10)
        ring_up = self._is_finger_extended(16, 14)
        return index_up and middle_up and not ring_up

    def is_open_hand(self):
        margin = 0.07
        tip_pip_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        return bool(self.landmarks) and all(
            self.landmarks[tip].y < self.landmarks[pip].y - margin
            for tip, pip in tip_pip_pairs
        )
