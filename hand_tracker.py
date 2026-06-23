import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._draw_utils = mp.solutions.drawing_utils
        self.landmarks = None

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        self.landmarks = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.landmarks = hand.landmark
            if draw:
                self._draw_utils.draw_landmarks(
                    frame, hand, self._mp_hands.HAND_CONNECTIONS
                )

        return frame

    def get_landmark_position(self, frame, landmark_id):
        if not self.landmarks:
            return None
        h, w = frame.shape[:2]
        lm = self.landmarks[landmark_id]
        return int(lm.x * w), int(lm.y * h)

    def _is_finger_extended(self, tip_id, pip_id):
        if not self.landmarks:
            return False
        # Tip above PIP in image coordinates (y increases downward)
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

    def is_fist(self):
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        return bool(self.landmarks) and all(
            not self._is_finger_extended(tip, pip) for tip, pip in finger_pairs
        )
