import cv2
import numpy as np
from hand_tracker import HandTracker

COLORS = [
    (255, 255, 255),  # White
    (255, 80, 80),    # Blue
    (80, 255, 80),    # Green
    (80, 80, 255),    # Red
    (80, 255, 255),   # Yellow
    (255, 80, 255),   # Magenta
]

BRUSH_THICKNESS = 8
COLOR_SWITCH_COOLDOWN_FRAMES = 25


def overlay_canvas(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    return cv2.add(background, canvas)


def draw_ui(frame, color, color_index, gesture_label):
    h = frame.shape[0]
    cv2.circle(frame, (35, 35), 20, color, -1)
    cv2.putText(
        frame, f"{color_index + 1}/{len(COLORS)}", (65, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.putText(
        frame, f"Gesture: {gesture_label}",
        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    cv2.putText(
        frame, "1-finger: Draw | 2-finger: Color | Fist: Erase | Q: Quit",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    tracker = HandTracker()

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera.")
        cap.release()
        return

    canvas = np.zeros_like(cv2.flip(first_frame, 1))
    color_index = 0
    prev_point = None
    color_switch_cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame)

        if color_switch_cooldown > 0:
            color_switch_cooldown -= 1

        current_color = COLORS[color_index]
        index_tip = tracker.get_landmark_position(frame, 8)

        gesture_label = "none"
        if tracker.is_open_hand():
            gesture_label = "erase"
            prev_point = None
            canvas[:] = 0

        elif tracker.is_two_fingers_up() and color_switch_cooldown == 0:
            gesture_label = "color"
            prev_point = None
            color_index = (color_index + 1) % len(COLORS)
            color_switch_cooldown = COLOR_SWITCH_COOLDOWN_FRAMES

        elif tracker.is_index_drawing() and index_tip:
            gesture_label = "draw"
            if prev_point:
                cv2.line(canvas, prev_point, index_tip, current_color, BRUSH_THICKNESS)
            prev_point = index_tip

        else:
            prev_point = None

        output = overlay_canvas(frame, canvas)
        draw_ui(output, current_color, color_index, gesture_label)
        cv2.imshow("Air Drawing", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
