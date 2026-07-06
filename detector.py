"""Real-time ISL alphabet detector using MediaPipe."""

import time

import cv2
import mediapipe as mp
from utils import draw_text
from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    WINDOW_NAME,
    QUIT_KEY,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FONT,
    GREEN,
    GRAY,
    LIGHT_GRAY,
    DETECTION_COLOR,
    UI_COLOR,
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

ISL_ONE_HAND = {
    (1, 0, 0, 0, 0): "A",
    (0, 1, 1, 1, 1): "B",
    (1, 1, 1, 1, 1): "C",
    (0, 1, 0, 0, 0): "D",
    (0, 0, 0, 0, 0): "E",
    (0, 0, 0, 0, 1): "I",
    (1, 1, 0, 0, 0): "L",
    (1, 1, 1, 0, 0): "U",
    (0, 1, 1, 0, 0): "V",
    (0, 1, 1, 1, 0): "W",
    (1, 0, 0, 0, 1): "Y",
}

# Two-hand signs (tuples of left and right finger states)
ISL_TWO_HAND = {
    ((1, 0, 0, 0, 0), (1, 0, 0, 0, 0)): "M",  # Both thumbs only
    ((0, 1, 1, 0, 0), (0, 1, 1, 0, 0)): "W",  # Both V shape
    ((0, 0, 0, 0, 1), (0, 0, 0, 0, 1)): "J",  # Both pinkies
    ((0, 1, 1, 1, 1), (0, 1, 1, 1, 1)): "BOTH B",  # Both hands open
}

def get_finger_states(landmarks, is_right: bool) -> tuple:
    """Determine which fingers are extended from hand landmarks."""
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    if is_right:
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    for i in range(1, 5):
        fingers.append(1 if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y else 0)

    return tuple(fingers)

def main():
    """Run the ISL detector loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    prev_time = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_states = []
        handedness = []

        if results.multi_hand_landmarks:
            for handLms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                is_right = handed.classification[0].label == "Right"
                lm = handLms.landmark
                fs = get_finger_states(lm, is_right)
                hand_states.append(fs)
                handedness.append(is_right)

                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        detected_text = None

        if len(hand_states) == 1:
            detected_text = ISL_ONE_HAND.get(hand_states[0])
        elif len(hand_states) == 2:
            left, right = hand_states
            if handedness[0]:
                left, right = hand_states[1], hand_states[0]
            detected_text = ISL_TWO_HAND.get((left, right))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        hand_count = len(hand_states)
        draw_text(img, f"Hands: {hand_count}", (10, 30), color=UI_COLOR)

        if detected_text:
            draw_text(img, f"ISL: {detected_text}", (50, 50), scale=1.2, color=DETECTION_COLOR, thickness=2)

        draw_text(img, f"FPS: {fps:.1f}", (10, img.shape[0] - 10), color=LIGHT_GRAY)

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) & 0xFF
        if key == QUIT_KEY:
            break
        elif key == ord('s'):
            cv2.imwrite("isl_screenshot.png", img)
            print("Screenshot saved")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
