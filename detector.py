import cv2
import mediapipe as mp
import numpy as np

# Init MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Single hand static signs
ISL_ONE_HAND = {
    (1, 0, 0, 0, 0): "A",
    (0, 1, 1, 1, 1): "B",
    (1, 1, 1, 1, 1): "C",
    (0, 1, 0, 0, 0): "D",
    (0, 0, 0, 0, 1): "I",
    (1, 1, 0, 0, 0): "L",
    (0, 1, 1, 0, 0): "V",
    (0, 1, 1, 1, 0): "W",
    (1, 0, 0, 0, 1): "Y"
}

# Two-hand signs (tuples of left and right finger states)
ISL_TWO_HAND = {
    ((1, 0, 0, 0, 0), (1, 0, 0, 0, 0)): "M",  # Both thumbs only
    ((0, 1, 1, 0, 0), (0, 1, 1, 0, 0)): "W",  # Both V shape
    ((0, 0, 0, 0, 1), (0, 0, 0, 0, 1)): "J",  # Both pinkies
    ((0, 1, 1, 1, 1), (0, 1, 1, 1, 1)): "BOTH B",  # Both hands open
}

def get_finger_states(lm, is_right):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if is_right:
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if lm[tip_ids[i]].y < lm[tip_ids[i]-2].y else 0)

    return tuple(fingers)

# Webcam
cap = cv2.VideoCapture(0)

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

    detected = None

    # One hand
    if len(hand_states) == 1:
        if hand_states[0] in ISL_ONE_HAND:
            detected = ISL_ONE_HAND[hand_states[0]]

    # Two hands
    elif len(hand_states) == 2:
        # Ensure left, right order
        if handedness[0]:
            left, right = hand_states[1], hand_states[0]
        else:
            left, right = hand_states[0], hand_states[1]

        if (left, right) in ISL_TWO_HAND:
            detected = ISL_TWO_HAND[(left, right)]

    # Display
    if detected:
        cv2.putText(img, f"ISL: {detected}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Two-Hand ISL Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
