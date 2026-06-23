import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ---------------------------------
# SETTINGS
# ---------------------------------

pyautogui.FAILSAFE = True

frameR = 150
alpha = 0.25
dead_zone = 10

# ---------------------------------
# CAMERA
# ---------------------------------

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

# ---------------------------------
# MEDIAPIPE
# ---------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

draw = mp.solutions.drawing_utils

# ---------------------------------
# VARIABLES
# ---------------------------------

current_x = 0
current_y = 0

dragging = False
pinch_start_time = 0

last_scroll_time = 0

prev_time = time.time()

# ---------------------------------
# MAIN LOOP
# ---------------------------------

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    cv2.rectangle(
        img,
        (frameR, frameR),
        (w - frameR, h - frameR),
        (255, 0, 255),
        2
    )

    current_time = time.time()

    fps = 1 / max(current_time - prev_time, 0.0001)

    prev_time = current_time

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        draw.draw_landmarks(
            img,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # --------------------------
        # LANDMARKS
        # --------------------------

        x1 = int(hand.landmark[8].x * w)
        y1 = int(hand.landmark[8].y * h)

        x2 = int(hand.landmark[12].x * w)
        y2 = int(hand.landmark[12].y * h)

        # --------------------------
        # FINGER STATES
        # --------------------------

        index_up = hand.landmark[8].y < hand.landmark[6].y
        middle_up = hand.landmark[12].y < hand.landmark[10].y
        ring_up = hand.landmark[16].y < hand.landmark[14].y
        pinky_up = hand.landmark[20].y < hand.landmark[18].y

        finger_count = sum([
            index_up,
            middle_up,
            ring_up,
            pinky_up
        ])

        # --------------------------
        # CURSOR DOT
        # --------------------------

        cv2.circle(
            img,
            (x1, y1),
            10,
            (0, 255, 0),
            cv2.FILLED
        )

        # --------------------------
        # CAMERA -> SCREEN MAPPING
        # --------------------------

        screen_x = (
            (x1 - frameR)
            * screen_w
            / (w - (2 * frameR))
        )

        screen_y = (
            (y1 - frameR)
            * screen_h
            / (h - (2 * frameR))
        )

        screen_x = max(
            0,
            min(screen_w, screen_x)
        )

        screen_y = max(
            0,
            min(screen_h, screen_y)
        )

        # --------------------------
        # DEAD ZONE
        # --------------------------

        if abs(screen_x - current_x) < dead_zone:
            screen_x = current_x

        if abs(screen_y - current_y) < dead_zone:
            screen_y = current_y

        # --------------------------
        # EXPONENTIAL SMOOTHING
        # --------------------------

        current_x = (
            alpha * screen_x
            + (1 - alpha) * current_x
        )

        current_y = (
            alpha * screen_y
            + (1 - alpha) * current_y
        )

        # --------------------------
        # MOVE CURSOR
        # --------------------------

        if finger_count == 1:

            pyautogui.moveTo(
                current_x,
                current_y,
                duration=0
            )

        # --------------------------
        # PINCH DISTANCE
        # --------------------------

        distance = math.hypot(
            x2 - x1,
            y2 - y1
        )

        line_color = (
            (0, 255, 0)
            if distance < 60
            else (255, 0, 255)
        )

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            line_color,
            4
        )

        # --------------------------
        # SCROLLING
        # --------------------------

        if current_time - last_scroll_time > 0.3:

            if finger_count == 3:

                pyautogui.scroll(120)

                last_scroll_time = current_time

            elif finger_count == 4:

                pyautogui.scroll(-120)

                last_scroll_time = current_time

        # --------------------------
        # CLICK / DRAG
        # --------------------------

        if distance < 60 and finger_count <= 2:

            if pinch_start_time == 0:

                pinch_start_time = current_time

            hold_time = (
                current_time
                - pinch_start_time
            )

            if hold_time > 0.15 and not dragging:

                pyautogui.mouseDown()

                dragging = True

        else:

            if pinch_start_time != 0:

                hold_time = (
                    current_time
                    - pinch_start_time
                )

                if hold_time < 0.15 and not dragging:

                    pyautogui.click()

            pinch_start_time = 0

            if dragging:

                pyautogui.mouseUp()

                dragging = False

        # --------------------------
        # UI TEXT
        # --------------------------

        cv2.putText(
            img,
            f"Fingers: {finger_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Distance: {int(distance)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if dragging:

            cv2.putText(
                img,
                "DRAGGING",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        if finger_count == 0:

            cv2.putText(
                img,
                "PAUSED",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        elif finger_count == 3:

            cv2.putText(
                img,
                "SCROLL UP",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

        elif finger_count == 4:

            cv2.putText(
                img,
                "SCROLL DOWN",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

    # --------------------------
    # FPS DISPLAY
    # --------------------------

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Virtual Mouse",
        img
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------------
# CLEANUP
# ---------------------------------

if dragging:
    pyautogui.mouseUp()

cap.release()

cv2.destroyAllWindows()