import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

prev_gesture = ""

pTime = 0

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    lmList = []

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        draw.draw_landmarks(
            img,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        h, w, c = img.shape

        for idx, lm in enumerate(hand.landmark):

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmList.append([idx, cx, cy])

        if len(lmList) == 21:

            fingers = []

            hand_label = results.multi_handedness[0].classification[0].label

            # Thumb
            if hand_label == "Right":

                if lmList[4][1] < lmList[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            else:

                if lmList[4][1] > lmList[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Index
            fingers.append(
                1 if lmList[8][2] < lmList[6][2] else 0
            )

            # Middle
            fingers.append(
                1 if lmList[12][2] < lmList[10][2] else 0
            )

            # Ring
            fingers.append(
                1 if lmList[16][2] < lmList[14][2] else 0
            )

            # Pinky
            fingers.append(
                1 if lmList[20][2] < lmList[18][2] else 0
            )

            total = sum(fingers)

            if total == 0:
                gesture = "FIST"

            elif total == 1:
                gesture = "ONE"

            elif total == 2:
                gesture = "PEACE"

            elif total == 3:
                gesture = "THREE"

            elif total == 4:
                gesture = "FOUR"

            elif total == 5:
                gesture = "OPEN"

            else:
                gesture = "UNKNOWN"

            if gesture != prev_gesture:
                print("Gesture:", gesture)
                prev_gesture = gesture

            cv2.putText(
                img,
                f"Fingers: {total}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                img,
                gesture,
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3
            )

            cv2.putText(
                img,
                f"Hand: {hand_label}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    cTime = time.time()

    fps = 1 / (cTime - pTime) if pTime != 0 else 0

    pTime = cTime

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()