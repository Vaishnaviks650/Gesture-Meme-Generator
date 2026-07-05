import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh()

draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

rock = cv2.imread("memes/rock.jpg")
paper = cv2.imread("memes/paper.jpg")
scissors = cv2.imread("memes/scissors.jpg")
thumb = cv2.imread("memes/thumbsup.jpg")


def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in tips[1:]:
        if hand.landmark[tip].y < hand.landmark[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:

    ret, frame = cap.read()

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)

    gesture = "None"
    meme = None

    if face_result.multi_face_landmarks:
        for face in face_result.multi_face_landmarks:
            draw.draw_landmarks(
                frame,
                face,
                mp_face.FACEMESH_TESSELATION
            )

    if hand_result.multi_hand_landmarks:

        for hand in hand_result.multi_hand_landmarks:

            draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = fingers_up(hand)

            total = fingers.count(1)

            if total == 0:
                gesture = "ROCK"
                meme = rock

            elif total == 2:
                gesture = "SCISSORS"
                meme = scissors

            elif total == 5:
                gesture = "PAPER"
                meme = paper

            elif fingers == [1,0,0,0,0]:
                gesture = "THUMBS UP"
                meme = thumb

    cv2.putText(
        frame,
        gesture,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    if meme is not None:

        meme = cv2.resize(meme,(220,180))

        frame[20:200,400:620] = meme

    cv2.imshow("Gesture Meme Generator",frame)

    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
