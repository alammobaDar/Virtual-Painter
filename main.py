import cv2
import mediapipe as mp
import numpy as np
import time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
      # Flip the image horizontally for a selfie-view display.
      for hand_landmark in results.multi_hand_landmarks:
        index_tip_landmark = hand_landmark.landmark[8]
        middle_tip_landmark = hand_landmark.landmark[12]

        index_num_x = float(index_tip_landmark.x)
        index_num_y = float(index_tip_landmark.y)

        middle_num_x = float(middle_tip_landmark.x)
        middle_num_y = float(middle_tip_landmark.y)

        cartesian_x = int((index_num_x)* cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cartesian_y = int((index_num_y)* cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (index_num_x - middle_num_x) >= -0.05 and (index_num_x - middle_num_x) <= 0.05:
          pen = cv2.circle(image, (cartesian_x, cartesian_y), 10, (0,255,0), -1)




    
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()