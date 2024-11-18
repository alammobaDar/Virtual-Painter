import cv2
import mediapipe as mp
import numpy as np
import time



def fingers_is_close(index_x, index_y, middle_x, middle_y):
      if (index_x + index_y) - (middle_x + middle_y) <=0.06:
        print(f"index({index_x + index_y})")
        print(f"middle({middle_x + middle_y})")
        return True
      

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


img_canvas = np.zeros((480, 640, 3), np.uint8)

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
              
          index_tip_landmark = hand_landmarks.landmark[8]
          middle_tip_landmark = hand_landmarks.landmark[12]

          index_num_x = float(index_tip_landmark.x)
          index_num_y = float(index_tip_landmark.y)

          middle_num_x = float(middle_tip_landmark.x)
          middle_num_y = float(middle_tip_landmark.y)
        # for the tip of the pen
          
          cartesian_x = int((index_num_x)* cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          cartesian_y = int((index_num_y)* cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

          if fingers_is_close(index_x=index_num_x, index_y=index_num_y, middle_x=middle_num_x, middle_y=middle_num_y):
            cv2.circle(image, (cartesian_x, cartesian_y), 10, (0,255,0), -1)
            if True:
              xp, yp = 0, 0

              if xp == 0 and yp == 0:
                xp, yp = cartesian_x, cartesian_y
              cv2.line(image, (xp,yp),(cartesian_x, cartesian_y), (0,0,0), 10)
              cv2.line(img_canvas, (xp,yp),(cartesian_x, cartesian_y), (0, 255,0), 20)
              xp, yp = cartesian_x, cartesian_y
          
    cv2.imshow("Canvas", cv2.flip(img_canvas, 1))
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()