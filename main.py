import cv2
import mediapipe as mp
import numpy as np
import time


HEIGHT = 720
WIDTH = 1280
DISTANCE = 0.065
PEN_COLOR = (0, 255, 0)
PEN_SIZE = 20


def fingers_is_close(index_x, index_y, middle_x, middle_y, index_z):
    
      # if (index_x + index_y) - (middle_x + middle_y) / index_z >= 2:
      #   # print(f"depth({(index_x + index_y) - (middle_x + middle_y) / index_z})")
        if (index_x + index_y) - (middle_x + middle_y) <= DISTANCE:
          # print(f"z({index_z})")
          # print(f"difference({(index_x + index_y) - (middle_x + middle_y)})")
          return True


    

def circle_between(image, x, y, index_x, index_y):
    circle=cv2.circle(image, (x, y), 25, (0,255,0), -1)

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # i[0], i[1] -> x, y center of the circle
            # i[2] -> radius
            print(f"Circle detected at: x = {i[0]}, y = {i[1]}, radius = {i[2]}")
    else:
        print("No circles detected.")
        
    
    
    

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

img_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
with mp_hands.Hands(
    max_num_hands=1,
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

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGRA2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, img_inv)
    image = cv2.bitwise_or(image, img_canvas)

    

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
          index_num_z = float(index_tip_landmark.z)
          middle_num_x = float(middle_tip_landmark.x)
          middle_num_y = float(middle_tip_landmark.y)
        #converting the coordinates from NDC to cartesian coordinates
          cartesian_index_x = int((index_num_x)* cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          cartesian_index_y = int((index_num_y)* cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          cartesian_middle_x= int((middle_num_x)* cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          cartesian_middle_y = int((middle_num_y)* cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

          
          
          circle_between(image, cartesian_middle_x+15, cartesian_middle_y+15, cartesian_index_x, cartesian_index_y)

          if fingers_is_close(index_x=index_num_x, index_y=index_num_y, middle_x=middle_num_x, middle_y=middle_num_y, index_z=index_num_z):
            # cv2.circle(image, (combination_x, combination_y), 10, (0,255,0), -1)

            
            if True:
              xp, yp = 0, 0

              if xp == 0 and yp == 0:
                xp, yp = cartesian_index_x, cartesian_index_y
              cv2.line(image, (xp,yp),(cartesian_index_x, cartesian_index_y), PEN_COLOR, PEN_SIZE)
              cv2.line(img_canvas, (xp,yp),(cartesian_index_x, cartesian_index_y), PEN_COLOR, PEN_SIZE)
              xp, yp = cartesian_index_x, cartesian_index_y


          
          
    cv2.flip(img_canvas, 1)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()