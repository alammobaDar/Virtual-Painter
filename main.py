import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Pen color range (assuming a distinct pen tip color, like red)
lower_color = np.array([15, 100, 100])
upper_color = np.array([35, 255, 255])

# for white

lower_white = np.array([0, 0, 0])
upper_white = np.array([180, 50, 255])

# Open the video feed (or use an image set if preferred)
cap = cv.VideoCapture(0)  # Using webcam for real-time hand detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB for MediaPipe
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Prepare to store 2D points (image coordinates of hand landmarks)
            hand_points = []

            for landmark in hand_landmarks.landmark:
                # Get the image coordinates of each landmark
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                # Add the points to the imgpoints array (2D)
                hand_points.append([x, y])

            # Optionally: Visualize the landmarks on the image
            for point in hand_points:
                cv.circle(frame, tuple(point), 5, (255, 0, 0), -1)

    # Detect the pen tip using color segmentation
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask_yellow_orange = cv.inRange(hsv, lower_color, upper_color)
    mask_white = cv.inRange(hsv, lower_white, upper_white)

    mask = cv.bitwise_or(mask_yellow_orange, mask_white)

    # Find contours of the pen
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(largest_contour)

        if radius > 5:  # Only consider large enough pen tips
            # Draw the detected pen tip
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # Display the processed frame with hand landmarks and pen detection
    cv.imshow('Hand and Pen Detection', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
