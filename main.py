import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Arrays to store 3D object points and 2D image points
objpoints = []  # 3D points in real-world space (if applicable)
imgpoints = []  # 2D points in image plane (detected hand landmarks)

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

            # Append the detected landmarks for calibration
            imgpoints.append(np.array(hand_points, dtype=np.float32))

            # Optionally: Visualize the landmarks on the image
            for point in hand_points:
                cv.circle(frame, tuple(point), 5, (255, 0, 0), -1)

    # Display the processed frame with hand landmarks
    cv.imshow('Hand Calibration', frame)

    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
