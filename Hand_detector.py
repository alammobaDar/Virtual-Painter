import numpy as np
import mediapipe as mp
import cv2

class Hand_detection:
    def __init__(self, mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_num_hands
        self.complexity = model_complexity
        self.detect = min_detection_confidence
        self.tracking = min_tracking_confidence

        self.tip_ids = [4, 8, 12, 16, 20]
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detect, self.tracking)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks,self.mp_hands.HAND_CONNECTIONS)

def main():

    cap = cv2.VideoCapture(0)
    detector = Hand_detection()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("nigger")
            continue

        detector.find_hands(image)

        cv2.imshow('Hand Detection', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()