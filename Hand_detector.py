import numpy as np
import mediapipe as mp
import cv2

class Hand_detection:
    def __init__(self, mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mode = mode
        self.max_hands = max_num_hands
        self.complexity = model_complexity
        self.detect = min_detection_confidence
        self.tracking = min_tracking_confidence

        self.tip_ids = [4, 8, 12, 16, 20]
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detect, self.tracking)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks,self.mp_hands.HAND_CONNECTIONS)

        return img_rgb
        

    def detect_finger_tips(self, img, hands_no = 0):
        landmark_list = []

        if self.results.multi_hand_landmarks:
            selected_hands = self.results.multi_hand_landmarks[hands_no]
            for landmark_id, landmarks in enumerate(selected_hands.landmark):
                height, width, _ = img.shape

                cartesian_x, cartesian_y = int(landmarks.x * width), int(landmarks.y * height)
                landmark_list.append([landmark_id, cartesian_x, cartesian_y])

        return landmark_list
            
                
                



def main():

    cap = cv2.VideoCapture(0)
    detector = Hand_detection()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("nigger")
            continue

        detector.find_hands(image)
        detector.detect_finger_tips(image)

        cv2.imshow('Hand Detection', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()