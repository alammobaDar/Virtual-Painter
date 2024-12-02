from turtle import width
import numpy as np
import mediapipe as mp
import cv2
import math

class Hand_detection:
    def __init__(self, mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mode = mode
        self.max_hands = max_num_hands
        self.complexity = model_complexity
        self.detect = min_detection_confidence
        self.tracking = min_tracking_confidence

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
        

    def detect_finger_position(self, img, hands_no = 0):
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            selected_hands = self.results.multi_hand_landmarks[hands_no]
            for landmark_id, landmarks in enumerate(selected_hands.landmark):
                height, width, c = img.shape

                cartesian_x, cartesian_y, z = int(landmarks.x * width), int(landmarks.y * height), landmarks.z
                self.landmark_list.append([landmark_id, cartesian_x, cartesian_y, z])

        return self.landmark_list

    def distance(self, tip_x, tip_y, knuckle_x, knuckle_y):
        return math.sqrt(((tip_x - knuckle_x)**2)+((tip_y - knuckle_y)**2))
    
    def limit(self, tip_z, amount=50):
        return (amount+ abs(tip_z*1000)) - abs(tip_z*100+10)

    def selection_mode(self, lm_list):

        if not lm_list:
            return None

        elif self.distance(tip_x=lm_list[12][1], tip_y=lm_list[12][2], knuckle_x=lm_list[9][1], knuckle_y=lm_list[9][2]) >  self.limit(tip_z=lm_list[12][3]):
            return True

    def drawing_mode(self, lm_list):

        if not lm_list:
            return None
        elif self.distance(tip_x=lm_list[8][1], tip_y=lm_list[8][2], knuckle_x=lm_list[5][1], knuckle_y=lm_list[5][2]) >  self.limit(tip_z=lm_list[8][3], amount=30):
            return True


    def get_lc(self, landmark, coordinate):
        return self.landmark_list[landmark][coordinate]
#_____________________________________

HEIGHT = 720
WIDTH = 1280
PEN_SIZE = 25
PEN_COLOR = (0, 255, 0)
ERASER_COLOR = (0,0,0)
ERASER_SIZE = 40

#_____________________________________



def main():

    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    detector = Hand_detection()

    image_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("di ka nag success eh")
            continue

        detector.find_hands(image)
        lm_list = detector.detect_finger_position(image)
        
        img_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGRA2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, img_inv)
        image = cv2.bitwise_or(image, image_canvas)


        xp, yp = 0, 0


        if detector.selection_mode(lm_list):
            print("Selection Mode")

        elif detector.drawing_mode(lm_list):
            if xp ==0 and yp == 0:
                xp, yp = lm_list[8][1], lm_list[8][2]

            cv2.line(image_canvas, (xp, yp), (lm_list[8][1], lm_list[8][2]), PEN_COLOR, PEN_SIZE)

            xp, yp = lm_list[8][1], lm_list[8][2]

        cv2.flip(image_canvas, 1)
        cv2.imshow('Hand Detection', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()