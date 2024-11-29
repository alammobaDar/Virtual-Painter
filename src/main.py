import numpy as np
import mediapipe as mp
import cv2
import math

from Hand_detector import Hand_detection

HEIGHT = 720
WIDTH = 1280
PEN_SIZE = 25
PEN_COLOR = (0, 255, 0)
ERASER_COLOR = (0,0,0)
ERASER_SIZE = 40


def main():

    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    detector = Hand_detection()

    image_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    setter_image = cv2.imread("src\images\SETTER.png")

    width, height = 300, 720
    setter_image = cv2.resize(setter_image, (width, height))

    img2gray = cv2.cvtColor(setter_image, cv2.COLOR_BGR2GRAY) 
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY) 


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

        # Region of Image (ROI), where we want to insert logo 
        roi = image[-height-10:10, -width-10:10] 
    
        # Set an index of where the mask is 
        roi[np.where(mask)] = 0
        roi += setter_image

        print(f"roi {roi.shape}")
        print(f"mask {mask.shape}")

        if detector.erase_mode(lm_list):
            cv2.circle(image, (lm_list[12][1], lm_list[12][2]), ERASER_SIZE, (0, 0, 0), -1)

            if xp ==0 and yp == 0:
                xp, yp = lm_list[12][1], lm_list[12][2]

            cv2.line(image_canvas, (xp, yp), (lm_list[12][1], lm_list[12][2]), ERASER_COLOR, ERASER_SIZE)

            xp, yp = lm_list[8][1], lm_list[8][2]

        elif detector.selection_mode(lm_list):
            # print("Selection Mode")
            pass

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