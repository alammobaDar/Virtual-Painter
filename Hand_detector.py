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