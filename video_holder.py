import sys
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
import cv2

class VideoHolder(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Virtual Painter")
        self.setGeometry(100, 100, 1300, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black")
        self.start_button = QPushButton("Start Capture")
        self.stop_button = QPushButton("Stop Capture")

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)


    def start_capture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("self.cap is not open")

        self.timer.start(30)
    
    def stop_capture(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_label.clear()

    def update_frame(self):
        if self.cap.isOpened():
            success, image = self.cap.read()

            if not success:
                print("Biglang di nagsuccess")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            qimg = QImage(image.data, width, height, channel * width, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

    def close_frame(self, event):
        self.stop_capture()
        event.accept
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoHolder()
    main_window.show()
    sys.exit(app.exec())