import sys
import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton


class VideoCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OpenCV Video Capture in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # Layout and widgets
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel()  # Label to display video frames
        self.video_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start Capture")
        self.stop_button = QPushButton("Stop Capture")
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.central_widget.setLayout(layout)

        # Video capture attributes
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Button actions
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)

    def start_capture(self):
        """Start the video capture."""
        self.cap = cv2.VideoCapture(0)  # 0 is the default camera
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        self.timer.start(30)  # Update every 30ms (~30 FPS)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_capture(self):
        """Stop the video capture."""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        """Update the video frame."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to QImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                height, width, channel = frame.shape
                qimg = QImage(frame.data, width, height, channel * width, QImage.Format_RGB888)

                # Display the image on QLabel
                pixmap = QPixmap.fromImage(qimg)
                self.video_label.setPixmap(pixmap)
            else:
                print("Error: Could not read frame.")

    def closeEvent(self, event):
        """Ensure the video capture is released on close."""
        self.stop_capture()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoCaptureApp()
    main_window.show()
    sys.exit(app.exec())