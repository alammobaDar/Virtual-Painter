import sys
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap

class VideoHolder(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Virtual Painter")
        self.setGeometry(100, 100, 1300, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black")

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.video_label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoHolder()
    main_window.show()
    sys.exit(app.exec())