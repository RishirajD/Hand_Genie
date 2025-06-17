import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit,
    QWidget, QGridLayout, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer
import qdarkstyle
import pyttsx3
from tensorflow.keras.models import load_model
import pyqtgraph as pg

# Load model and labels
model = load_model("model/handgenie_model.h5")
labels = {}
with open("labels.txt", "r") as f:
    for line in f:
        index, label = line.strip().split(":")
        labels[int(index)] = label

def predict(model, image):
    img = cv2.resize(image, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)[0]
    predicted_class = np.argmax(preds)
    confidence = np.max(preds)
    return labels[predicted_class], confidence

class HandGenieApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧙‍♂️ HandGenie - PyQt Edition")
        self.setGeometry(100, 100, 900, 700)
        self.setWindowIcon(QIcon("icon.png"))

        # Font
        self.setFont(QFont("Segoe UI", 11))

        # Widgets
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        self.prediction_label = QLabel("Prediction: ")
        self.current_word_label = QLabel("Current Word: ")
        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setFixedHeight(100)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen(color='b', width=2))
        self.predictions_history = []

        self.add_letter_btn = QPushButton("Add Letter")
        self.space_btn = QPushButton("Add Space")
        self.clear_btn = QPushButton("Clear")
        self.speak_btn = QPushButton("Speak")

        for btn in [self.add_letter_btn, self.space_btn, self.clear_btn, self.speak_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0078d7;
                    color: white;
                    border-radius: 6px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #005fa3;
                }
            """)

        # Layouts
        video_group = QVBoxLayout()
        video_group.addWidget(self.video_label)
        video_group.addWidget(self.plot_widget)

        info_group = QVBoxLayout()
        info_group.addWidget(self.prediction_label)
        info_group.addWidget(self.current_word_label)
        info_group.addWidget(self.sentence_box)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_letter_btn)
        button_layout.addWidget(self.space_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.speak_btn)

        info_group.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_group)
        main_layout.addLayout(info_group)

        self.setLayout(main_layout)

        # Variables
        self.cap = cv2.VideoCapture(0)
        self.current_letter = ""
        self.current_word = []
        self.sentence = ""

        # Connections
        self.add_letter_btn.clicked.connect(self.add_letter)
        self.space_btn.clicked.connect(self.add_space)
        self.clear_btn.clicked.connect(self.clear_all)
        self.speak_btn.clicked.connect(self.speak_sentence)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        x1, y1, x2, y2 = 100, 100, 400, 400
        roi = frame[y1:y2, x1:x2]
        try:
            gesture, conf = predict(model, roi)
            self.current_letter = gesture
            self.predictions_history.append(conf)
            if len(self.predictions_history) > 100:
                self.predictions_history.pop(0)
            self.plot_curve.setData(self.predictions_history)
        except:
            self.current_letter = ""

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{self.current_letter}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.prediction_label.setText(f"Prediction: {self.current_letter}")
        self.current_word_label.setText(f"Current Word: {''.join(self.current_word)}")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_image = qt_image.scaled(self.video_label.width(), self.video_label.height())
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def add_letter(self):
        if self.current_letter:
            self.current_word.append(self.current_letter)
            self.current_word_label.setText(f"Current Word: {''.join(self.current_word)}")

    def add_space(self):
        if self.current_word:
            self.sentence += ''.join(self.current_word) + ' '
            self.sentence_box.setPlainText(self.sentence)
            self.current_word = []
            self.current_word_label.setText("Current Word: ")

    def clear_all(self):
        self.sentence = ""
        self.current_word = []
        self.sentence_box.setPlainText("")
        self.current_word_label.setText("Current Word: ")

    def speak_sentence(self):
        engine = pyttsx3.init()
        engine.say(self.sentence)
        engine.runAndWait()

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 11))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win = HandGenieApp()
    win.show()
    sys.exit(app.exec_())
