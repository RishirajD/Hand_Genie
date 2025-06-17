import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from tensorflow.keras.models import load_model
import pyttsx3

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
        self.setWindowTitle("HandGenie - PyQt Version")

        self.video_label = QLabel()
        self.prediction_label = QLabel("Prediction: ")
        self.current_word_label = QLabel("Current Word: ")
        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)

        self.add_letter_btn = QPushButton("Add Letter")
        self.space_btn = QPushButton("Add Space")
        self.clear_btn = QPushButton("Clear")
        self.speak_btn = QPushButton("Speak")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.current_word_label)
        layout.addWidget(self.sentence_box)
        layout.addWidget(self.add_letter_btn)
        layout.addWidget(self.space_btn)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.speak_btn)
        self.setLayout(layout)

        # Variables
        self.cap = cv2.VideoCapture(0)
        self.current_letter = ""
        self.current_word = []
        self.sentence = ""

        # Button connections
        self.add_letter_btn.clicked.connect(self.add_letter)
        self.space_btn.clicked.connect(self.add_space)
        self.clear_btn.clicked.connect(self.clear_all)
        self.speak_btn.clicked.connect(self.speak_sentence)

        # Timer for video
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
        except:
            self.current_letter = ""

        # Draw ROI and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{self.current_letter}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.prediction_label.setText(f"Prediction: {self.current_letter}")
        self.current_word_label.setText(f"Current Word: {''.join(self.current_word)}")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480)
        self.video_label.setPixmap(QPixmap.fromImage(p))

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
    win = HandGenieApp()
    win.show()
    sys.exit(app.exec_())
