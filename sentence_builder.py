# sentence_builder.py

import cv2
import numpy as np
import streamlit as st
import time
import pyttsx3
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("model/handgenie_model.h5")

def load_labels(filepath="labels.txt"):
    labels = {}
    with open(filepath, "r") as f:
        for line in f:
            index, label = line.strip().split(":")
            labels[int(index)] = label
    return labels

labels = load_labels()

# Predict gesture
def predict(model, image):
    img = cv2.resize(image, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)[0]
    predicted_class = np.argmax(preds)
    confidence = np.max(preds)
    return labels[predicted_class], confidence

# Text-to-speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit setup
st.set_page_config(page_title="HandGenie Sentence Builder", layout="centered")
st.title("🧠 HandGenie - Build Sentences with Hand Gestures")
st.markdown("Use your hand gestures to form letters, words, and sentences in real-time.")

# Session State Initialization
if "current_word" not in st.session_state:
    st.session_state.current_word = []
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_letter" not in st.session_state:
    st.session_state.last_letter = ""
if "prediction_text" not in st.session_state:
    st.session_state.prediction_text = ""

# Start camera
start_cam = st.checkbox("🎥 Start Camera")
frame_placeholder = st.image([])

if start_cam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
    else:
        st.success("✅ Camera started successfully.")
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            x1, y1, x2, y2 = 100, 100, 400, 400
            roi = frame[y1:y2, x1:x2]

            try:
                gesture, confidence = predict(model, roi)
                if confidence > 0.90:
                    st.session_state.last_letter = gesture
                    st.session_state.prediction_text = f"{gesture} ({confidence * 100:.2f}%)"
            except:
                st.session_state.prediction_text = "?"

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, st.session_state.prediction_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb)

            time.sleep(0.1)

        cap.release()
else:
    st.info("Click checkbox above to start the camera.")

# Show predicted letter
st.markdown("### 🔠 Predicted Letter")
st.write(f"**Letter:** `{st.session_state.last_letter}`")

# Sentence building controls
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("➕ Add Letter"):
        st.session_state.current_word.append(st.session_state.last_letter)
with col2:
    if st.button("⏎ End Word"):
        word = ''.join(st.session_state.current_word)
        st.session_state.sentence += word + " "
        st.session_state.current_word = []
with col3:
    if st.button("␣ Add Space"):
        st.session_state.sentence += " "
with col4:
    if st.button("🔄 Reset"):
        st.session_state.current_word = []
        st.session_state.sentence = ""

col5, col6, col7 = st.columns(3)
with col5:
    if st.button("🔙 Backspace"):
        if st.session_state.current_word:
            st.session_state.current_word.pop()
with col6:
    if st.button("🗑️ Delete Word"):
        st.session_state.current_word = []
with col7:
    if st.button("🔊 Speak"):
        if st.session_state.sentence.strip():
            speak_text(st.session_state.sentence.strip())

# Show status
st.markdown("### ✍️ Sentence Formation")
st.write(f"**Current Word:** `{''.join(st.session_state.current_word)}`")
st.write(f"**Sentence:** `{st.session_state.sentence.strip()}`")
