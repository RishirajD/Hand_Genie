import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Set up Streamlit page
st.set_page_config(page_title="HandGenie", layout="centered")
st.title("🤖 HandGenie - Real-time Hand Gesture Recognition")
st.markdown("Use your hand gestures to form letters, words, and sentences in real-time.")
st.markdown("Place your hand in the box below and let the AI predict your gesture.")

# Load the trained model
model = load_model("model/handgenie_model.h5")

# Load class labels
def load_labels(filepath="labels.txt"):
    labels = {}
    with open(filepath, "r") as f:
        for line in f:
            index, label = line.strip().split(":")
            labels[int(index)] = label
    return labels

labels = load_labels()

# Define predict function
def predict(model, image):
    img = cv2.resize(image, (224, 224))  # Resize to model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    preds = model.predict(img, verbose=0)[0]
    predicted_class = np.argmax(preds)
    confidence = np.max(preds)
    return labels[predicted_class], confidence

# Initialize webcam
run = st.checkbox("Start Camera")
frame_window = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    st.markdown("**Press 'Stop Camera' to quit.**")
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        # Define ROI
        x1, y1, x2, y2 = 100, 100, 400, 400
        roi = frame[y1:y2, x1:x2]

        # Predict gesture
        try:
            gesture, confidence = predict(model, roi)
            text = f"{gesture} ({confidence * 100:.2f}%)"
        except Exception as e:
            text = f"Error: {e}"

        # Draw ROI and prediction text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert BGR to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()
    cv2.destroyAllWindows()
else:
    st.info("Turn on the checkbox above to start the webcam.")