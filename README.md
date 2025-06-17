# Hand_Genie
Title: HandGenie – Real-time American Sign Language Recognition
Author: Rishiraj Das
Tech Stack: Python, TensorFlow, OpenCV, InceptionV3, Streamlit, PyQt5, QML

🔍 1. Project Overview
Objective:
To build a real-time system that recognizes American Sign Language (ASL) hand gestures using a camera and translates them into meaningful text and speech.
Goal:
Enhance communication accessibility for the hearing and speech-impaired by integrating vision and AI.

🧠 2. Machine Learning Model
➤ Model Architecture: InceptionV3
A deep convolutional neural network optimized for image classification.Pretrained on ImageNet, then fine-tuned for ASL alphabets.Final layers customized for 26 classes (A–Z) + preprocessing.
➤ Dataset
Name: Custom ASL Hand Gesture Dataset
Size: Over 10,400 images
Classes: 26 (A–Z)
Data Augmentation: Applied to improve generalization.
Input Shape: 224x224x3
➤ Training Details
Optimizer: Adam
Loss: Categorical Crossentropy
Epochs: ~70 (early stopping used)
Accuracy: > 98.86% on validation set

📹 3. Real-Time Prediction Pipeline
Camera Input: Captured via OpenCV (region of interest: hand box).
Preprocessing: Resize, normalize, expand dimensions
Model Inference: Predict letter with associated confidence
Postprocessing:
Append letter to form word
Space insertion to form sentence
Optional: Clear / Speak the sentence

🌐 4. Streamlit Web Application
✔ Features
Live webcam preview
Draws a prediction box (ROI)
Predicts gesture in real-time with confidence %
Sentence builder with buttons: Add, Space, Clear, Speak
Integrated with pyttsx3 for speech synthesis
Minimal, modern UI with emoji and Markdown styling

✔ Pros
Easy to deploy on web or localhost,No frontend coding required,Clean UI with minimal Python code,Lightweight for demo and testing

✔ Limitations
Slower frame refresh rate (due to Streamlit reloading images),No native support for live video in browser,Freezes webcam sometimes due to synchronous execution

🖥️ 5. PyQt5 Desktop Application with QML
✔ Features
Fully responsive desktop app
Live model prediction and sentence generation
Real-time updates via signals and slots
Integrated speech synthesis
QML GUI for animated transitions, smooth UI, better UX
Backend logic isolated in HandGenieBackend.py

✔ UI Components
Live prediction text
Sentence text area
Buttons for Add Letter, Space, Clear, Speak
Stylish layout with animations using QtQuick

✔ Pros
Faster and smoother than Streamlit
Real-time signal-slot architecture
Native desktop experience with animations
Future-ready for mobile/tablet (via QtQuick)

✔ Limitations
Requires PyQt and QML knowledge
Slightly more setup overhead
Distribution requires PyInstaller or similar

🧪 6. Testing and Validation
Model evaluated on unseen gestures
Real-time test cases for:
Single letter recognition
Word formation speed
Lighting and background variation
Edge cases tested:
Occluded hands
Incorrect hand shapes
Fast movement

🧰 7. Tools and Libraries Used
Component	Library Used
Model	TensorFlow, Keras
Image Processing	OpenCV
Web UI	Streamlit
Desktop UI	PyQt5, QML
Speech Engine	pyttsx3
Data Visualization (optional)	Matplotlib
