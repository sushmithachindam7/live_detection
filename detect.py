import streamlit as st
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Live Webcam Detection")

# Start button
if st.button("Start Webcam Detection"):
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    stframe = st.empty()  # placeholder for video frames

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Run YOLOv8 detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Show frame in Streamlit
        stframe.image(annotated_frame, channels="BGR")

        # Stop if user presses "Stop"
        if st.button("Stop"):
            break

    cap.release()
