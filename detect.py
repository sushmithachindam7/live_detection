import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.title("YOLOv8 Live Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run YOLOv8 detection
    results = model(img)

    # Annotate results
    annotated_img = results[0].plot()

    # Display in Streamlit
    st.image(annotated_img, channels="BGR")
