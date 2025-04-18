import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from picamera2 import Picamera2
import time
import datetime
import requests
import io
import os

# Configuration
MODEL_PATH = "model.tflite"  # Update path to your model
LABELS_PATH = "labels.txt"    # Update path to your labels
UPLOAD_ENDPOINT = "http://your-server.com/upload"  # Replace with your endpoint

# Food database (customize as needed)
FOOD_DB = {
    "Apple": {"calories": 52, "healthy": True},
    "Banana": {"calories": 89, "healthy": True},
    "Burger": {"calories": 313, "healthy": False}
}

# Initialize Raspberry Pi Camera
def init_camera():
    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (640, 480)})
    cam.configure(config)
    return cam

# Capture image and return as PIL Image
def capture_image(cam):
    array = cam.capture_array()
    return Image.fromarray(array)

# Upload image to server
def upload_image(image):
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('food.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(UPLOAD_ENDPOINT, files=files)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return False

# Streamlit App
def main():
    st.title("🍓 Raspberry Pi Food Tracker")
    st.write("Capture food with Pi Camera → Upload → Analyze")

    # Initialize camera only once
    if 'camera' not in st.session_state:
        st.session_state.camera = init_camera()
        st.session_state.camera.start()

    # Capture button
    if st.button("📸 Capture & Upload"):
        img = capture_image(st.session_state.camera)
        st.image(img, caption="Captured Food")

        if upload_image(img):
            st.success("✅ Photo uploaded successfully!")
        else:
            st.error("❌ Upload failed")

    # Cleanup on app close
    def on_close():
        if 'camera' in st.session_state:
            st.session_state.camera.stop()
    st.session_state.on_close = on_close

if __name__ == "__main__":
    main()
