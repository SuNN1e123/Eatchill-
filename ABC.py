import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tensorflow as tf
from picamera2 import Picamera2
from libcamera import controls

class FoodDetector:
    # ... (keep your existing FoodDetector class exactly as is) ...

class PiCameraHandler:
    def __init__(self):
        self.cam = None
        
    def __enter__(self):
        try:
            self.cam = Picamera2()
            config = self.cam.create_still_configuration()
            self.cam.configure(config)
            self.cam.start()
            return self
        except Exception as e:
            st.error(f"Camera initialization failed: {str(e)}")
            return None

    def capture_image(self):
        if self.cam is None:
            return None
            
        try:
            self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            time.sleep(1)  # Shorter delay for autofocus
            array = self.cam.capture_array()
            return Image.fromarray(array)
        except Exception as e:
            st.error(f"Capture failed: {str(e)}")
            return None
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cam is not None:
            self.cam.stop()

# Initialize only the detector at top level
detector = FoodDetector()

# Streamlit app
st.set_page_config(page_title="Food Calorie Calculator", layout="wide")
st.title("🍏 Food Calorie Calculator")

# Initialize session state
if 'detected_food' not in st.session_state:
    st.session_state.detected_food = None
    st.session_state.food_image = None

# Main columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("Food Detection")
    capture_option = st.radio(
        "Image source:",
        ("Upload", "Capture from Camera")
    )
    
    if capture_option == "Upload":
        uploaded_file = st.file_uploader("Choose food image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            if st.button("Detect Food"):
                detected_food, confidence = detector.detect_food(image)
                if detected_food:
                    st.session_state.detected_food = detected_food
                    st.session_state.food_image = image
                    st.success(f"Detected: {detected_food} ({confidence:.1%})")
                else:
                    st.warning("No confident detection")
    
    else:
        if st.button("Capture Image"):
            with PiCameraHandler() as camera:
                if camera:
                    image = camera.capture_image()
                    if image:
                        st.image(image, use_column_width=True)
                        detected_food, confidence = detector.detect_food(image)
                        if detected_food:
                            st.session_state.detected_food = detected_food
                            st.session_state.food_image = image
                            st.success(f"Detected: {detected_food} ({confidence:.1%})")
                        else:
                            st.warning("No confident detection")

# ... (keep your existing Column 2 nutritional information code exactly as is) ...
