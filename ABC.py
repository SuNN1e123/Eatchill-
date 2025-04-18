import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
from datetime import datetime
import tensorflow as tf
from picamera2 import Picamera2
from libcamera import controls
import os

# Mock food recognition model (replace with your actual model)
class FoodDetector:
    def __init__(self):
        # Load model and labels
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
    
    def load_model(self):
        """Load the TFLite model"""
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    
    def load_labels(self):
        """Load food labels from file"""
        return [
            "Apple", "Banana", "Burger", "Chocolate", 
            "Chocolate Donut", "French Fries", "Fruit Oatmeal",
            "Pear", "Potato Chips", "Rice"
        ]
    
    def load_food_database(self):
        """Load food calorie database"""
        return {
            "Apple": {"calories": 52, "healthy": True},
            "Banana": {"calories": 89, "healthy": True},
            "Burger": {"calories": 313, "healthy": False},
            "Chocolate": {"calories": 535, "healthy": False},
            "Chocolate Donut": {"calories": 452, "healthy": False},
            "French Fries": {"calories": 312, "healthy": False},
            "Fruit Oatmeal": {"calories": 68, "healthy": True},
            "Pear": {"calories": 57, "healthy": True},
            "Potato Chips": {"calories": 536, "healthy": False},
            "Rice": {"calories": 130, "healthy": True}
        }
    
    def detect_food(self, image):
        """Detect food from image using the model"""
        try:
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            input_shape = input_details[0]['shape'][1:3]
            
            image = image.resize(input_shape)
            input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
            input_array = input_array[:, :, :, (2, 1, 0)]  # Convert to BGR
            
            self.model.set_tensor(input_details[0]['index'], input_array)
            self.model.invoke()
            
            outputs = self.model.get_tensor(output_details[0]['index'])
            max_index = np.argmax(outputs[0])
            tag = self.labels[max_index]
            probability = outputs[0][max_index]
            
            if probability < 0.5:
                return None, 0.0
                
            return tag, probability
            
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return None, 0.0

# Camera Manager with improved error handling
class CameraManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CameraManager, cls).__new__(cls)
            cls._instance.camera = None
            cls._instance.is_initialized = False
        return cls._instance
    
    def initialize(self):
        if not self.is_initialized:
            try:
                if os.uname().machine.startswith('arm'):
                    print("Initializing Raspberry Pi Camera...")
                    self.camera = Picamera2()
                    config = self.camera.create_still_configuration(
                        main={"size": (1280, 720)},
                        lores={"size": (640, 480)},
                        display="lores"
                    )
                    self.camera.configure(config)
                    self.camera.start()
                    time.sleep(2)  # Warm-up period
                    self.is_initialized = True
                    print("Camera initialized successfully")
                    return True
                else:
                    print("Running in camera simulation mode")
                    self.is_initialized = True
                    return True
            except Exception as e:
                print(f"Camera initialization failed: {str(e)}")
                self.is_initialized = False
                return False
        return True
    
    def capture_image(self):
        if not self.initialize():
            return None
            
        try:
            if os.uname().machine.startswith('arm'):
                print("Capturing image from Raspberry Pi Camera...")
                image_array = self.camera.capture_array("main")
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                print("Generating simulated image")
                image_array = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            
            return Image.fromarray(image_array)
            
        except Exception as e:
            print(f"Image capture failed: {str(e)}")
            return None
    
    def cleanup(self):
        if self.camera is not None:
            try:
                print("Cleaning up camera resources...")
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                print(f"Error during camera cleanup: {str(e)}")
            self.camera = None
            self.is_initialized = False

# Initialize components
detector = FoodDetector()
camera_manager = CameraManager()

# Streamlit UI
st.set_page_config(page_title="Food Calorie Calculator", layout="wide")
st.title("🍏 Food Calorie Calculator")
st.markdown("Capture an image of your food to detect and calculate nutritional information.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("Food Detection")
    capture_option = st.radio(
        "How would you like to provide the food image?",
        ("Upload an image", "Capture from Raspberry Pi Camera")
    )
    
    if capture_option == "Upload an image":
        uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Image", use_container_width=True)
            
            if st.button("Detect Food"):
                with st.spinner("Detecting food..."):
                    detected_food, confidence = detector.detect_food(image)
                    if detected_food:
                        st.success(f"Detected: {detected_food} (Confidence: {confidence:.1%})")
                        st.session_state.detected_food = detected_food
                        st.session_state.food_image = image
                    else:
                        st.warning("No food item detected with sufficient confidence")
    
    else:
        st.markdown("### Raspberry Pi Camera Capture")
        if st.button("Capture Image"):
            with st.spinner("Capturing image..."):
                captured_image = camera_manager.capture_image()
                if captured_image is not None:
                    st.image(captured_image, caption="Captured Food Image", use_container_width=True)
                    with st.spinner("Detecting food..."):
                        detected_food, confidence = detector.detect_food(captured_image)
                        if detected_food:
                            st.success(f"Detected: {detected_food} (Confidence: {confidence:.1%})")
                            st.session_state.detected_food = detected_food
                            st.session_state.food_image = captured_image
                        else:
                            st.warning("No food item detected with sufficient confidence")

with col2:
    if 'detected_food' in st.session_state:
        st.header("Nutritional Information")
        st.subheader("Detected Food")
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(st.session_state.food_image, width=150)
        with col_info:
            st.markdown(f"**{st.session_state.detected_food}**")
            food_data = detector.food_db.get(st.session_state.detected_food)
            if food_data['healthy']:
                st.success("✅ Healthy food")
            else:
                st.warning("⚠️ Not recommended for frequent consumption")
        
        st.subheader("Portion Details")
        portion_option = st.selectbox(
            "Select portion size:",
            ["Custom", "Small (50g)", "Medium (100g)", "Large (150g)"],
            index=1
        )
        
        if portion_option == "Custom":
            weight = st.number_input("Enter weight (grams):", min_value=1, value=100)
        elif portion_option == "Small (50g)":
            weight = 50
        elif portion_option == "Medium (100g)":
            weight = 100
        else:
            weight = 150
        
        if st.button("Calculate Nutrition"):
            food_data = detector.food_db.get(st.session_state.detected_food)
            if food_data:
                calories = (food_data['calories'] * weight) / 100
                st.subheader("Nutritional Information")
                cols = st.columns(2)
                cols[0].metric("Food", st.session_state.detected_food)
                cols[1].metric("Weight", f"{weight}g")
                cols = st.columns(2)
                cols[0].metric("Calories", f"{calories:.1f} kcal")
                cols[1].metric("Calories per 100g", f"{food_data['calories']} kcal")
                if food_data['healthy']:
                    st.success("This is a healthy food choice!")
                else:
                    st.warning("Consider healthier alternatives for frequent consumption")
    else:
        st.header("Nutritional Information")
        st.info("Please detect a food item first to see nutritional information")

# Cleanup
def cleanup():
    camera_manager.cleanup()

import atexit
atexit.register(cleanup)
