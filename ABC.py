import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import time
import io
from datetime import datetime
import tensorflow as tf

# Initialize Session State variables
if "food_image" not in st.session_state:
    st.session_state.food_image = None

if "detected_food" not in st.session_state:
    st.session_state.detected_food = None

# Mock food recognition model
class FoodDetector:
    def __init__(self):
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
    
    def load_model(self):
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    
    def load_labels(self):
        return [
            "Apple", "Banana", "Burger", "Chocolate", 
            "Chocolate Donut", "French Fries", "Fruit Oatmeal",
            "Pear", "Potato Chips", "Rice"
        ]
    
    def load_food_database(self):
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

detector = FoodDetector()

# App configuration
st.set_page_config(page_title="Food Calorie Calculator", layout="wide")

# Main UI
st.title("🍏 Food Calorie Calculator")
st.markdown("Capture an image of your food to detect and calculate nutritional information.")

col1, col2 = st.columns([1, 1], gap="large")

# Image Input Column
with col1:
    st.header("Food Detection")
    capture_option = st.radio(
        "Input method:",
        ("Upload an image", "Capture from Raspberry Pi Pico")
    )
    
    if capture_option == "Upload an image":
        uploaded_file = st.file_uploader("Choose food image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.session_state.food_image = Image.open(uploaded_file)
            st.image(st.session_state.food_image, caption="Uploaded Food Image", use_column_width=True)
            
            if st.button("Detect Food"):
                with st.spinner("Analyzing..."):
                    detected_food, confidence = detector.detect_food(st.session_state.food_image)
                    if detected_food:
                        st.session_state.detected_food = detected_food
                        st.success(f"Detected: {detected_food} ({confidence:.1%} confidence)")
                    else:
                        st.session_state.detected_food = None
                        st.warning("No confident detection")
    else:
        st.markdown("### Raspberry Pi Pico Integration")
        st.info("Demo placeholder - actual implementation would connect to Pico hardware")
        
        if st.button("Simulate Pico Capture"):
            try:
                st.session_state.food_image = Image.open("demo_food.jpg")
                st.image(st.session_state.food_image, caption="Simulated Capture", use_column_width=True)
                st.session_state.detected_food = "Apple"  # Demo value
                st.success("Demo detection: Apple (95% confidence)")
            except FileNotFoundError:
                st.error("Demo image not found")

# Results Column
with col2:
    if st.session_state.food_image is not None:
        st.header("Nutritional Analysis")
        
        st.subheader("Detection Results")
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(st.session_state.food_image, width=150)
        with col_info:
            if st.session_state.detected_food:
                food_data = detector.food_db.get(st.session_state.detected_food)
                status = "✅ Healthy" if food_data['healthy'] else "⚠️ Unhealthy"
                st.markdown(f"""
                **{st.session_state.detected_food}**  
                {status}
                """)
        
        st.subheader("Portion Analysis")
        portion = st.selectbox(
            "Portion size:",
            ["Small (50g)", "Medium (100g)", "Custom"],
            index=1
        )
        
        weight = 100  # Default
        if portion == "Small (50g)":
            weight = 50
        elif portion == "Custom":
            weight = st.number_input("Enter grams:", min_value=1, value=100)
        
        if st.button("Calculate"):
            if st.session_state.detected_food:
                food_data = detector.food_db.get(st.session_state.detected_food)
                if food_data:
                    calories = (food_data['calories'] * weight) / 100
                    cols = st.columns(2)
                    cols[0].metric("Total Calories", f"{calories:.1f} kcal")
                    cols[1].metric("Per 100g", f"{food_data['calories']} kcal")
                else:
                    st.error("Nutrition data unavailable")
            else:
                st.warning("No food detected")
    else:
        st.info("👈 Capture or upload a food image to begin analysis")

# Footer
st.markdown("---")
st.caption("Note: This is a demo application. Consult a nutritionist for accurate dietary advice.")
