import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from picamera2 import Picamera2
import time
import datetime
import os
import requests  # For uploading images

# Configuration
MODEL_PATH = "/home/pi/Downloads/model.tflite"
LABELS_PATH = "labels.txt"
UPLOAD_ENDPOINT = "https://your-upload-endpoint.com/api/upload"  # Replace with your actual endpoint

# Load food database (same as your original)
FOOD_DB = {
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

def load_model():
    """Load the TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def load_labels():
    """Load food labels from file"""
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

def initialize_camera():
    """Initialize the Raspberry Pi camera"""
    cam = Picamera2()
    cam_config = cam.create_still_configuration(main={"size": (640, 480)})
    cam.configure(cam_config)
    return cam

def capture_image(cam):
    """Capture an image from the camera and return as PIL Image"""
    rgb_array = cam.capture_array()
    return Image.fromarray(rgb_array)

def upload_image(image, filename):
    """Upload image to a server"""
    try:
        # Save image temporarily
        temp_path = f"/tmp/{filename}"
        image.save(temp_path)
        
        # Upload the file
        with open(temp_path, 'rb') as f:
            files = {'file': (filename, f)}
            response = requests.post(UPLOAD_ENDPOINT, files=files)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return response.status_code == 200
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return False

def detect_food(model, labels, image):
    """Detect food from image using the model"""
    try:
        # Get model input details
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        input_shape = input_details[0]['shape'][1:3]
        
        # Preprocess image
        image = image.resize(input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array[:, :, :, (2, 1, 0)]  # Convert to BGR
        
        # Run inference
        model.set_tensor(input_details[0]['index'], input_array)
        model.invoke()
        
        # Get results
        outputs = model.get_tensor(output_details[0]['index'])
        max_index = np.argmax(outputs[0])
        tag = labels[max_index]
        probability = outputs[0][max_index]
        
        # Apply confidence threshold
        if probability < 0.5:  # 50% confidence threshold
            return None, 0.0
            
        return tag, probability
        
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None, 0.0

def calculate_calories(food, weight):
    """Calculate calories based on detected food and portion"""
    food_data = FOOD_DB.get(food)
    if not food_data:
        return None, None
    
    calories = (food_data['calories'] * weight) / 100
    return calories, food_data['healthy']

def main():
    st.title("🍏 Food Calorie Calculator")
    st.write("Capture food images with your Raspberry Pi camera to calculate calories")
    
    # Initialize session state
    if 'detected_food' not in st.session_state:
        st.session_state.detected_food = None
        st.session_state.confidence = 0
        st.session_state.last_image = None
    
    # Load model and labels
    model = load_model()
    labels = load_labels()
    
    # Camera section
    st.header("Food Detection")
    
    # Initialize camera only when needed
    if st.button("Start Camera"):
        cam = initialize_camera()
        cam.start()
        
        # Create placeholder for camera preview
        preview_placeholder = st.empty()
        stop_button = st.button("Stop Camera")
        
        # Camera preview loop
        while not stop_button:
            # Capture image
            image = capture_image(cam)
            
            # Display preview
            preview_placeholder.image(image, caption="Camera Preview", use_column_width=True)
            
            # Check for capture button
            if st.button("Capture and Detect", key="capture_button"):
                # Detect food
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"food_{timestamp}.jpg"
                
                # Save and upload image
                if upload_image(image, filename):
                    st.success("Image uploaded successfully!")
                
                # Detect food
                detected_food, confidence = detect_food(model, labels, image)
                
                if detected_food:
                    st.session_state.detected_food = detected_food
                    st.session_state.confidence = confidence
                    st.session_state.last_image = image
                    st.success(f"Detected: {detected_food} ({confidence:.1%} confidence)")
                else:
                    st.warning("No food item detected with sufficient confidence")
                
                break
            
            time.sleep(0.1)
        
        cam.stop()
    
    # Display last captured image if available
    if st.session_state.last_image:
        st.image(st.session_state.last_image, caption="Last Captured Image", use_column_width=True)
    
    # Portion and calculation section
    if st.session_state.detected_food:
        st.header("Nutritional Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detected Food")
            st.write(f"**{st.session_state.detected_food}**")
            st.write(f"Confidence: {st.session_state.confidence:.1%}")
        
        with col2:
            weight = st.number_input("Weight (grams)", min_value=1, value=100, step=1)
            portion = st.selectbox("Standard portions", 
                                 ["Custom", "Small (50g)", "Medium (100g)", "Large (150g)"],
                                 index=1)
            
            if portion == "Small (50g)":
                weight = 50
            elif portion == "Medium (100g)":
                weight = 100
            elif portion == "Large (150g)":
                weight = 150
        
        # Calculate calories
        calories, is_healthy = calculate_calories(st.session_state.detected_food, weight)
        
        if calories is not None:
            st.subheader("Nutritional Information")
            st.write(f"**Food:** {st.session_state.detected_food}")
            st.write(f"**Weight:** {weight}g")
            st.write(f"**Calories:** {calories:.1f} kcal")
            
            if is_healthy:
                st.success("HEALTHY")
            else:
                st.warning("NOT RECOMMENDED")

if __name__ == "__main__":
    main()
