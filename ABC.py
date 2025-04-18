import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import time
import io
from datetime import datetime
from tflite_runtime.interpreter import Interpreter  # Adjusted import for TensorFlow Lite
from picamera2 import Picamera2
from libcamera import controls

# Mock food recognition model (replace with your actual model)
class FoodDetector:
    def __init__(self):
        # Load model and labels
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
    
    def load_model(self):
        """Load the TFLite model"""
        interpreter = Interpreter(model_path="model.tflite")
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
            
            # Preprocess image
            image = image.resize(input_shape)
            input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
            input_array = input_array[:, :, :, (2, 1, 0)]  # Convert to BGR
            
            # Run inference
            self.model.set_tensor(input_details[0]['index'], input_array)
            self.model.invoke()
            
            # Get results
            outputs = self.model.get_tensor(output_details[0]['index'])
            max_index = np.argmax(outputs[0])
            tag = self.labels[max_index]
            probability = outputs[0][max_index]
            
            # Apply confidence threshold
            if probability < 0.5:
                return None, 0.0
                
            return tag, probability
            
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return None, 0.0

# Initialize the detector
detector = FoodDetector()

# Initialize camera (will be initialized when needed)
camera = None

def init_camera():
    """Initialize the camera"""
    global camera
    if camera is None:
        try:
            camera = Picamera2()
            config = camera.create_still_configuration(main={"size": (1920, 1080)})
            camera.configure(config)
            camera.start()
            return True
        except Exception as e:
            st.error(f"Failed to initialize camera: {str(e)}")
            return False
    return True

def capture_image():
    """Capture an image from the camera"""
    try:
        if not init_camera():
            return None
            
        # Capture image
        image_array = camera.capture_array()
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_array)
        return image
        
    except Exception as e:
        st.error(f"Failed to capture image: {str(e)}")
        return None

# Streamlit app layout
st.set_page_config(page_title="Food Calorie Calculator", layout="wide")

# Title and description
st.title("🍏 Food Calorie Calculator")
st.markdown("""
Capture an image of your food to detect what it is and calculate its nutritional information.
""")

# Create two main columns
col1, col2 = st.columns([1, 1], gap="large")

# Column 1: Image Capture and Detection
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
            st.image(image, caption="Uploaded Food Image", use_column_width=True)
            
            if st.button("Detect Food"):
                with st.spinner("Detecting food..."):
                    detected_food, confidence = detector.detect_food(image)
                    
                    if detected_food:
                        st.success(f"Detected: {detected_food} (Confidence: {confidence:.1%})")
                        st.session_state.detected_food = detected_food
                        st.session_state.food_image = image
                    else:
                        st.warning("No food item detected with sufficient confidence")
    
    else:  # Raspberry Pi Camera capture
        st.markdown("### Raspberry Pi Camera Capture")
        
        if st.button("Capture Image"):
            with st.spinner("Capturing image..."):
                captured_image = capture_image()
                
                if captured_image is not None:
                    st.image(captured_image, caption="Captured Food Image", use_column_width=True)
                    
                    with st.spinner("Detecting food..."):
                        detected_food, confidence = detector.detect_food(captured_image)
                        
                        if detected_food:
                            st.success(f"Detected: {detected_food} (Confidence: {confidence:.1%})")
                            st.session_state.detected_food = detected_food
                            st.session_state.food_image = captured_image
                        else:
                            st.warning("No food item detected with sufficient confidence")

# Column 2: Portion Input and Results
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
        
        # Portion input
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
        else:  # Large
            weight = 150
        
        # Calculate calories
        if st.button("Calculate Nutrition"):
            food_data = detector.food_db.get(st.session_state.detected_food)
            if food_data:
                calories = (food_data['calories'] * weight) / 100
                
                # Display results
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
                st.error("Nutritional data not available for this food item")
    else:
        st.header("Nutritional Information")
        st.info("Please detect a food item first to see nutritional information")

# Footer
st.markdown("---")
st.markdown("""
**How to use:**
1. Upload or capture an image of your food
2. The system will detect what food it is
3. Select the portion size
4. View the calculated nutritional information
""")

# Clean up camera when the app is closed
def cleanup():
    global camera
