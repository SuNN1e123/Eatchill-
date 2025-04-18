import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import platform

# Environment detection
IS_RASPBERRY_PI = (sys.version_info[0] == 3 and sys.version_info[1] == 9 and platform.system() == 'Linux')

# Conditional imports
if IS_RASPBERRY_PI:
    try:
        import tflite_runtime.interpreter as tflite
        from picamera2 import Picamera2
        from libcamera import controls
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
else:
    TFLITE_AVAILABLE = False

# Food recognition model with fallbacks
class FoodDetector:
    def __init__(self):
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
        self.model = None
        
        if TFLITE_AVAILABLE:
            try:
                self.model = tflite.Interpreter(model_path="model.tflite")
                self.model.allocate_tensors()
            except Exception as e:
                st.warning(f"TFLite failed to load: {str(e)}")
                self.model = None
    
    def load_labels(self):
        """Load food labels"""
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
        """Detect food from image with fallback to mock data"""
        if not TFLITE_AVAILABLE or self.model is None:
            # Mock detection for cloud/local testing
            return "Apple", 0.95  # (food_name, confidence)
        
        try:
            # Original TFLite detection code
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

# Camera handling with fallback
camera = None

def init_camera():
    """Initialize camera with fallback"""
    if not IS_RASPBERRY_PI:
        st.warning("Camera only available on Raspberry Pi")
        return False
        
    global camera
    try:
        camera = Picamera2()
        config = camera.create_still_configuration(main={"size": (1920, 1080)})
        camera.configure(config)
        camera.start()
        return True
    except Exception as e:
        st.warning(f"Camera unavailable: {str(e)}")
        return False

def capture_image():
    """Capture image with fallback to sample image"""
    if not IS_RASPBERRY_PI:
        # Use a sample image in cloud/local dev
        sample_image = Image.new('RGB', (800, 600), color='green')
        st.info("Using sample image (camera not available)")
        return sample_image
        
    try:
        if not init_camera():
            return None
            
        image_array = camera.capture_array()
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_array)
    except Exception as e:
        st.error(f"Capture failed: {str(e)}")
        return None

# Streamlit UI (unchanged from your original)
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
            st.image(image, caption="Uploaded Food Image", use_column_width=True)
            
            if st.button("Detect Food"):
                with st.spinner("Detecting food..."):
                    detected_food, confidence = detector.detect_food(image)
                    if detected_food:
                        st.success(f"Detected: {detected_food} (Confidence: {confidence:.1%})")
                        st.session_state.detected_food = detected_food
                        st.session_state.food_image = image
                    else:
                        st.warning("No food detected with sufficient confidence")
    else:
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
                            st.warning("No food detected with sufficient confidence")

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
            st.success("✅ Healthy food") if food_data['healthy'] else st.warning("⚠️ Not recommended")
        
        st.subheader("Portion Details")
        portion_option = st.selectbox(
            "Select portion size:",
            ["Custom", "Small (50g)", "Medium (100g)", "Large (150g)"],
            index=1
        )
        
        weight = (
            st.number_input("Enter weight (grams):", min_value=1, value=100) 
            if portion_option == "Custom" else
            50 if portion_option == "Small (50g)" else
            100 if portion_option == "Medium (100g)" else
            150
        )
        
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
                st.success("Healthy choice!") if food_data['healthy'] else st.warning("Consider healthier options")
    else:
        st.header("Nutritional Information")
        st.info("Please detect a food item first")

st.markdown("---")
st.markdown("**How to use:**\n1. Upload/capture food image\n2. System detects food\n3. Select portion size\n4. View nutrition info")

# Cleanup
def cleanup():
    global camera
    if camera is not None:
        camera.stop()
        camera.close()

import atexit
atexit.register(cleanup)

# Initialize detector
detector = FoodDetector()
