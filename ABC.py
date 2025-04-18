import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
from datetime import datetime
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import atexit

class FoodDetector:
    def __init__(self):
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
    
    def load_model(self):
        """Load the TFLite model"""
        try:
            interpreter = Interpreter(model_path="model.tflite")
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            raise

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
            
            # Convert and resize image
            image = image.convert('RGB').resize(input_shape)
            input_array = np.array(image, dtype=np.float32) / 255.0
            input_array = np.expand_dims(input_array, axis=0)

            # Check if model expects BGR input
            if input_details[0]['dtype'] == np.uint8:
                input_array = (input_array * 255).astype(np.uint8)
            
            # Run inference
            self.model.set_tensor(input_details[0]['index'], input_array)
            self.model.invoke()
            
            # Process outputs
            outputs = self.model.get_tensor(output_details[0]['index'])
            max_index = np.argmax(outputs[0])
            tag = self.labels[max_index]
            probability = outputs[0][max_index]
            
            return (tag, probability) if probability >= 0.5 else (None, 0.0)
        
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            return None, 0.0

# Camera handling functions
camera = None

def init_camera():
    global camera
    if camera is None:
        try:
            camera = Picamera2()
            config = camera.create_still_configuration(main={"size": (1280, 720)})
            camera.configure(config)
            camera.start()
            time.sleep(2)  # Camera warm-up
            return True
        except Exception as e:
            st.error(f"Camera init failed: {str(e)}")
            return False
    return True

def capture_image():
    try:
        if not init_camera():
            return None
        
        array = camera.capture_array()
        image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
        
    except Exception as e:
        st.error(f"Capture failed: {str(e)}")
        return None

# Streamlit UI Configuration
st.set_page_config(page_title="RPi Food Analyzer", layout="wide")

# Initialize detector
try:
    detector = FoodDetector()
except Exception as e:
    st.error(f"""
    Critical error: {str(e)}
    Please ensure:
    1. model.tflite exists in current directory
    2. Required packages are installed
    3. Camera is properly connected
    """)
    st.stop()

# Main UI
st.title("🍓 Raspberry Pi Food Analyzer")
st.markdown("### AI-Powered Nutrition Analysis")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("Image Input")
    input_method = st.radio("Select input method:", ["Camera Capture", "Image Upload"])
    
    img_data = None
    
    if input_method == "Camera Capture":
        if st.button("Capture Food Image"):
            with st.spinner("Capturing..."):
                img_data = capture_image()
                if img_data:
                    st.session_state.food_image = img_data
    
    else:
        uploaded_file = st.file_uploader("Upload food image:", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_data = Image.open(uploaded_file)
            st.session_state.food_image = img_data
    
    if 'food_image' in st.session_state:
        st.image(st.session_state.food_image, caption="Food Image", use_column_width=True)
        
        if st.button("Analyze Food"):
            with st.spinner("Analyzing..."):
                food, confidence = detector.detect_food(st.session_state.food_image)
                if food:
                    st.session_state.detected_food = food
                    st.success(f"Detected: {food} ({confidence:.0%} confidence)")
                else:
                    st.warning("No food detected with sufficient confidence")
                    st.session_state.pop('detected_food', None)

with col2:
    st.header("Nutrition Analysis")
    
    if 'detected_food' in st.session_state:
        food_data = detector.food_db.get(st.session_state.detected_food, {})
        
        if not food_data:
            st.error("Nutrition data unavailable")
            st.stop()
        
        st.subheader(f"Analysis for {st.session_state.detected_food}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Base Calories (100g)", f"{food_data['calories']} kcal")
            st.image(st.session_state.food_image, width=200)
        
        with col_b:
            health_status = "✅ Healthy" if food_data['healthy'] else "⚠️ Limit Consumption"
            st.metric("Health Status", health_status)
        
        st.subheader("Portion Adjustment")
        portion = st.radio("Select portion size:", ["Small (50g)", "Medium (100g)", "Large (150g)", "Custom"], index=1)
        
        if portion == "Custom":
            weight = st.number_input("Enter weight (grams):", min_value=1, max_value=1000, value=100)
        else:
            weight = {
                "Small (50g)": 50,
                "Medium (100g)": 100,
                "Large (150g)": 150
            }[portion]
        
        calories = (food_data['calories'] * weight) / 100
        
        st.subheader("Results")
        st.markdown(f"""
        - **Total Weight:** {weight}g
        - **Estimated Calories:** {calories:.0f} kcal
        - **Calories per 100g:** {food_data['calories']} kcal
        """)
        
        if food_data['healthy']:
            st.success("This is a healthy choice! Great for balanced diets.")
        else:
            st.warning("Consider healthier alternatives for regular consumption")
    else:
        st.info("Capture or upload a food image to begin analysis")

# Footer and cleanup
def shutdown():
    global camera
    if camera is not None:
        camera.stop()
        camera.close()

atexit.register(shutdown)
