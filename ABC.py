import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import time
import io
from datetime import datetime
from tflite_runtime.interpreter import Interpreter  # Changed import
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
        # Use tflite_runtime instead of full tensorflow
        interpreter = Interpreter(model_path="model.tflite")  # Changed line
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
            # Get model input details
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            input_shape = input_details[0]['shape'][1:3]
            
            # Preprocess image
            image = image.resize(input_shape)
            input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
            input_array = input_array[:, :, :, (2, 1, 0)]  # Convert to BGR
            
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




# ... (rest of the code remains the same)
