# 🍏 Food Calorie Calculator - Raspberry Pi Edition

A Streamlit app that detects food items from camera/images and calculates their calories using TensorFlow Lite.

![App Screenshot](demo_screenshot.jpg)  <!-- Add a screenshot later -->

## Features
- Real-time food detection via Raspberry Pi Camera
- Calorie calculation based on portion size
- Healthiness classification (healthy/unhealthy)
- Supports both camera capture and image upload

## Hardware Requirements
- Raspberry Pi (3/4/5 recommended)
- Official Raspberry Pi Camera Module (or compatible USB webcam)
- Python 3.9+

## Setup Instructions

### 1. Install Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Install Python dependencies
pip install -r requirements.txt
