import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# Function to download model from Google Drive
def download_model_from_drive(gdrive_url, output_path):
    if not os.path.exists(output_path):
        gdown.download(gdrive_url, output_path, quiet=False)

# Function to load model
@st.cache_resource
def load_selected_model(model_name):
    model_paths = {
        'CNN': ('https://drive.google.com/uc?id=YOUR_CNN_FILE_ID', 'cnn_model.h5'),
        'ResNet50': ('https://drive.google.com/uc?id=YOUR_RESNET50_FILE_ID', 'resnet50_model.h5'),
        'VGG19': ('https://drive.google.com/uc?id=YOUR_VGG19_FILE_ID', 'vgg19_model.h5'),
    }
    
    drive_url, file_name = model_paths[model_name]
    download_model_from_drive(drive_url, file_name)
    model = load_model(file_name)
    return model

# Preprocess the uploaded image
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Handling images with alpha channel
        img_array = img_array[..., :3]
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# Streamlit UI
st.title("🛠️ Structural Defect Detection")
st.write("Upload an image and select the model to predict defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

model_choice = st.selectbox("Select Model", ["CNN", "ResNet50", "VGG19"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    model = load_selected_model(model_choice)
    processed_img = preprocess_image(image)

    if st.button('Predict'):
        prediction = model.predict(processed_img)
        
        # Assume binary classification [defect, no defect]
        if prediction.shape[-1] == 1:
            predicted_class = 'Defective' if prediction[0][0] > 0.5 else 'Non-Defective'
        else:
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_names = {0: "Non-Defective", 1: "Defective"}  # You can change this based on your training
            predicted_class = class_names.get(predicted_class, "Unknown")

        st.success(f"Prediction: {predicted_class}")
