import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# Function to download models from Google Drive
def download_model(gdrive_url, filename):
    if not os.path.exists(filename):
        gdown.download(gdrive_url, filename, quiet=False)

# Cache the model loading
@st.cache_resource
def load_model_from_drive(model_name):
    model_links = {
        'Custom CNN': ('https://drive.google.com/uc?id=1dX_uFMbggm9fDkfPZORing9GDxWwY2-v', 'cnn_model.h5'),
        'ResNet50': ('https://drive.google.com/uc?id=1hBHYBBnkd0t3ZRVS7Pxqecf5eyu9HUlk', 'resnet50_model.h5'),
        'VGG19': ('https://drive.google.com/uc?id=19Jy70lJjwp0QSbzFrZv_JaHHOmhTIm_y', 'vgg19_model.h5'),

    }
    gdrive_url, filename = model_links[model_name]
    download_model(gdrive_url, filename)
    model = load_model(filename)
    return model

# Image Preprocessing
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # If PNG with alpha channel
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# Class names based on your project
structure_classes = ['Deck', 'Pavement', 'Wall']
crack_classes = ['Cracked', 'Non Cracked']

# Streamlit app
st.set_page_config(page_title="Structural Defect Detection", page_icon="🏗️")

st.title("🏗️ Structural Defect Detection App")
st.write("Upload an image of concrete structure to detect type and crack status.")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
model_selection = st.selectbox("Select the Model for Prediction", ["Custom CNN", "ResNet50", "VGG19"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    model = load_model_from_drive(model_selection)
    processed_image = preprocess_image(image)

    if st.button("Predict"):
        prediction = model.predict(processed_image)

        if prediction.shape[-1] == 6:
            # 6 outputs: Deck Cracked, Deck NonCracked, Pavement Cracked, Pavement NonCracked, Wall Cracked, Wall NonCracked
            predicted_idx = np.argmax(prediction, axis=1)[0]
            
            # Map index to structure type and crack status
            mapping = {
                0: ('Deck', 'Cracked'),
                1: ('Deck', 'Non Cracked'),
                2: ('Pavement', 'Cracked'),
                3: ('Pavement', 'Non Cracked'),
                4: ('Wall', 'Cracked'),
                5: ('Wall', 'Non Cracked')
            }

            structure_type, crack_status = mapping[predicted_idx]
            confidence = np.max(prediction)

            st.success(f"Prediction: {structure_type} - {crack_status} ({confidence*100:.2f}% confidence)")
        else:
            st.error("Model output shape not matching expected 6 classes. Please check your model.")
