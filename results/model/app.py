import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r'E:\DUK\SEM 3\mini project\results\model\cnn_violence2.h5')


# Function to make predictions
def make_prediction(img, model):
    img = img.resize((128, 128))
    img_array = np.array(img)
    input_img = np.expand_dims(img_array, axis=0)
    res = model.predict(input_img)
    return res[0][0]  # Return the probability of violence

# Streamlit app layout
st.title("Violence Detection")
st.write("Upload an image to check if it contains violence.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    result = make_prediction(image, model)

    # Display the result
    if result > 0.5:
        st.success("Violence Detected")
    else:
        st.success("No Violence Detected")
