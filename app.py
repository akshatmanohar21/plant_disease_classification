import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the TensorFlow model
model = tf.keras.models.load_model("1.h5")

# Define the class names
class_names = ["Early", "Healthy", "Late"]  # Update with your actual class names

# Function to preprocess the image and make predictions
def predict(image):
    # Preprocess the image
    image = np.array(image)
    image = tf.image.resize(image, (256, 256))  # Resize the image to match the model input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    
    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    
    return class_names[predicted_class]

# Streamlit app layout
st.title("Plant Disease Classification")
st.write("Upload a photo to classify the stage")

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict the class
    predicted_class = predict(image)
    
    st.write("Predicted Class:", predicted_class)
