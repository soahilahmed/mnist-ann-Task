import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model("mnist.keras")


# -----------------------------
# Title
# -----------------------------

st.title("🔢 MNIST Digit Predictor")
st.write("Upload a handwritten digit image and the model will predict it.")


# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your digit image",
    type=["png", "jpg", "jpeg"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Display original image
    st.image(
        img,
        caption="Uploaded Image",
        width=200
    )

    # Convert to grayscale
    img = img.convert("L")

    # Resize to 28x28
    img = img.resize((28, 28))

    # Convert to numpy array
    img = np.array(img)

    # Normalize
    img = img.astype("float32") / 255.0

    # Reshape
    img = img.reshape(1, 784)

    # Prediction
    prediction = model.predict(img, verbose=0)

    # Predicted digit
    digit = np.argmax(prediction)

    # Confidence
    confidence = np.max(prediction) * 100

    # -----------------------------
    # Show Result
    # -----------------------------

    st.success(f"Predicted Digit: {digit}")

    st.info(
        f"Confidence: {confidence:.2f}%"
    )