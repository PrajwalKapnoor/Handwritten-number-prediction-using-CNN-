"""
Streamlit frontend for handwritten digit recognition.

Flow:
    1. User uploads an image.
    2. Image is preprocessed to match what the CNN was trained on.
    3. Preprocessed pixels are sent to the FastAPI /predict endpoint.
    4. Prediction and confidence are displayed.
"""

from typing import List

import numpy as np
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000/predict"

IMAGE_SIZE = (28, 28)
INVERSION_THRESHOLD = 127


def preprocess_image(image: Image.Image) -> List[float]:
    """Convert an uploaded image into the flattened, normalized format the model expects."""
    # 1. Grayscale
    image = image.convert("L")

    # 2. Resize to 28x28
    image = image.resize(IMAGE_SIZE)

    # 3. To a numpy array of pixel values (0-255)
    pixels = np.array(image, dtype="float32")

    # 4. Handle black-on-white images: invert so digit strokes are bright,
    #    background is dark, matching MNIST's convention.
    if pixels.mean() > INVERSION_THRESHOLD:
        pixels = 255.0 - pixels

    # 5. Normalize to [0, 1]
    pixels = pixels / 255.0

    # 6. Flatten to a plain list of 784 values for the API
    return pixels.flatten().tolist()


st.set_page_config(page_title="Digit Recognition", page_icon="🔢")
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a single handwritten digit (0-9) to get a prediction.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=200)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            pixels = preprocess_image(image)

            try:
                response = requests.post(API_URL, json={"pixels": pixels}, timeout=10)
                response.raise_for_status()
                result = response.json()

                st.success(f"Predicted digit: {result['predicted_digit']}")
                st.write(f"Confidence: {result['confidence'] * 100:.2f}%")

                with st.expander("Show probabilities for every digit"):
                    st.bar_chart(result["probabilities"])

            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the backend API: {exc}")
