"""
Handles loading the already-trained CNN model and running predictions.
The model is loaded once and reused for every request, instead of being
reloaded from disk on each call.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.keras"

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1


class ModelService:

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model file found at {MODEL_PATH}. "
                "Place your trained model there (e.g. models/model.keras)."
            )
        self._model = load_model(MODEL_PATH)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, pixels: List[float]) -> Tuple[int, float, List[float]]:

        if self._model is None:
            raise RuntimeError("Model has not been loaded yet.")

        image_array = np.array(pixels, dtype="float32").reshape(
            1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
        )

        probabilities = self._model.predict(image_array, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])

        return predicted_digit, confidence, probabilities.tolist()


# A single shared instance used across the whole app.
model_service = ModelService()
