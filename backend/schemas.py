"""
Pydantic models that define the shape of data going in and out of the API.
FastAPI uses these to validate requests automatically and to document the API.
"""

from typing import List

from pydantic import BaseModel, field_validator

# A 28x28 image flattened into a single list of 784 pixel values.
EXPECTED_PIXEL_COUNT = 28 * 28


class PredictionRequest(BaseModel):
    """What the frontend sends: a flattened, normalized 28x28 image."""

    pixels: List[float]

    @field_validator("pixels")
    @classmethod
    def validate_pixel_count(cls, value: List[float]) -> List[float]:
        if len(value) != EXPECTED_PIXEL_COUNT:
            raise ValueError(
                f"'pixels' must contain exactly {EXPECTED_PIXEL_COUNT} values "
                f"(a flattened 28x28 image), got {len(value)}."
            )
        return value


class PredictionResponse(BaseModel):
    """What the backend sends back after running the model."""

    predicted_digit: int
    confidence: float
    probabilities: List[float]


class HealthResponse(BaseModel):
    """Simple status payload for the /health endpoint."""

    status: str
    model_loaded: bool
