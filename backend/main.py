"""
FastAPI backend for handwritten digit recognition.

Endpoints:
    GET  /health   - simple check that the API and model are ready
    POST /predict  - takes a flattened 28x28 image and returns a prediction
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from model_service import model_service
from schemas import HealthResponse, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_service.load()
    yield


app = FastAPI(title="Digit Recognition API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_service.is_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict_digit(request: PredictionRequest) -> PredictionResponse:
    try:
        digit, confidence, probabilities = model_service.predict(request.pixels)
    except RuntimeError as exc:
        # Model wasn't loaded for some reason.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Anything else that goes wrong during prediction.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        predicted_digit=digit,
        confidence=confidence,
        probabilities=probabilities,
    )


# run : C:/Users/Prajwal/anaconda3/python.exe -m streamlit run app.py
# run :uvicorn main:app --reload
    
