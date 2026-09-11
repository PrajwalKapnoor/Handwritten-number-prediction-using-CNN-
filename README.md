# Digit Recognition — Streamlit + FastAPI

A simple two-part app: a Streamlit frontend for uploading a digit image, and
a FastAPI backend that runs your already-trained CNN model on it.

## Project structure

```
digit-recognition/
│
├── backend/
│   ├── main.py            # FastAPI app: /health and /predict endpoints
│   ├── model_service.py   # Loads the CNN once, runs predictions
│   └── schemas.py         # Pydantic request/response models
│
├── frontend/
│   └── app.py              # Streamlit UI: upload, preprocess, call API, show result
│
├── models/
│   └── model.keras         # <-- put your already-trained model file here
│
├── requirements.txt
└── README.md
```

**What each file does**
- `backend/schemas.py` — defines and validates the JSON shapes going in/out of the API (e.g. "pixels must be a list of exactly 784 numbers").
- `backend/model_service.py` — loads `models/model.keras` once at startup and exposes a `predict()` method the API can call.
- `backend/main.py` — the actual FastAPI app; wires the endpoints to `model_service`.
- `frontend/app.py` — the Streamlit page: file upload, image preprocessing (grayscale, resize, normalize, invert if needed), calls the backend, displays results.

## Setup

```bash
cd digit-recognition
pip install -r requirements.txt
```

Put your trained model at `models/model.keras`
(if you saved it differently, e.g. `model.h5`, either rename it or update
`MODEL_PATH` in `backend/model_service.py`).

## Run

Open two terminals.

**Terminal 1 — backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Check it's alive: open http://localhost:8000/health

**Terminal 2 — frontend:**
```bash
cd frontend
streamlit run app.py
```
This opens the app in your browser (usually http://localhost:8501).

## Notes

- The frontend does all image preprocessing (grayscale → 28x28 → normalize →
  auto-invert if needed) before sending plain pixel values to the backend —
  the backend only ever deals with numbers, not image files.
- If predictions look wrong on real photos, it's almost always a
  preprocessing mismatch (inversion, centering, or scaling) rather than the
  model itself — MNIST-trained models are sensitive to digits not being
  centered/sized the way training images were.
