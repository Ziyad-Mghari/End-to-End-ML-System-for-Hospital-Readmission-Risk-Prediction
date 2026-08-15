# This file defines the FastAPI application.
# The goal is to expose the trained machine learning model through an API.

import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# We add the src/ folder to Python's path.
# This allows us to import config.py and data_preprocessing.py from the src folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from config import MODEL_PATH  # noqa: E402
from data_preprocessing import preprocess_data  # noqa: E402


# Create the FastAPI app.
app = FastAPI(
    title="Hospital Readmission Risk Prediction API",
    description="API for predicting hospital readmission risk within 30 days.",
    version="0.1.0"
)


# Global variables used by the API.
# They will be loaded when the app starts.
model = None
expected_columns = None
sample_patient = None


class PatientRequest(BaseModel):
    """
    Request format for the prediction endpoint.

    data contains the patient features.
    threshold is the classification threshold used to convert the risk score into a class.
    """

    data: Dict[str, Any]
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """
    Response format returned by the prediction endpoint.
    """

    risk_score: float
    prediction: int
    risk_label: str
    threshold: float


@app.on_event("startup")
def load_resources():
    """
    Load the trained model and expected input columns when the API starts.

    This avoids loading the model again at every request.
    """

    global model, expected_columns, sample_patient

    # Check that the trained model exists.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run `python src/train.py` before starting the API."
        )

    # Load the trained scikit-learn pipeline.
    model = joblib.load(MODEL_PATH)

    # Reuse the preprocessing script to recover the expected input columns.
    # This ensures that the API expects the same features as the training pipeline.
    X, y, numeric_features, categorical_features = preprocess_data()

    expected_columns = X.columns.tolist()

    # Store one example patient for testing the API.
    sample_patient = X.iloc[0].to_dict()


@app.get("/")
def root():
    """
    Root endpoint.

    It simply confirms that the API is running.
    """

    return {
        "message": "Hospital Readmission Risk Prediction API is running."
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.

    It confirms whether the model has been loaded correctly.
    """

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "number_of_expected_features": len(expected_columns) if expected_columns else None
    }


@app.get("/sample-patient")
def get_sample_patient():
    """
    Return one sample patient.

    This endpoint is useful for testing the /predict endpoint.
    """

    if sample_patient is None:
        raise HTTPException(status_code=500, detail="Sample patient not loaded.")

    # Convert the sample patient into a clean JSON-compatible dictionary.
    clean_sample = {}

    for key, value in sample_patient.items():
        # Convert NumPy values into standard Python values.
        if hasattr(value, "item"):
            value = value.item()

        # Convert missing values into None so that JSON can handle them.
        if pd.isna(value):
            value = None

        clean_sample[key] = value

    return {
        "sample_patient": clean_sample
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_readmission(request: PatientRequest):
    """
    Predict hospital readmission risk for one patient.

    The input must contain the same features used during model training.
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if expected_columns is None:
        raise HTTPException(status_code=500, detail="Expected columns are not loaded.")

    # Convert the input dictionary into a one-row DataFrame.
    patient_df = pd.DataFrame([request.data])

    # Check if some required columns are missing.
    missing_columns = [
        col for col in expected_columns
        if col not in patient_df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required columns.",
                "missing_columns": missing_columns
            }
        )

    # Keep only the expected columns and put them in the correct order.
    patient_df = patient_df[expected_columns]

    # Predict the probability of class 1 = high readmission risk.
    risk_score = float(model.predict_proba(patient_df)[:, 1][0])

    # Convert the probability into a binary prediction using the chosen threshold.
    prediction = int(risk_score >= request.threshold)

    # Convert the numerical prediction into a readable label.
    risk_label = "high_risk" if prediction == 1 else "low_risk"

    return PredictionResponse(
        risk_score=risk_score,
        prediction=prediction,
        risk_label=risk_label,
        threshold=request.threshold
    )