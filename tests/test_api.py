# These tests check that the FastAPI application works correctly.

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project folders to Python's path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import MODEL_PATH
from api.main import app


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found. Run `python src/train.py` first."
)
def test_health_endpoint():
    """
    Check that the health endpoint returns a valid response.
    """

    # Using TestClient as a context manager ensures that FastAPI startup events run.
    with TestClient(app) as client:

        # Call the health endpoint.
        response = client.get("/health")

        # The endpoint should respond successfully.
        assert response.status_code == 200

        # Convert the response to a dictionary.
        data = response.json()

        # Check important response fields.
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["number_of_expected_features"] == 38


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found. Run `python src/train.py` first."
)
def test_sample_patient_endpoint():
    """
    Check that the sample-patient endpoint returns patient data.
    """

    # Using TestClient as a context manager ensures that FastAPI startup events run.
    with TestClient(app) as client:

        # Call the sample-patient endpoint.
        response = client.get("/sample-patient")

        # The endpoint should respond successfully.
        assert response.status_code == 200

        # Convert the response to a dictionary.
        data = response.json()

        # Check that the response contains a sample patient.
        assert "sample_patient" in data
        assert isinstance(data["sample_patient"], dict)

        # The sample patient should contain all expected features.
        assert len(data["sample_patient"]) == 38


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found. Run `python src/train.py` first."
)
def test_predict_endpoint():
    """
    Check that the predict endpoint returns a valid prediction.
    """

    # Using TestClient as a context manager ensures that FastAPI startup events run.
    with TestClient(app) as client:

        # First get a valid sample patient.
        sample_response = client.get("/sample-patient")

        # The sample-patient endpoint should work before testing prediction.
        assert sample_response.status_code == 200

        sample_patient = sample_response.json()["sample_patient"]

        # Create a prediction request.
        payload = {
            "data": sample_patient,
            "threshold": 0.45
        }

        # Call the prediction endpoint.
        response = client.post("/predict", json=payload)

        # The endpoint should respond successfully.
        assert response.status_code == 200

        # Convert the response to a dictionary.
        data = response.json()

        # Check that the prediction response contains the expected fields.
        assert "risk_score" in data
        assert "prediction" in data
        assert "risk_label" in data
        assert "threshold" in data

        # Check value ranges.
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["prediction"] in [0, 1]
        assert data["risk_label"] in ["low_risk", "high_risk"]
        assert data["threshold"] == 0.45