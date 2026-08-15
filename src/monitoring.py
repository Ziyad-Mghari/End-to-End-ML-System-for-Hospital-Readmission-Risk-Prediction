# This file contains basic monitoring utilities.
# The goal is to log API predictions and summarize prediction behavior.

from datetime import datetime, timezone

import pandas as pd

from config import PREDICTION_LOG_PATH, MONITORING_DIR


def log_prediction(input_data, risk_score, prediction, risk_label, threshold):
    """
    Log one prediction made by the API.

    Parameters
    ----------
    input_data : dict
        Patient input features sent to the API.

    risk_score : float
        Predicted probability of readmission within 30 days.

    prediction : int
        Binary prediction: 0 = low risk, 1 = high risk.

    risk_label : str
        Readable prediction label.

    threshold : float
        Classification threshold used for the prediction.

    Returns
    -------
    None
        The prediction is appended to a CSV log file.
    """

    # Make sure the monitoring folder exists.
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    # Create a compact log row.
    # We do not store all patient features, only useful monitoring information.
    log_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk_score": risk_score,
        "prediction": prediction,
        "risk_label": risk_label,
        "threshold": threshold,
        "age": input_data.get("age"),
        "gender": input_data.get("gender"),
        "time_in_hospital": input_data.get("time_in_hospital"),
        "num_medications": input_data.get("num_medications"),
        "number_inpatient": input_data.get("number_inpatient"),
        "number_emergency": input_data.get("number_emergency"),
        "diabetesMed": input_data.get("diabetesMed")
    }

    # Convert the row into a one-line DataFrame.
    log_df = pd.DataFrame([log_row])

    # If the log file already exists, append without writing the header again.
    # If it does not exist, create it with a header.
    if PREDICTION_LOG_PATH.exists():
        log_df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(PREDICTION_LOG_PATH, mode="w", header=True, index=False)


def load_prediction_logs():
    """
    Load the prediction log file.

    Returns
    -------
    pd.DataFrame
        Prediction logs.
    """

    # If no predictions have been logged yet, return an empty DataFrame.
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()

    # Load the CSV log file.
    logs = pd.read_csv(PREDICTION_LOG_PATH)

    return logs


def compute_monitoring_summary():
    """
    Compute a basic monitoring summary from prediction logs.

    Returns
    -------
    dict
        Summary statistics about predictions.
    """

    # Load prediction logs.
    logs = load_prediction_logs()

    # If there are no logs yet, return an empty summary.
    if logs.empty:
        return {
            "number_of_predictions": 0,
            "message": "No predictions logged yet."
        }

    # Compute basic monitoring indicators.
    summary = {
        "number_of_predictions": int(len(logs)),
        "average_risk_score": float(logs["risk_score"].mean()),
        "high_risk_rate": float((logs["prediction"] == 1).mean()),
        "low_risk_rate": float((logs["prediction"] == 0).mean()),
        "average_threshold": float(logs["threshold"].mean())
    }

    return summary