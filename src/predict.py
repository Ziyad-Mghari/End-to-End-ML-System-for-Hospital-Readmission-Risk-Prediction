# This script loads the trained model and makes predictions.
# It is used to check that the saved model can be reused without retraining.

import joblib
import pandas as pd

from config import MODEL_PATH
from data_preprocessing import preprocess_data


def load_model():
    """
    Load the trained model from the models folder.

    Returns
    -------
    model
        Trained scikit-learn pipeline.
    """

    # Load the saved model using joblib.
    model = joblib.load(MODEL_PATH)

    return model


def predict_sample(model, X, n_samples=5):
    """
    Make predictions on a small sample of patients.

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn model.

    X : pd.DataFrame
        Input features.

    n_samples : int
        Number of patients to predict.

    Returns
    -------
    pd.DataFrame
        Predictions with risk scores.
    """

    # Select a small sample of patients.
    X_sample = X.head(n_samples)

    # Predict class labels: 0 = low risk, 1 = high risk.
    predictions = model.predict(X_sample)

    # Predict probability of class 1 = high readmission risk.
    risk_scores = model.predict_proba(X_sample)[:, 1]

    # Create a readable results table.
    results = pd.DataFrame({
        "prediction": predictions,
        "risk_score": risk_scores
    })

    # Convert prediction into a more readable label.
    results["risk_label"] = results["prediction"].map({
        0: "low_risk",
        1: "high_risk"
    })

    return results


def main():
    """
    Main prediction function.

    This function:
    1. loads the preprocessed features
    2. loads the trained model
    3. makes predictions on a small sample
    4. prints the results
    """

    print("Loading data...")

    # We reuse the preprocessing pipeline to get the same input features as during training.
    X, y, numeric_features, categorical_features = preprocess_data()

    print("Data loaded successfully.")
    print("Features shape:", X.shape)

    print("\nLoading trained model...")

    # Load the trained model.
    model = load_model()

    print("Model loaded successfully.")

    print("\nGenerating predictions...")

    # Predict on a small sample.
    results = predict_sample(model, X, n_samples=5)

    print("\nSample predictions:")
    print(results)


if __name__ == "__main__":
    main()