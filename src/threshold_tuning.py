# This script performs threshold tuning for the baseline Logistic Regression model.
# The goal is to study how changing the classification threshold affects precision, recall, and F1-score.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from data_preprocessing import preprocess_data
from train import build_baseline_model
from config import RANDOM_STATE, TEST_SIZE, TRAIN_SAMPLE_SIZE


def evaluate_threshold(y_true, y_proba, threshold):
    """
    Evaluate model predictions for a given classification threshold.

    Parameters
    ----------
    y_true : pd.Series
        True labels from the test set.

    y_proba : array-like
        Predicted probabilities for class 1 = high readmission risk.

    threshold : float
        Classification threshold used to convert probabilities into class labels.

    Returns
    -------
    dict
        Evaluation metrics for the selected threshold.
    """

    # Convert probabilities into binary predictions using the selected threshold.
    # If the predicted probability is greater than or equal to the threshold,
    # the patient is classified as high-risk.
    y_pred_threshold = (y_proba >= threshold).astype(int)

    # Compute the confusion matrix.
    # The matrix gives:
    # TN = true low-risk correctly predicted
    # FP = low-risk incorrectly predicted as high-risk
    # FN = high-risk incorrectly predicted as low-risk
    # TP = true high-risk correctly predicted
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_threshold).ravel()

    # Store all useful metrics in a dictionary.
    metrics = {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred_threshold),
        "Precision": precision_score(y_true, y_pred_threshold, zero_division=0),
        "Recall": recall_score(y_true, y_pred_threshold),
        "F1-score": f1_score(y_true, y_pred_threshold),
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    }

    return metrics


def main():
    """
    Main function for threshold tuning.

    Steps:
    1. Load and preprocess the data.
    2. Split the data into train and test sets.
    3. Train the baseline Logistic Regression model.
    4. Get predicted probabilities on the test set.
    5. Evaluate several thresholds.
    6. Display a comparison table.
    """

    print("Loading and preprocessing data...")

    # Load features, target, and feature type lists.
    X, y, numeric_features, categorical_features = preprocess_data()

    print("Data loaded successfully.")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # Split the data into training and testing sets.
    # stratify=y keeps the same class distribution in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain/test split completed.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Use a training sample to keep execution fast.
    if TRAIN_SAMPLE_SIZE is not None:
        X_train_sample = X_train.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )
        y_train_sample = y_train.loc[X_train_sample.index]

        print(f"\nTraining on a sample of {TRAIN_SAMPLE_SIZE} rows.")
    else:
        X_train_sample = X_train
        y_train_sample = y_train

        print("\nTraining on the full training set.")

    # Build the same baseline model as in train.py.
    model = build_baseline_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    print("\nTraining baseline model...")

    # Train the Logistic Regression model.
    model.fit(X_train_sample, y_train_sample)

    print("Model trained successfully.")

    # Get predicted probabilities for class 1 = high readmission risk.
    y_proba = model.predict_proba(X_test)[:, 1]

    # Define thresholds to test.
    # Lower thresholds usually increase recall but reduce precision.
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    # Evaluate every threshold and store the results.
    results = []

    for threshold in thresholds:
        metrics = evaluate_threshold(
            y_true=y_test,
            y_proba=y_proba,
            threshold=threshold
        )

        results.append(metrics)

    # Convert results into a DataFrame for readability.
    results_df = pd.DataFrame(results)

    # Display rounded metrics.
    print("\nThreshold tuning results:")
    print(results_df.round(3))

    # Identify the threshold with the highest F1-score.
    best_f1_row = results_df.loc[results_df["F1-score"].idxmax()]

    print("\nBest threshold according to F1-score:")
    print(best_f1_row.round(3))

    # Identify the threshold with recall greater than or equal to 0.70 if possible.
    high_recall_candidates = results_df[results_df["Recall"] >= 0.70]

    if not high_recall_candidates.empty:
        # Among thresholds with recall >= 0.70, choose the one with the highest precision.
        best_high_recall_row = high_recall_candidates.loc[
            high_recall_candidates["Precision"].idxmax()
        ]

        print("\nBest threshold with recall >= 0.70:")
        print(best_high_recall_row.round(3))
    else:
        print("\nNo threshold reached recall >= 0.70.")


if __name__ == "__main__":
    main()
    