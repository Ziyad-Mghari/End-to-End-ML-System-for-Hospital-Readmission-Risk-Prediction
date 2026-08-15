# This script trains the first baseline model for hospital readmission risk prediction.
# It uses the preprocessing functions defined in data_preprocessing.py.

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from data_preprocessing import preprocess_data
from config import (
    MODEL_PATH,
    RANDOM_STATE,
    MODEL_METADATA_PATH,
    TEST_SIZE,
    TRAIN_SAMPLE_SIZE
)


def build_baseline_model(numeric_features, categorical_features):
    """
    Build the preprocessing and Logistic Regression pipeline.

    Parameters
    ----------
    numeric_features : list
        List of numerical feature names.

    categorical_features : list
        List of categorical feature names.

    Returns
    -------
    Pipeline
        Complete scikit-learn pipeline including preprocessing and model.
    """

    # Numerical features are standardized.
    # StandardScaler puts numerical variables on a comparable scale.
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # Categorical features are transformed with one-hot encoding.
    # handle_unknown="ignore" avoids errors if a new category appears later.
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # ColumnTransformer applies the correct preprocessing to each type of column.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Logistic Regression is used as the first baseline model.
    # class_weight="balanced" helps with the imbalanced target.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="liblinear"
            ))
        ]
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Parameters
    ----------
    model : Pipeline
        Trained scikit-learn model pipeline.

    X_test : pd.DataFrame
        Test features.

    y_test : pd.Series
        True test labels.

    Returns
    -------
    None
        Prints evaluation metrics.
    """

    # Predict class labels: 0 or 1.
    y_pred = model.predict(X_test)

    # Predict probabilities for class 1 = high readmission risk.
    y_proba = model.predict_proba(X_test)[:, 1]

    # Compute classification metrics.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nBaseline Logistic Regression results")
    print("------------------------------------")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))


def main():
    """
    Main training function.

    This function:
    1. loads and preprocesses the data
    2. splits the data into train and test sets
    3. optionally samples the training data
    4. trains the baseline model
    5. evaluates the model
    6. saves the trained model
    """

    print("Loading and preprocessing data...")

    # Load data and get feature types from the preprocessing script.
    X, y, numeric_features, categorical_features = preprocess_data()

    print("Data loaded successfully.")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # Split data into train and test sets.
    # stratify=y keeps the same class proportions in train and test sets.
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

    # Use a sample of the training set for the first baseline.
    # This keeps training fast and avoids long execution times.
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

    # Build the baseline model pipeline.
    model = build_baseline_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    print("\nTraining baseline model...")

    # Train the model.
    model.fit(X_train_sample, y_train_sample)

    print("Model trained successfully.")

    # Evaluate the trained model.
    evaluate_model(model, X_test, y_test)

    # Save the trained model to the models folder.
    
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved successfully at: {MODEL_PATH}")

    # Save metadata needed by the API.
    # The API needs to know the exact input columns expected by the model.
    sample_patient = {}

    for key, value in X.iloc[0].to_dict().items():
        # Convert NumPy values into standard Python values.
        if hasattr(value, "item"):
            value = value.item()

        # Convert missing values into None for JSON compatibility.
        if pd.isna(value):
            value = None

        sample_patient[key] = value

    metadata = {
        "expected_columns": X.columns.tolist(),
        "sample_patient": sample_patient
    }

    joblib.dump(metadata, MODEL_METADATA_PATH)

    print(f"Model metadata saved successfully at: {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()