# This script compares several machine learning models
# for hospital readmission risk prediction.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_preprocessing import preprocess_data
from config import RANDOM_STATE, TEST_SIZE, TRAIN_SAMPLE_SIZE


def build_preprocessor(numeric_features, categorical_features):
    """
    Build the preprocessing pipeline.

    Numerical variables are scaled.
    Categorical variables are one-hot encoded.
    """

    # StandardScaler puts numerical features on a comparable scale.
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # OneHotEncoder transforms categorical variables into numerical columns.
    # sparse_output=False makes the output easier to use with tree-based models.
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    # ColumnTransformer applies the right transformation to the right columns.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def build_models(preprocessor):
    """
    Build the models to compare.

    Each model is combined with the same preprocessing pipeline.
    """

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    solver="liblinear"
                ))
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ]
        ),

        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=RANDOM_STATE
                ))
            ]
        )
    }

    return models


def evaluate_model(model, X_test, y_test):
    """
    Evaluate one trained model on the test set.
    """

    # Predict final class labels: 0 or 1.
    y_pred = model.predict(X_test)

    # Predict risk scores for class 1 = high readmission risk.
    y_proba = model.predict_proba(X_test)[:, 1]

    # Store the metrics in a dictionary.
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    return metrics


def main():
    """
    Main function used to compare several models.

    Steps:
    1. Load and preprocess the data.
    2. Split data into train and test sets.
    3. Train several models.
    4. Evaluate each model.
    5. Display a comparison table.
    """

    print("Loading and preprocessing data...")

    # Load features, target, and feature type lists.
    X, y, numeric_features, categorical_features = preprocess_data()

    print("Data loaded successfully.")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # Split the data into training and testing sets.
    # stratify=y keeps the same target distribution in train and test.
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

    # Use a sample of the training set to keep comparison fast.
    # This is useful for the first version of the project.
    if TRAIN_SAMPLE_SIZE is not None:
        X_train_sample = X_train.sample(
            n=TRAIN_SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )
        y_train_sample = y_train.loc[X_train_sample.index]

        print(f"\nTraining models on a sample of {TRAIN_SAMPLE_SIZE} rows.")
    else:
        X_train_sample = X_train
        y_train_sample = y_train

        print("\nTraining models on the full training set.")

    # Build the preprocessing pipeline.
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Build all models to compare.
    models = build_models(preprocessor)

    # This list will store the results of all models.
    results = []

    # Train and evaluate each model.
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Train the model.
        model.fit(X_train_sample, y_train_sample)

        print(f"{model_name} trained successfully.")

        # Evaluate the model.
        metrics = evaluate_model(model, X_test, y_test)

        # Add the model name to the metrics.
        metrics["Model"] = model_name

        # Store results.
        results.append(metrics)

    # Convert results to a DataFrame.
    results_df = pd.DataFrame(results)

    # Put the model name as the first column.
    results_df = results_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    ]

    print("\nModel comparison results:")
    print(results_df.round(3))


if __name__ == "__main__":
    main()