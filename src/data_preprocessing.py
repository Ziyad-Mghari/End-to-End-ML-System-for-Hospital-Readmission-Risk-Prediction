# This file contains reusable functions for loading and preprocessing the data.
# The goal is to move the cleaning logic out of the notebook and into Python scripts.

import pandas as pd
import numpy as np

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN


def load_raw_data():
    """
    Load the raw hospital readmission dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset loaded from data/raw/diabetic_data.csv.
    """

    # Read the CSV file from the raw data folder.
    df = pd.read_csv(RAW_DATA_PATH)

    return df


def create_binary_target(df):
    """
    Create the binary target variable.

    The original column 'readmitted' has three values:
    - 'NO'
    - '>30'
    - '<30'

    We define:
    - target = 1 if the patient was readmitted within 30 days
    - target = 0 otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with a new binary target column.
    """

    # Copy the dataframe to avoid modifying the original object directly.
    df = df.copy()

    # Create the binary target.
    df[TARGET_COLUMN] = (df["readmitted"] == "<30").astype(int)

    return df


def clean_missing_values(df):
    """
    Clean missing values in the dataset.

    In this dataset, missing values are encoded as '?'.
    We replace them with np.nan, remove columns with too many missing values,
    and fill missing diagnosis values with 'Unknown'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    # Copy the dataframe to avoid modifying the original object directly.
    df = df.copy()

    # Replace '?' by real missing values.
    df = df.replace("?", np.nan)

    # Remove columns that are not kept in the first version of the project.
    columns_to_drop = [
        "weight",
        "payer_code",
        "medical_specialty",
        "race"
    ]

    df = df.drop(columns=columns_to_drop)

    # Fill missing diagnosis values with "Unknown".
    diagnosis_columns = ["diag_1", "diag_2", "diag_3"]

    df[diagnosis_columns] = df[diagnosis_columns].fillna("Unknown")

    return df


def select_features_for_baseline(df):
    """
    Remove columns that should not be used in the first baseline model.

    These columns are removed for different reasons:
    - identifiers
    - data leakage
    - too many categories
    - no useful information

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Dataset ready for the first baseline model.
    """

    # Copy the dataframe to avoid modifying the original object directly.
    df = df.copy()

    columns_to_remove_for_model = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "diag_1",
        "diag_2",
        "diag_3",
        "examide",
        "citoglipton"
    ]

    df_model = df.drop(columns=columns_to_remove_for_model)

    return df_model


def split_features_and_target(df_model):
    """
    Separate the input features X and the target y.

    Parameters
    ----------
    df_model : pd.DataFrame
        Dataset prepared for modeling.

    Returns
    -------
    X : pd.DataFrame
        Input features used by the model.
    y : pd.Series
        Target variable.
    """

    # X contains all columns except the target.
    X = df_model.drop(columns=[TARGET_COLUMN])

    # y contains only the target.
    y = df_model[TARGET_COLUMN]

    return X, y


def get_feature_types(X):
    """
    Identify numerical and categorical features.

    Some columns are numerical codes but should be treated as categorical variables.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.

    Returns
    -------
    numeric_features : list
        List of numerical feature names.
    categorical_features : list
        List of categorical feature names.
    """

    # These columns are numeric codes, not continuous numerical quantities.
    categorical_id_features = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id"
    ]

    # Detect numerical columns.
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Detect categorical columns.
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Remove ID-like categorical features from numerical features.
    numeric_features = [
        col for col in numeric_features
        if col not in categorical_id_features
    ]

    # Add these ID-like features to categorical features.
    categorical_features = categorical_features + categorical_id_features

    # Remove possible duplicates.
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))

    return numeric_features, categorical_features


def preprocess_data():
    """
    Full preprocessing pipeline for the first baseline model.

    This function:
    1. loads the raw data
    2. creates the binary target
    3. cleans missing values
    4. selects baseline features
    5. separates X and y
    6. identifies numerical and categorical features

    Returns
    -------
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target variable.
    numeric_features : list
        Numerical feature names.
    categorical_features : list
        Categorical feature names.
    """

    # Load raw data.
    df = load_raw_data()

    # Create binary target.
    df = create_binary_target(df)

    # Clean missing values.
    df_clean = clean_missing_values(df)

    # Select features for the first baseline model.
    df_model = select_features_for_baseline(df_clean)

    # Separate X and y.
    X, y = split_features_and_target(df_model)

    # Identify numerical and categorical features.
    numeric_features, categorical_features = get_feature_types(X)

    return X, y, numeric_features, categorical_features


if __name__ == "__main__":
    # This block runs only when we execute this file directly.
    # It is useful for checking that the preprocessing works.

    X, y, numeric_features, categorical_features = preprocess_data()

    print("Preprocessing completed successfully.")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Number of numerical features:", len(numeric_features))
    print("Number of categorical features:", len(categorical_features))
    print("Target distribution:")
    print(y.value_counts(normalize=True))
    