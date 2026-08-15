# These tests check that the preprocessing pipeline works correctly.

import sys
from pathlib import Path

# Add the src folder to Python's path so we can import project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_preprocessing import preprocess_data


def test_preprocess_data_shapes():
    """
    Check that preprocessing returns X and y with compatible shapes.
    """

    # Run the full preprocessing pipeline.
    X, y, numeric_features, categorical_features = preprocess_data()

    # X and y should have the same number of rows.
    assert X.shape[0] == y.shape[0]

    # The project baseline currently uses 38 features.
    assert X.shape[1] == 38

    # The target should contain only 0 and 1.
    assert set(y.unique()).issubset({0, 1})


def test_feature_lists_are_not_empty():
    """
    Check that numerical and categorical feature lists are correctly created.
    """

    # Run the full preprocessing pipeline.
    X, y, numeric_features, categorical_features = preprocess_data()

    # We expect at least one numerical feature.
    assert len(numeric_features) > 0

    # We expect at least one categorical feature.
    assert len(categorical_features) > 0


def test_no_duplicate_features():
    """
    Check that feature lists do not contain duplicates.
    """

    # Run the full preprocessing pipeline.
    X, y, numeric_features, categorical_features = preprocess_data()

    # Feature names should be unique in each list.
    assert len(numeric_features) == len(set(numeric_features))
    assert len(categorical_features) == len(set(categorical_features))