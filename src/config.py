# This file centralizes important paths used in the project.
# The goal is to avoid rewriting file paths manually in every script.

from pathlib import Path


# PROJECT_ROOT corresponds to the main project folder.
# __file__ is the current file path: src/config.py
# parents[1] goes two levels up: src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Model folder
MODELS_DIR = PROJECT_ROOT / "models"


# Main raw dataset file
RAW_DATA_PATH = RAW_DATA_DIR / "diabetic_data.csv"


# Main processed dataset file
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.csv"


# Target column used for machine learning
TARGET_COLUMN = "target"