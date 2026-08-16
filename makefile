# Makefile for the Hospital Readmission Risk ML System
# It provides simple shortcuts to run common project commands.

PYTHON = python
IMAGE_NAME = hospital-readmission-api

.PHONY: help install preprocess train predict compare threshold test api docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install        Install Python dependencies"
	@echo "  make preprocess     Run data preprocessing check"
	@echo "  make train          Train and save the baseline model"
	@echo "  make predict        Load the saved model and run sample predictions"
	@echo "  make compare        Compare multiple models"
	@echo "  make threshold      Run threshold tuning"
	@echo "  make test           Run automated tests"
	@echo "  make api            Start the FastAPI app locally"
	@echo "  make docker-build   Build the Docker image"
	@echo "  make docker-run     Run the Docker container"

install:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) src/data_preprocessing.py

train:
	$(PYTHON) src/train.py

predict:
	$(PYTHON) src/predict.py

compare:
	$(PYTHON) src/compare_models.py

threshold:
	$(PYTHON) src/threshold_tuning.py

test:
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn api.main:app --reload

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE_NAME)
