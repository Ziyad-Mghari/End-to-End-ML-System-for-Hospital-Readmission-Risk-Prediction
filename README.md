# Hospital Readmission Risk ML System

End-to-end machine learning system for hospital readmission risk prediction.

## Project Overview

This project aims to build a complete machine learning system to predict whether a patient is at risk of hospital readmission.

The goal is not only to train a model, but to structure the project like a real ML engineering project.

The project will include:

- exploratory data analysis
- data preprocessing
- model training and evaluation
- FastAPI deployment
- Docker
- automated tests
- basic monitoring
- clear documentation

## Problem Statement

Hospital readmissions are an important issue in healthcare.

In this project, we treat the task as a binary classification problem:

- low readmission risk
- high readmission risk

## Dataset

The dataset used is:

Diabetes 130-US Hospitals for Years 1999-2008

Source: UCI Machine Learning Repository

The dataset is not stored directly in this repository.

Expected files:

- data/raw/diabetic_data.csv
- data/raw/IDS_mapping.csv

## Planned Project Structure

- data/
- notebooks/
- src/
- api/
- tests/
- models/
- reports/
- Dockerfile
- requirements.txt
- README.md

## Machine Learning Approach

The project will compare several models:

- Logistic Regression
- Random Forest
- Gradient Boosting model

The models will be evaluated using:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

In a healthcare context, recall is especially important because the model should avoid missing high-risk patients.

## API

The trained model will be served using FastAPI.

The API will receive patient information and return:

- a risk score
- a risk category

## Deployment

The project will include Docker to make the API easy to run in a reproducible environment.

## Tests

The project will include basic tests using Pytest.

## Monitoring

A basic monitoring module will be added to track prediction distribution and potential data drift.

## Status

Project in progress.

Current stage:

- Repository initialized
- README created
- Dataset selected
- Exploratory analysis to be started

## Skills Demonstrated

- Machine Learning
- Data preprocessing
- Model evaluation
- FastAPI
- Docker
- Pytest
- ML Engineering
- Monitoring
- Project documentation

  Initial README
