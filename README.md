# End-to-End ML System for Hospital Readmission Risk Prediction

End-to-end machine learning project for predicting hospital readmission risk using clinical tabular data.

The objective is to predict whether a diabetic patient is at risk of being readmitted to the hospital within 30 days.

This project is structured as a production-oriented machine learning workflow, including exploratory analysis, preprocessing, model training, API serving, Docker support, automated tests, and basic monitoring.

---

## Project Overview

This project covers the full machine learning workflow:

* exploratory data analysis
* missing values handling
* binary target creation
* feature selection
* preprocessing pipeline
* baseline model training
* model comparison
* threshold tuning
* model saving and prediction script
* FastAPI prediction endpoint
* Docker containerization
* automated tests
* basic prediction monitoring

The goal is not only to train a model, but to build a clean and reusable ML system.

---

## Dataset

The dataset used is:

**Diabetes 130-US Hospitals for Years 1999-2008**

Source: UCI Machine Learning Repository

The dataset contains hospital records of diabetic patients collected from 130 US hospitals between 1999 and 2008.

The original target column is:

```text
readmitted
```

It contains three possible values:

```text
NO
>30
<30
```

For this project, the target was converted into a binary classification problem:

```python
target = 1 if readmitted == "<30" else 0
```

Meaning:

* `0`: patient not readmitted within 30 days
* `1`: patient readmitted within 30 days

The dataset is not stored directly in this repository.

Expected local files:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

---

## Project Structure

```text
End-to-End-ML-System-for-Hospital-Readmission-Risk-Prediction/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── monitoring/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── reports/
│   └── results_report.md
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── compare_models.py
│   ├── threshold_tuning.py
│   └── monitoring.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis

The exploratory analysis is available in:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

The analysis includes:

* dataset inspection
* missing values analysis
* target distribution analysis
* numerical feature analysis
* categorical feature analysis
* feature selection for the first baseline model

The binary target is highly imbalanced:

| Class | Meaning                       | Proportion |
| ----- | ----------------------------- | ---------: |
| 0     | Not readmitted within 30 days |     88.84% |
| 1     | Readmitted within 30 days     |     11.16% |

Because of this imbalance, accuracy alone is not sufficient. The project focuses on precision, recall, F1-score, ROC-AUC, and the confusion matrix.

---

## Preprocessing

The preprocessing logic is implemented in:

```text
src/data_preprocessing.py
```

The preprocessing includes:

* replacing missing values encoded as `"?"`
* removing columns with too many missing values
* removing sensitive or non-informative columns
* creating the binary target
* selecting features for the baseline model
* separating features and target
* identifying numerical and categorical variables

Numerical features are standardized with `StandardScaler`.

Categorical features are encoded with `OneHotEncoder`.

The preprocessing and model are combined inside a scikit-learn `Pipeline`.

---

## Model Training

The baseline training script is:

```text
src/train.py
```

It performs:

* data loading
* preprocessing
* train/test split
* baseline model training
* model evaluation
* model saving

The baseline model is a Logistic Regression model with class balancing.

Run training with:

```bash
python src/train.py
```

or, if `make` is available:

```bash
make train
```

The trained model is saved locally in:

```text
models/baseline_logistic_regression.joblib
```

A metadata file is also saved locally:

```text
models/model_metadata.joblib
```

---

## Model Results

Several models were compared using the same preprocessing strategy.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.665 |     0.181 |  0.568 |    0.275 |   0.668 |
| Random Forest       |    0.677 |     0.183 |  0.548 |    0.275 |   0.670 |
| Gradient Boosting   |    0.888 |     0.458 |  0.010 |    0.019 |   0.671 |

Logistic Regression and Random Forest produced similar results.

Gradient Boosting achieved high accuracy but very low recall, meaning that it failed to detect most high-risk patients.

Because this is an imbalanced healthcare problem, recall is especially important. At this stage, Logistic Regression remains a strong baseline.

---

## Threshold Tuning

Threshold tuning was performed on the Logistic Regression baseline.

The default classification threshold is `0.50`.

A lower threshold can increase recall, meaning that more high-risk patients are detected, but it also increases false positives.

| Threshold | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives |
| --------: | -------: | --------: | -----: | -------: | --------------: | --------------: | -------------: |
|      0.45 |    0.548 |     0.158 |  0.706 |    0.259 |            8523 |             668 |           1603 |
|      0.50 |    0.665 |     0.181 |  0.568 |    0.275 |            5840 |             980 |           1291 |
|      0.55 |    0.748 |     0.203 |  0.430 |    0.276 |            3840 |            1294 |            977 |

A threshold of `0.45` increases recall to approximately `0.706`, detecting more high-risk patients at the cost of more false positives.

---

## Prediction Script

The prediction script is:

```text
src/predict.py
```

It loads the saved model and runs predictions on a small sample of patients.

Run it with:

```bash
python src/predict.py
```

or:

```bash
make predict
```

---

## API Usage

A FastAPI application was created in:

```text
api/main.py
```

Start the API locally with:

```bash
python -m uvicorn api.main:app --reload
```

or:

```bash
make api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

| Endpoint              | Method | Description                         |
| --------------------- | ------ | ----------------------------------- |
| `/`                   | GET    | Checks that the API is running      |
| `/health`             | GET    | Checks whether the model is loaded  |
| `/sample-patient`     | GET    | Returns one example patient         |
| `/predict`            | POST   | Predicts readmission risk           |
| `/monitoring/summary` | GET    | Returns basic monitoring statistics |

Example prediction response:

```json
{
  "risk_score": 0.37,
  "prediction": 0,
  "risk_label": "low_risk",
  "threshold": 0.45
}
```

---

## Docker Usage

Docker support was added to run the FastAPI app in a reproducible environment.

Before building the Docker image, train the model locally:

```bash
python src/train.py
```

Build the Docker image:

```bash
docker build -t hospital-readmission-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 hospital-readmission-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The Dockerized API was tested locally with:

* `GET /health`
* `GET /sample-patient`
* `POST /predict`

---

## Tests

Automated tests were added using Pytest.

Test files:

```text
tests/test_preprocessing.py
tests/test_api.py
```

The tests check:

* preprocessing output shapes
* target values
* feature lists
* duplicate feature detection
* FastAPI health endpoint
* sample patient endpoint
* prediction endpoint

Run tests with:

```bash
pytest
```

or:

```bash
make test
```

---

## Basic Monitoring

A basic monitoring module was added in:

```text
src/monitoring.py
```

Each call to `/predict` logs useful prediction information into:

```text
monitoring/predictions_log.csv
```

The log includes:

* timestamp
* risk score
* prediction
* risk label
* threshold
* selected patient features

The endpoint:

```text
GET /monitoring/summary
```

returns summary statistics such as:

* number of predictions
* average risk score
* high-risk prediction rate
* low-risk prediction rate
* average threshold used

The prediction log file is ignored by Git because it is generated locally.

---

## Makefile Commands

A `Makefile` was added to simplify common commands.

Available commands:

```bash
make install
make preprocess
make train
make predict
make compare
make threshold
make test
make api
make docker-build
make docker-run
```

If `make` is not installed, the equivalent Python and Docker commands can still be run manually.

---

## Detailed Report

A complete technical report is available here:

```text
reports/results_report.md
```

It includes detailed explanations of:

* exploratory data analysis
* preprocessing choices
* model comparison
* threshold tuning
* API usage
* Docker support
* tests
* monitoring
* limitations
* future improvements

---

## Skills Demonstrated

This project demonstrates skills in:

* Python
* Pandas
* NumPy
* Scikit-learn
* exploratory data analysis
* data preprocessing
* feature selection
* machine learning pipelines
* imbalanced classification
* model evaluation
* model comparison
* threshold tuning
* FastAPI
* Docker
* Pytest
* API testing
* basic ML monitoring
* Git and GitHub
* project documentation

---

## Current Status

The current version includes:

* exploratory data analysis
* reusable preprocessing script
* baseline training script
* prediction script
* model comparison script
* threshold tuning script
* FastAPI application
* Docker support
* automated tests
* basic prediction monitoring
* detailed results report
* Makefile shortcuts

---

## Next Steps

Possible next improvements include:

* training models on the full dataset
* improving diagnosis code feature engineering
* adding hyperparameter tuning
* adding model calibration
* adding GitHub Actions for continuous testing
* deploying the API online


Initial README