# End-to-End ML System for Hospital Readmission Risk Prediction

End-to-end machine learning project for predicting hospital readmission risk using clinical tabular data.

## Project Overview

This project focuses on predicting whether a diabetic patient is at risk of being readmitted to the hospital within 30 days.

The goal of this first version is to build a clean and understandable machine learning baseline, starting from exploratory data analysis and ending with a first evaluated model.

The current version includes:

* project repository setup
* structured project folders
* exploratory data analysis
* missing values analysis
* binary target creation
* feature selection
* preprocessing pipeline
* Logistic Regression baseline model
* model evaluation with classification metrics
* confusion matrix
* ROC curve

## Problem Statement

Hospital readmissions are an important healthcare issue because they can indicate complications, insufficient follow-up, or worsening patient condition.

In this project, the task is treated as a binary classification problem:

* `0`: the patient was not readmitted within 30 days
* `1`: the patient was readmitted within 30 days

The objective is to identify patients with higher risk of early hospital readmission.

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

For this project, the target was transformed into a binary variable:

```python
target = 1 if readmitted == "<30" else 0
```

This means that the model focuses specifically on early readmission within 30 days.

The dataset is not stored directly in this repository.

Expected local files:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

## Project Structure

```text
End-to-End-ML-System-for-Hospital-Readmission-Risk-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│
├── api/
│
├── tests/
│
├── models/
│
├── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Exploratory Data Analysis

The exploratory data analysis was performed in:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

The analysis covered:

* dataset loading
* dataset shape
* column inspection
* target variable analysis
* missing values detection
* class imbalance analysis
* numerical feature analysis
* categorical feature analysis
* feature selection for the first baseline model

## Missing Values

In this dataset, some missing values are encoded as `"?"`.

The main columns containing missing values were:

```text
weight               98569
medical_specialty    49949
payer_code           40256
race                  2273
diag_3                1423
diag_2                 358
diag_1                  21
```

The following columns were removed in the first version:

```text
weight
payer_code
medical_specialty
race
```

Reasons:

* `weight` contains too many missing values
* `payer_code` is not central for a first medical baseline
* `medical_specialty` contains many missing values
* `race` is a sensitive attribute and was removed to reduce the risk of unfair bias

The diagnosis columns `diag_1`, `diag_2`, and `diag_3` contained fewer missing values. Their missing values were replaced with `"Unknown"` during the cleaning step.

## Target Distribution

The binary target is imbalanced:

| Class | Meaning                       | Proportion |
| ----- | ----------------------------- | ---------: |
| 0     | Not readmitted within 30 days |     88.84% |
| 1     | Readmitted within 30 days     |     11.16% |

This imbalance means that accuracy alone is not enough to evaluate the model.

For this reason, the model was evaluated using:

* accuracy
* precision
* recall
* F1-score
* ROC-AUC
* confusion matrix

In this healthcare context, recall is especially important because false negatives correspond to high-risk patients that the model fails to detect.

## Numerical Feature Analysis

Several numerical variables were compared between low-risk and high-risk patients.

The most important observation was related to previous hospital usage.

For example, the average value of `number_inpatient` was:

| Target | Average `number_inpatient` |
| ------ | -------------------------: |
| 0      |                       0.56 |
| 1      |                       1.22 |

Patients readmitted within 30 days had, on average, more than twice as many previous inpatient admissions.

The variable `number_emergency` was also higher for high-risk patients:

| Target | Average `number_emergency` |
| ------ | -------------------------: |
| 0      |                       0.18 |
| 1      |                       0.36 |

These first observations suggest that previous healthcare utilization is an important signal for predicting early hospital readmission.

## Categorical Feature Analysis

The categorical variables were analyzed by counting the number of unique values in each column.

Some variables had a small number of categories, such as:

```text
age
gender
diabetesMed
change
A1Cresult
max_glu_serum
```

Other variables had a very large number of categories:

```text
diag_1    717
diag_2    749
diag_3    790
```

Because the diagnosis columns contain many distinct medical codes, they were removed from the first baseline model to keep the preprocessing simple and interpretable.

## Feature Selection

For the first baseline model, the following columns were removed:

```text
encounter_id
patient_nbr
readmitted
diag_1
diag_2
diag_3
examide
citoglipton
```

Reasons:

* `encounter_id` and `patient_nbr` are identifiers
* `readmitted` would cause data leakage because it was used to create the binary target
* `diag_1`, `diag_2`, and `diag_3` contain many distinct categories
* `examide` and `citoglipton` contain only one unique value and do not provide useful information

After feature selection, the dataset used for modeling contained:

```text
101766 rows
38 features
```

## Preprocessing Pipeline

A preprocessing pipeline was created using scikit-learn.

The pipeline applies different transformations depending on the type of variable:

* numerical features are standardized with `StandardScaler`
* categorical features are encoded with `OneHotEncoder`

The columns `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` were treated as categorical variables because they are numerical codes, not continuous numerical quantities.

The preprocessing was combined with the model inside a scikit-learn `Pipeline`.

This makes the workflow cleaner and reduces the risk of applying inconsistent transformations between training and testing.

## Baseline Model

The first model trained was a Logistic Regression baseline.

Logistic Regression was used because it is:

* simple
* fast to train
* interpretable
* useful as a first reference model

Because the target variable is imbalanced, the model was trained with class balancing.

## Baseline Results

The Logistic Regression baseline was evaluated on a separate test set.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |     0.67 |      0.18 |   0.56 |     0.28 |    0.68 |

## Results Interpretation

The baseline model achieved a ROC-AUC of approximately 0.68.

This indicates a moderate ability to distinguish between patients with low and high readmission risk.

The recall for the high-risk class is approximately 0.56. This means that the model detected more than half of the patients who were readmitted within 30 days.

However, the precision is low, around 0.18. This means that many patients predicted as high-risk were not actually readmitted within 30 days.

This trade-off is understandable for a first baseline on an imbalanced healthcare dataset. In a healthcare context, detecting high-risk patients is important, but the model still produces many false positives.

## Model Comparison

Several models were compared using the same preprocessing strategy and the same train/test split.

The compared models were:

* Logistic Regression
* Random Forest
* Gradient Boosting

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.665 |     0.181 |  0.568 |    0.275 |   0.668 |
| Random Forest       |    0.677 |     0.183 |  0.548 |    0.275 |   0.670 |
| Gradient Boosting   |    0.888 |     0.458 |  0.010 |    0.019 |   0.671 |

### Model Comparison Interpretation

Logistic Regression and Random Forest produced very similar results.

Random Forest achieved a slightly higher accuracy and ROC-AUC, but its recall was slightly lower than Logistic Regression.

Gradient Boosting achieved a high accuracy, but its recall for the high-risk class was extremely low. This means that the model predicted almost all patients as low risk and failed to detect most patients readmitted within 30 days.

Because the dataset is highly imbalanced, accuracy alone is not a reliable metric. In this healthcare context, recall is especially important because the objective is to detect patients at high risk of early readmission.

For this reason, Logistic Regression remains a strong baseline model at this stage of the project.

## Threshold Tuning

The default classification threshold is 0.50. This means that a patient is classified as high risk when the predicted probability is greater than or equal to 0.50.

However, in a healthcare context, changing the threshold can be useful. A lower threshold can increase recall, meaning that more high-risk patients are detected, at the cost of more false positives.

Several thresholds were tested for the Logistic Regression baseline.

| Threshold | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives |
| --------: | -------: | --------: | -----: | -------: | --------------: | --------------: | -------------: |
|      0.45 |    0.548 |     0.158 |  0.706 |    0.259 |            8523 |             668 |           1603 |
|      0.50 |    0.665 |     0.181 |  0.568 |    0.275 |            5840 |             980 |           1291 |
|      0.55 |    0.748 |     0.203 |  0.430 |    0.276 |            3840 |            1294 |            977 |

### Threshold Tuning Interpretation

The threshold of 0.55 achieved the best F1-score, but it reduced recall to 0.430.

The threshold of 0.45 is more interesting for a healthcare-oriented use case because it increases recall to 0.706. This means that the model detects around 70% of patients who are readmitted within 30 days.

Compared with the default threshold of 0.50, the threshold of 0.45 detects more high-risk patients:

* threshold 0.50: 1291 true positives
* threshold 0.45: 1603 true positives

This corresponds to 312 additional high-risk patients detected.

However, this improvement comes with more false positives:

* threshold 0.50: 5840 false positives
* threshold 0.45: 8523 false positives

This illustrates the trade-off between recall and precision. In a medical screening context, increasing recall can be useful, but the number of false alerts must also be considered.

## API Usage

A FastAPI application was created to expose the trained model through a prediction endpoint.

The API is defined in:

```text
api/main.py
```

Before starting the API, the baseline model must be trained and saved locally:

```bash
python src/train.py
```

Then the API can be launched with:

```bash
uvicorn api.main:app --reload
```

Once the server is running, the interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The API currently provides the following endpoints:

| Endpoint          | Method | Description                                   |
| ----------------- | ------ | --------------------------------------------- |
| `/`               | GET    | Checks that the API is running                |
| `/health`         | GET    | Checks whether the model is loaded            |
| `/sample-patient` | GET    | Returns one example patient from the dataset  |
| `/predict`        | POST   | Predicts the readmission risk for one patient |

### Prediction Request Example

The `/predict` endpoint expects a JSON object containing:

* `data`: the patient features
* `threshold`: the decision threshold used to classify the patient as low risk or high risk

Example request:

```json
{
  "data": {
    "gender": "Female",
    "age": "[0-10)",
    "admission_type_id": 6,
    "discharge_disposition_id": 25,
    "admission_source_id": 1,
    "time_in_hospital": 1,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 1,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 1,
    "max_glu_serum": null,
    "A1Cresult": null,
    "metformin": "No",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "No",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "insulin": "No",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "No"
  },
  "threshold": 0.45
}
```

Example response:

```json
{
  "risk_score": 0.37,
  "prediction": 0,
  "risk_label": "low_risk",
  "threshold": 0.45
}
```

The `risk_score` corresponds to the predicted probability of early readmission within 30 days.

The `threshold` controls how this probability is converted into a binary prediction:

* if `risk_score >= threshold`, the patient is classified as `high_risk`
* if `risk_score < threshold`, the patient is classified as `low_risk`

## Tests

Automated tests were added using Pytest.

The current test suite checks:

- the preprocessing pipeline
- feature and target shapes
- numerical and categorical feature detection
- absence of duplicate features
- FastAPI health endpoint
- FastAPI sample patient endpoint
- FastAPI prediction endpoint

The tests can be launched with:

```bash
pytest

## Docker Usage

Docker support was added to make the FastAPI application easier to run in a reproducible environment.

The Docker configuration is defined in:

```text
Dockerfile
.dockerignore
```

The Docker image contains:

* the FastAPI application
* the reusable Python scripts from `src/`
* the trained model
* the model metadata
* the required Python dependencies

Before building the Docker image, the model must be trained locally:

```bash
python src/train.py
```

This creates the local model files used by the API:

```text
models/baseline_logistic_regression.joblib
models/model_metadata.joblib
```

Then the Docker image can be built with:

```bash
docker build -t hospital-readmission-api .
```

The container can be started with:

```bash
docker run --rm -p 8000:8000 hospital-readmission-api
```

Once the container is running, the API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The Dockerized API was tested locally with the following endpoints:

* `GET /health`
* `GET /sample-patient`
* `POST /predict`

This confirms that the trained model can be served through FastAPI inside a Docker container.

## Basic Monitoring

A basic monitoring module was added to track predictions made by the API.

The monitoring logic is implemented in:

```text
src/monitoring.py
Each time the /predict endpoint is called, the API logs useful prediction information into a local CSV file:

monitoring/predictions_log.csv

The logged information includes:

timestamp
risk score
predicted class
risk label
threshold used
selected patient features such as age, gender, time in hospital, number of medications, previous inpatient visits, emergency visits, and diabetes medication status

The API also includes a monitoring endpoint:

Endpoint	Method	Description
/monitoring/summary	GET	Returns basic statistics about logged predictions

The monitoring summary includes:

number of predictions
average risk score
high-risk prediction rate
low-risk prediction rate
average threshold used

Example response:

{
  "number_of_predictions": 3,
  "average_risk_score": 0.42,
  "high_risk_rate": 0.33,
  "low_risk_rate": 0.67,
  "average_threshold": 0.45
}

The prediction log file is ignored by Git because it is generated locally during API usage.

## Current Status

The current version of the project includes:

* GitHub repository initialized
* project folder structure created
* README created and updated
* dataset selected
* exploratory data analysis completed
* missing values analyzed
* binary target created
* class imbalance analyzed
* numerical and categorical features analyzed
* feature selection completed for a first baseline
* preprocessing pipeline created
* Logistic Regression baseline trained
* baseline model evaluated
* confusion matrix generated
* ROC curve generated
* reusable preprocessing script created in `src/data_preprocessing.py`
* training script created in `src/train.py`
* prediction script created in `src/predict.py`
* model comparison script created in `src/compare_models.py`
* threshold tuning script created in `src/threshold_tuning.py`
* Logistic Regression, Random Forest, and Gradient Boosting compared
* threshold tuning performed on the Logistic Regression baseline
* FastAPI application created in `api/main.py`
* health check endpoint added
* sample patient endpoint added
* prediction endpoint added
* API tested locally with Swagger UI
* automated preprocessing tests added
* automated API tests added
* test suite successfully executed with Pytest
* Dockerfile created
* .dockerignore created
* FastAPI application successfully containerized
* Docker image built locally
* Docker container tested locally
* Dockerized API tested with Swagger UI

## Skills Demonstrated

This project demonstrates skills in:

* Python
* Pandas
* NumPy
* Scikit-learn
* exploratory data analysis
* missing values handling
* feature selection
* preprocessing pipelines
* one-hot encoding
* feature scaling
* binary classification
* imbalanced classification
* model evaluation
* model comparison
* threshold tuning
* confusion matrix analysis
* ROC-AUC analysis
* reusable Python scripts
* Git and GitHub
* project documentation
* Pytest
* API testing
* automated testing
* Docker
* containerization
* API deployment basics
* reproducible environments
* basic ML monitoring
* prediction logging
* monitoring endpoints

## Next Steps

The next step is to write a results report summarizing the full project workflow, model results, limitations, and future improvements.

Initial README