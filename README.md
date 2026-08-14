# End-to-End ML System for Hospital Readmission Risk Prediction

End-to-end machine learning project for predicting hospital readmission risk using clinical tabular data.

## Project Overview

This project aims to build a complete machine learning system to predict whether a diabetic patient is at risk of being readmitted to the hospital within 30 days.

The goal is not only to train a predictive model, but to structure the project like a real ML engineering project.

The project includes:

* exploratory data analysis
* data cleaning and preprocessing
* feature selection
* model training and evaluation
* baseline model comparison
* future FastAPI deployment
* future Docker support
* future automated tests
* future basic monitoring
* clear documentation

## Problem Statement

Hospital readmissions are an important issue in healthcare because they can indicate complications, insufficient follow-up, or worsening patient condition.

In this project, the task is treated as a binary classification problem:

* `0`: low risk, the patient was not readmitted within 30 days
* `1`: high risk, the patient was readmitted within 30 days

The main objective is to identify patients at high risk of early readmission.

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

This means that the model focuses on predicting early readmission within 30 days.

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

The exploratory analysis focused on:

* understanding the dataset structure
* identifying missing values
* creating the binary target variable
* analyzing class imbalance
* comparing numerical features between low-risk and high-risk patients
* identifying categorical variables
* selecting features for a first baseline model

### Missing Values

Some missing values in the dataset are encoded as `"?"`.

The main columns with missing values were:

```text
weight
medical_specialty
payer_code
race
diag_1
diag_2
diag_3
```

For the first version of the project:

* `weight` was removed because it contains too many missing values
* `payer_code` was removed because it is not central for a first medical baseline
* `medical_specialty` was removed because it contains many missing values
* `race` was removed because it is a sensitive attribute
* missing diagnosis values were replaced with `"Unknown"`

### Target Distribution

The binary target is imbalanced:

| Class | Meaning                       | Proportion |
| ----- | ----------------------------- | ---------: |
| 0     | Not readmitted within 30 days |     88.84% |
| 1     | Readmitted within 30 days     |     11.16% |

Because of this imbalance, accuracy alone is not sufficient to evaluate the model.

The project therefore focuses on additional metrics:

* precision
* recall
* F1-score
* ROC-AUC
* confusion matrix

In a healthcare context, recall is especially important because false negatives correspond to high-risk patients that the model fails to detect.

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
* `readmitted` would cause data leakage because it was used to create the target
* `diag_1`, `diag_2`, and `diag_3` contain many distinct categories and were removed for the first simple baseline
* `examide` and `citoglipton` contain only one unique value and do not provide useful information

## Baseline Model

The first baseline model is a Logistic Regression model.

The preprocessing pipeline includes:

* standardization of numerical features
* one-hot encoding of categorical features
* class balancing to account for target imbalance

The baseline model is useful as a first reference before testing more complex models.

## Baseline Results

A first Logistic Regression baseline was trained and evaluated on a separate test set.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |     0.67 |      0.18 |   0.56 |     0.28 |    0.68 |

### Interpretation

The baseline model achieves a ROC-AUC of approximately 0.68, which indicates a moderate ability to distinguish between low-risk and high-risk patients.

The recall for the high-risk class is approximately 0.56. This means that the model detects more than half of the patients who are readmitted within 30 days.

However, the precision is low, around 0.18. This means that many patients predicted as high-risk are not actually readmitted within 30 days.

This trade-off is understandable for a first baseline on an imbalanced healthcare dataset. In this context, detecting high-risk patients is important, but the model still needs improvement to reduce false positives.

## Current Status

Current project stage:

* repository initialized
* project structure created
* dataset selected
* exploratory data analysis completed
* missing values analyzed
* binary target created
* first baseline model trained
* baseline model evaluated
* confusion matrix and ROC curve generated

## Next Steps

The next steps are:

1. Move the preprocessing and training logic from the notebook into reusable Python scripts.
2. Train and compare additional models such as Random Forest and Gradient Boosting.
3. Save the best model.
4. Build a FastAPI prediction endpoint.
5. Add automated tests with Pytest.
6. Add Docker support.
7. Add basic monitoring.
8. Write a final results report.

## Skills Demonstrated

This project demonstrates skills in:

* Python
* Pandas
* NumPy
* Scikit-learn
* Exploratory Data Analysis
* Data preprocessing
* Feature selection
* Classification
* Model evaluation
* Imbalanced classification
* Machine Learning Engineering
* Git and GitHub
* Project documentation

## Future Improvements

Possible improvements include:

* using diagnosis codes with better feature engineering
* testing tree-based models
* tuning classification thresholds
* comparing recall and precision trade-offs
* adding model calibration
* adding MLflow experiment tracking
* deploying the model through a FastAPI application
* monitoring input data and prediction drift

  
- Project documentation

  Initial README
