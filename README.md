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

## Current Status

The current version of the project includes:

* GitHub repository initialized
* project folder structure created
* README created
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
* confusion matrix analysis
* ROC-AUC analysis
* Git and GitHub
* project documentation

## Next Steps

The next step is to move the preprocessing and training logic from the notebook into reusable Python scripts inside the `src/` folder.
