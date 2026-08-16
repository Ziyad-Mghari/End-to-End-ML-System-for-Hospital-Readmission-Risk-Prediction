# Results Report: Hospital Readmission Risk Prediction

## 1. Project Overview

This project aims to build an end-to-end machine learning system for predicting hospital readmission risk using clinical tabular data.

The main objective is to predict whether a diabetic patient is likely to be readmitted to the hospital within 30 days.

This first version of the project covers the full workflow from exploratory data analysis to model serving:

* exploratory data analysis
* missing values analysis
* binary target creation
* feature selection
* preprocessing pipeline
* baseline model training
* model comparison
* threshold tuning
* FastAPI prediction endpoint
* Docker containerization
* automated tests
* basic prediction monitoring

The project is designed as a production-oriented machine learning project rather than a simple notebook experiment.

---

## 2. Problem Statement

Hospital readmissions are an important healthcare issue. A patient being readmitted shortly after discharge can indicate complications, insufficient follow-up, or worsening health condition.

In this project, early hospital readmission is treated as a binary classification problem.

The goal is to predict whether a patient will be readmitted within 30 days.

The binary target is defined as:

* `0`: the patient was not readmitted within 30 days
* `1`: the patient was readmitted within 30 days

The main challenge is that the positive class is rare. Only around 11% of patients are readmitted within 30 days. This makes the problem an imbalanced classification task.

Because of this imbalance, accuracy alone is not sufficient to evaluate the model. Metrics such as recall, precision, F1-score, ROC-AUC, and the confusion matrix are more informative.

In a healthcare context, recall is especially important because false negatives correspond to high-risk patients that the model fails to detect.

---

## 3. Dataset

The dataset used in this project is:

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

The dataset is not stored directly in the repository. The expected local files are:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

---

## 4. Exploratory Data Analysis

The exploratory analysis was performed in:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

The analysis focused on understanding the structure of the dataset before building the machine learning pipeline.

The main steps were:

* loading the dataset
* inspecting the number of rows and columns
* analyzing the target variable
* identifying missing values
* creating the binary target
* analyzing class imbalance
* comparing numerical features by target class
* analyzing categorical feature cardinality
* selecting features for the first baseline model

The dataset contains:

```text
101766 rows
```

After feature selection, the modeling dataset contains:

```text
38 input features
```

---

## 5. Missing Values Analysis

In this dataset, some missing values are encoded as `"?"`.

The main columns containing missing values were:

| Column              | Number of missing values |
| ------------------- | -----------------------: |
| `weight`            |                    98569 |
| `medical_specialty` |                    49949 |
| `payer_code`        |                    40256 |
| `race`              |                     2273 |
| `diag_3`            |                     1423 |
| `diag_2`            |                      358 |
| `diag_1`            |                       21 |

The following columns were removed in the first version of the project:

```text
weight
payer_code
medical_specialty
race
```

The reasons were:

* `weight` contains too many missing values
* `payer_code` is not central for a first medical baseline
* `medical_specialty` contains many missing values
* `race` is a sensitive attribute and was removed to reduce the risk of unfair bias

The diagnosis columns `diag_1`, `diag_2`, and `diag_3` contained fewer missing values. Their missing values were replaced with `"Unknown"` during the cleaning step.

---

## 6. Target Distribution

The binary target is highly imbalanced.

| Class | Meaning                       | Proportion |
| ----- | ----------------------------- | ---------: |
| `0`   | Not readmitted within 30 days |     88.84% |
| `1`   | Readmitted within 30 days     |     11.16% |

This imbalance has an important consequence: a naive model that always predicts class `0` would already obtain a high accuracy of around 89%.

Therefore, accuracy alone is misleading.

The model evaluation focuses on:

* precision
* recall
* F1-score
* ROC-AUC
* confusion matrix

Recall is particularly important in this project because it measures the proportion of truly high-risk patients that the model successfully detects.

---

## 7. Numerical Feature Analysis

Several numerical variables were compared between low-risk and high-risk patients.

Important numerical features included:

```text
time_in_hospital
num_lab_procedures
num_procedures
num_medications
number_outpatient
number_emergency
number_inpatient
number_diagnoses
```

One of the clearest differences was observed for `number_inpatient`.

| Target | Average `number_inpatient` |
| ------ | -------------------------: |
| `0`    |                       0.56 |
| `1`    |                       1.22 |

Patients readmitted within 30 days had, on average, more than twice as many previous inpatient admissions.

Another important variable was `number_emergency`.

| Target | Average `number_emergency` |
| ------ | -------------------------: |
| `0`    |                       0.18 |
| `1`    |                       0.36 |

Patients readmitted within 30 days had more previous emergency visits on average.

These observations suggest that previous healthcare utilization is an important signal for predicting early readmission risk.

---

## 8. Categorical Feature Analysis

Categorical variables were analyzed by counting the number of unique values in each column.

Some variables had a small number of categories and were easier to use directly:

```text
age
gender
diabetesMed
change
A1Cresult
max_glu_serum
```

Other variables had a large number of categories:

| Column   | Number of unique values |
| -------- | ----------------------: |
| `diag_1` |                     717 |
| `diag_2` |                     749 |
| `diag_3` |                     790 |

The diagnosis columns contain many distinct medical codes. For the first baseline model, they were removed to keep the preprocessing simple and interpretable.

A future version of the project could use these diagnosis codes with better feature engineering, for example by grouping codes into broader medical categories.

---

## 9. Feature Selection

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

The reasons were:

* `encounter_id` and `patient_nbr` are identifiers
* `readmitted` would cause data leakage because it was used to create the binary target
* `diag_1`, `diag_2`, and `diag_3` contain many distinct categories
* `examide` and `citoglipton` contain only one unique value and do not provide useful information

After this feature selection step, the modeling dataset contained:

```text
101766 rows
38 features
```

---

## 10. Preprocessing Pipeline

A preprocessing pipeline was created using scikit-learn.

The pipeline applies different transformations depending on the feature type.

Numerical features are standardized with:

```text
StandardScaler
```

Categorical features are encoded with:

```text
OneHotEncoder
```

The columns:

```text
admission_type_id
discharge_disposition_id
admission_source_id
```

were treated as categorical variables because they are numerical codes, not continuous numerical quantities.

The preprocessing was combined with the model inside a scikit-learn `Pipeline`.

This approach has several advantages:

* it keeps preprocessing and modeling together
* it avoids inconsistent transformations between training and testing
* it makes the model easier to save and reuse
* it prepares the project for API deployment

---

## 11. Train/Test Split

The data was split into training and test sets.

The test set represents 20% of the dataset.

The split was stratified using the target variable. This means that the proportion of high-risk patients was preserved in both the training and test sets.

This is important because the target is imbalanced.

The split was performed using:

```text
train_test_split
test_size = 0.2
random_state = 42
stratify = y
```

For the first version of the project, models were trained on a sample of 20,000 training rows to keep training time reasonable.

---

## 12. Baseline Model

The first model trained was a Logistic Regression model.

Logistic Regression was selected as the baseline because it is:

* simple
* fast to train
* interpretable
* useful as a first reference model

Because the target variable is imbalanced, the model was trained with class balancing:

```text
class_weight = "balanced"
```

The baseline model was implemented inside a full scikit-learn pipeline combining preprocessing and classification.

---

## 13. Baseline Results

The Logistic Regression baseline was evaluated on a separate test set.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.665 |     0.181 |  0.568 |    0.275 |   0.668 |

### Interpretation

The baseline model achieved a ROC-AUC of approximately 0.67.

This indicates a moderate ability to distinguish between patients with low and high readmission risk.

The recall for the high-risk class is approximately 0.57. This means that the model detects more than half of the patients who are readmitted within 30 days.

However, the precision is low, around 0.18. This means that many patients predicted as high-risk are not actually readmitted within 30 days.

This trade-off is understandable for a first baseline on an imbalanced healthcare dataset.

In this healthcare-oriented setting, detecting high-risk patients is important, but the model still produces many false positives.

---

## 14. Model Comparison

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

### Interpretation

Logistic Regression and Random Forest produced very similar results.

Random Forest achieved a slightly higher accuracy and ROC-AUC, but its recall was slightly lower than Logistic Regression.

Gradient Boosting achieved a high accuracy, but its recall for the high-risk class was extremely low. This means that the model predicted almost all patients as low risk and failed to detect most patients readmitted within 30 days.

This result shows why accuracy is not sufficient in an imbalanced healthcare problem.

Even though Gradient Boosting achieved the highest accuracy, it was not the most useful model for the project objective because it failed to detect high-risk patients.

At this stage, Logistic Regression remains a strong baseline because it provides the best recall among the compared models.

---

## 15. Threshold Tuning

By default, a classification model predicts class `1` when the predicted probability is greater than or equal to 0.50.

However, in a healthcare context, the default threshold may not be optimal.

The classification threshold controls the trade-off between precision and recall.

A lower threshold usually increases recall because more patients are classified as high-risk. However, it also increases the number of false positives.

Several thresholds were tested for the Logistic Regression baseline.

| Threshold | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives |
| --------: | -------: | --------: | -----: | -------: | --------------: | --------------: | -------------: |
|      0.45 |    0.548 |     0.158 |  0.706 |    0.259 |            8523 |             668 |           1603 |
|      0.50 |    0.665 |     0.181 |  0.568 |    0.275 |            5840 |             980 |           1291 |
|      0.55 |    0.748 |     0.203 |  0.430 |    0.276 |            3840 |            1294 |            977 |

### Interpretation

The threshold of 0.55 achieved the best F1-score, but it reduced recall to 0.430.

The threshold of 0.45 is more interesting for a healthcare-oriented use case because it increases recall to 0.706. This means that the model detects around 70% of patients who are readmitted within 30 days.

Compared with the default threshold of 0.50, the threshold of 0.45 detects more high-risk patients:

| Threshold | True Positives | False Negatives |
| --------: | -------------: | --------------: |
|      0.50 |           1291 |             980 |
|      0.45 |           1603 |             668 |

Lowering the threshold from 0.50 to 0.45 detects 312 additional high-risk patients.

However, this improvement comes with more false positives:

| Threshold | False Positives |
| --------: | --------------: |
|      0.50 |            5840 |
|      0.45 |            8523 |

This illustrates the trade-off between recall and precision.

In a medical screening context, increasing recall can be useful because missing high-risk patients may be costly. However, the number of false alerts must also be considered.

---

## 16. Saved Model and Prediction Script

The baseline Logistic Regression pipeline was saved using `joblib`.

The saved model is stored locally in:

```text
models/baseline_logistic_regression.joblib
```

A model metadata file was also saved:

```text
models/model_metadata.joblib
```

The metadata file contains useful information for the API, such as:

* expected input columns
* sample patient data

A prediction script was implemented in:

```text
src/predict.py
```

This script loads the trained model and generates predictions on a small sample of patients.

This confirms that the trained model can be reused without retraining.

---

## 17. FastAPI Application

A FastAPI application was created to expose the trained model through an API.

The API is implemented in:

```text
api/main.py
```

The API provides the following endpoints:

| Endpoint              | Method | Description                                   |
| --------------------- | ------ | --------------------------------------------- |
| `/`                   | GET    | Checks that the API is running                |
| `/health`             | GET    | Checks whether the model is loaded            |
| `/sample-patient`     | GET    | Returns one example patient from the dataset  |
| `/predict`            | POST   | Predicts the readmission risk for one patient |
| `/monitoring/summary` | GET    | Returns basic monitoring statistics           |

The prediction endpoint receives patient features and a classification threshold.

It returns:

* risk score
* binary prediction
* readable risk label
* threshold used

Example response:

```json
{
  "risk_score": 0.37,
  "prediction": 0,
  "risk_label": "low_risk",
  "threshold": 0.45
}
```

The API was tested locally using Swagger UI at:

```text
http://127.0.0.1:8000/docs
```

---

## 18. Docker Containerization

Docker support was added to make the API easier to run in a reproducible environment.

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
* the Python dependencies

The image can be built with:

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

---

## 19. Automated Tests

Automated tests were added using Pytest.

The tests are located in:

```text
tests/test_preprocessing.py
tests/test_api.py
```

The preprocessing tests check:

* that `X` and `y` have compatible shapes
* that the target contains only `0` and `1`
* that numerical and categorical feature lists are not empty
* that feature lists do not contain duplicates

The API tests check:

* the health endpoint
* the sample patient endpoint
* the prediction endpoint
* the structure of the prediction response
* the validity of the risk score and prediction label

The tests can be launched with:

```bash
pytest
```

The test suite was successfully executed locally.

---

## 20. Basic Monitoring

A basic monitoring module was added to track predictions made by the API.

The monitoring logic is implemented in:

```text
src/monitoring.py
```

Each time the `/predict` endpoint is called, the API logs useful prediction information into a local CSV file:

```text
monitoring/predictions_log.csv
```

The logged information includes:

* timestamp
* risk score
* predicted class
* risk label
* threshold used
* selected patient features

The API also includes a monitoring endpoint:

```text
GET /monitoring/summary
```

This endpoint returns basic statistics about logged predictions, such as:

* number of predictions
* average risk score
* high-risk prediction rate
* low-risk prediction rate
* average threshold used

Example response:

```json
{
  "number_of_predictions": 3,
  "average_risk_score": 0.42,
  "high_risk_rate": 0.33,
  "low_risk_rate": 0.67,
  "average_threshold": 0.45
}
```

The prediction log file is ignored by Git because it is generated locally during API usage.

---

## 21. Project Structure

The current project structure is:

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
│   └── .gitkeep
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
├── requirements.txt
└── README.md
```

---

## 22. Main Limitations

This first version of the project is intentionally simple and focused on building a clean end-to-end workflow.

The main limitations are listed below.

### 22.1 Limited Feature Engineering

Some potentially useful variables were removed from the first baseline model.

In particular, the diagnosis columns were removed because they contain many distinct categories.

These columns may contain important medical information and could improve model performance if processed properly.

### 22.2 Simple Model Selection

Only three classical models were compared:

* Logistic Regression
* Random Forest
* Gradient Boosting

More models could be tested in the future, including:

* calibrated classifiers
* XGBoost
* LightGBM
* CatBoost
* regularized logistic regression variants

### 22.3 Limited Hyperparameter Tuning

The models were trained with simple initial hyperparameters.

A more complete project could include systematic hyperparameter tuning using:

* grid search
* randomized search
* cross-validation

### 22.4 Training on a Sample

For this first version, the models were trained on a sample of 20,000 training rows to keep execution time reasonable.

Training on the full dataset could slightly change the results and should be tested in a more complete version.

### 22.5 No External Validation

The model was evaluated on a test split from the same dataset.

It was not validated on data from another hospital system or another time period.

External validation would be necessary before considering any real-world use.

### 22.6 Basic Monitoring Only

The monitoring implemented in this project is intentionally simple.

It logs predictions and computes basic summary statistics.

A production system would require more advanced monitoring, including:

* input data drift
* prediction drift
* performance monitoring over time
* alerting
* model retraining strategy

### 22.7 Not Intended for Clinical Use

This project is an educational machine learning engineering project.

The model is not clinically validated and should not be used to make real medical decisions.

---

## 23. Future Improvements

Several improvements could be made in future versions of the project.

### 23.1 Improve Feature Engineering

Possible improvements include:

* grouping diagnosis codes into broader medical categories
* creating features based on previous healthcare utilization
* creating medication-related features
* using domain knowledge to improve interpretability

### 23.2 Train on the Full Dataset

The current version trains models on a 20,000-row sample for speed.

A future version could train and evaluate models on the full training set.

### 23.3 Add Hyperparameter Tuning

The model comparison could be improved using:

* cross-validation
* randomized search
* grid search
* threshold optimization based on business or clinical constraints

### 23.4 Improve Model Calibration

Because the API returns a risk score, calibration is important.

A future version could evaluate whether predicted probabilities are well calibrated using:

* calibration curves
* Brier score
* calibrated classifiers

### 23.5 Add Advanced Monitoring

Monitoring could be improved by tracking:

* input feature distributions
* prediction distribution over time
* drift between training data and incoming API data
* high-risk rate over time

### 23.6 Add CI/CD

A future version could add GitHub Actions to automatically run tests when code is pushed.

This would improve reliability and make the project closer to a real production workflow.

### 23.7 Deploy the API Online

The API currently runs locally.

A future version could deploy the API using a cloud platform or container service.

---

## 24. Conclusion

This project demonstrates a complete first version of an end-to-end machine learning system for hospital readmission risk prediction.

The project started with exploratory data analysis and progressed toward a more production-oriented structure.

The current version includes:

* data cleaning
* target creation
* feature selection
* preprocessing pipeline
* baseline model training
* model comparison
* threshold tuning
* model saving
* prediction script
* FastAPI application
* Docker support
* automated tests
* basic monitoring

The Logistic Regression baseline achieved a ROC-AUC of approximately 0.67 and a recall of approximately 0.57 at the default threshold.

Threshold tuning showed that lowering the classification threshold to 0.45 increases recall to approximately 0.71, allowing the model to detect more high-risk patients at the cost of more false positives.

Although the model performance is still limited, the project successfully demonstrates the main components of a machine learning engineering workflow:

* reproducible preprocessing
* model training
* evaluation
* API serving
* testing
* containerization
* monitoring

This makes the project a strong foundation for further improvements and a useful portfolio project for demonstrating machine learning engineering skills.
