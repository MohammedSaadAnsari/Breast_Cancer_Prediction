# Breast Cancer Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to classify breast cancer tumours as **Malignant** or **Benign** using features from the Breast Cancer Wisconsin Diagnostic Dataset.

The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), model comparison, hyperparameter tuning, model evaluation, and deployment using a FastAPI application.

**Disclaimer:** This is an educational machine learning project and is not intended for clinical diagnosis or medical decision-making.

## Problem Statement

Early identification of malignant tumours is an important medical challenge. This project explores how machine learning algorithms can classify tumour characteristics using a labelled dataset.

## Dataset

- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** Scikit-learn built-in dataset
- **Total Records:** 569
- **Input Features:** 30
- **Target Classes:** Malignant and Benign
- **Missing Values:** 0

### Target Labels

- `0` = Malignant
- `1` = Benign

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pytest

## Machine Learning Models

The following machine learning algorithms were trained and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest

SVM performed best in the initial model comparison. GridSearchCV was then used to optimize its hyperparameters using 5-fold cross-validation.

## Model Performance

### Best Model: Support Vector Machine (SVM)

**Best Hyperparameters:**

| Parameter | Value |
|---|---|
| C | 1 |
| Kernel | RBF |
| Gamma | Scale |
| Cross-validation | 5-Fold |
| Optimization Metric | Malignant Recall |

### Hyperparameter Tuning Result

- **Best Cross-Validation Malignant Recall:** 95.88%

### Final Model Evaluation

The tuned SVM model was evaluated on the held-out test dataset.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 98.25% |
| Malignant Precision | 97.62% |
| Malignant Recall | 97.62% |
| Malignant F1-Score | 97.62% |
| Malignant ROC-AUC | 99.50% |

### Confusion Matrix

```text
[[41  1]
 [ 1 71]]
```

| Actual / Predicted | Malignant | Benign |
|---|---:|---:|
| Malignant | 41 | 1 |
| Benign | 1 | 71 |

The model correctly classified 112 out of 114 test samples.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.98 | 0.98 | 0.98 | 42 |
| Benign | 0.99 | 0.99 | 0.99 | 72 |

**Note:** These results represent performance on one held-out test split and do not establish clinical reliability.

## Project Structure

```text
Breast_Cancer_Prediction/
│
├── data/
│   └── breast_cancer.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── preprocessing.ipynb
│
├── models/
│   ├── cancer_model.pkl
│   └── best_cancer_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── compare_models.py
│   ├── tune_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── app/
│   └── main.py
│
├── tests/
│   └── test_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

Clone the repository:

```powershell
git clone https://github.com/MohammedSaadAnsari/Breast_Cancer_Prediction.git
```

Navigate to the project directory:

```powershell
cd Breast_Cancer_Prediction
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

### 1. Generate the Dataset

```powershell
python src/preprocessing.py
```

### 2. Train the Initial Model

```powershell
python src/train.py
```

### 3. Compare Machine Learning Models

```powershell
python src/compare_models.py
```

### 4. Tune the SVM Model

```powershell
python src/tune_model.py
```

### 5. Evaluate the Final Model

```powershell
python src/evaluate_model.py
```

### 6. Run a Sample Prediction

```powershell
python src/predict.py
```

## Run the FastAPI Application

Start the FastAPI server:

```powershell
python -m uvicorn app.main:app --reload
```

**API Base URL:**

http://127.0.0.1:8000

**Interactive API Documentation:**

http://127.0.0.1:8000/docs

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check API status |
| POST | `/predict` | Predict tumour classification |

The prediction endpoint accepts exactly 30 numerical feature values in the original dataset feature order.

## Run Tests

Execute the following command:

```powershell
python -m pytest tests/ -v
```

**Test Results:** 3 tests passed.

- Home endpoint test
- Prediction endpoint test
- Invalid prediction input test

## Project Learning Outcomes

- Understanding an end-to-end machine learning workflow.
- Performing data preprocessing and exploratory data analysis.
- Training and comparing classification algorithms.
- Applying GridSearchCV for hyperparameter tuning.
- Evaluating classification models using performance metrics.
- Saving and loading trained models using Joblib.
- Building REST APIs using FastAPI.
- Testing APIs using Pytest.

## Future Improvements

- Develop a user-friendly frontend.
- Add model monitoring and logging.
- Improve validation and deployment.
- Explore additional classification algorithms.
- Evaluate model performance on suitable external validation datasets.

## Author

**Mohammed Saad Ansari**

- GitHub: [MohammedSaadAnsari](https://github.com/MohammedSaadAnsari)
- Project Repository: [Breast Cancer Prediction](https://github.com/MohammedSaadAnsari/Breast_Cancer_Prediction)

---

**This project was developed for educational and machine learning portfolio purposes only. It is not a clinically validated medical diagnostic system.**