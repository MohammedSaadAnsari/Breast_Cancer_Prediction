# Breast Cancer Prediction using Machine Learning

## Project Overview

This project uses Machine Learning to classify breast cancer tumours as **Malignant** or **Benign** using features from the Breast Cancer Wisconsin dataset.

The project includes data preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, model evaluation, and a FastAPI application for predictions.

**Disclaimer:** This is an educational machine learning project and is not intended for clinical diagnosis or medical decision-making.

## Problem Statement

Early identification of malignant tumours is an important medical challenge. This project explores how machine learning can classify tumour characteristics using a labelled dataset.

## Dataset

- Dataset: Breast Cancer Wisconsin Diagnostic Dataset
- Source: Scikit-learn built-in dataset
- Total records: 569
- Input features: 30
- Target classes: Malignant and Benign
- Missing values: 0

Target labels:
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

The following algorithms were compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest

SVM was the leading model in the initial comparison. GridSearchCV was then used to tune its hyperparameters.

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

Clone the repository or open the project folder.

Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

Generate the dataset:

```powershell
python src/preprocessing.py
```

Train the initial model:

```powershell
python src/train.py
```

Compare machine learning models:

```powershell
python src/compare_models.py
```

Tune the SVM model:

```powershell
python src/tune_model.py
```

Evaluate the tuned model:

```powershell
python src/evaluate_model.py
```

Run a sample prediction:

```powershell
python src/predict.py
```

## Run the FastAPI Application

Start the server:

```powershell
python -m uvicorn app.main:app --reload
```

API base URL:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check API status |
| POST | `/predict` | Predict tumour classification |

The prediction endpoint accepts exactly 30 numerical feature values in the original dataset feature order.

## Run Tests

Execute:

```powershell
python -m pytest tests/ -v
```

Current test result: **3 tests passed**.

## Project Learning Outcomes

- Understanding an end-to-end machine learning workflow.
- Performing data preprocessing and EDA.
- Training and comparing classification algorithms.
- Applying GridSearchCV for hyperparameter tuning.
- Evaluating classification models.
- Saving and loading trained models with Joblib.
- Building REST APIs with FastAPI.
- Testing APIs using Pytest.

## Future Improvements

- Add a user-friendly frontend.
- Add model monitoring and logging.
- Improve validation and deployment.
- Explore additional classification algorithms.
- Evaluate performance on suitable external validation data.

## Author

**Mohammed Saad Ansari**

GitHub: [MohammedSaadAnsari](https://github.com/MohammedSaadAnsari)

---

*This project was developed for educational and portfolio purposes.*