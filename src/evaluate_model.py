
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_DIR / "data" / "breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split dataset (same split used during tuning)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load tuned model
model = joblib.load(
    PROJECT_DIR / "models" / "best_cancer_model.pkl"
)

# Predictions
y_pred = model.predict(X_test)

# Probability for malignant class (label 0)
malignant_probability = model.predict_proba(X_test)[:, 0]

# Evaluation
print("FINAL MODEL EVALUATION")
print("----------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))

print(
    "Malignant Precision:",
    precision_score(y_test, y_pred, pos_label=0)
)

print(
    "Malignant Recall:",
    recall_score(y_test, y_pred, pos_label=0)
)

print(
    "Malignant F1 Score:",
    f1_score(y_test, y_pred, pos_label=0)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Malignant", "Benign"],
    zero_division=0
))

print(
    "\nMalignant ROC-AUC:",
    roc_auc_score(y_test == 0, malignant_probability)
)

print("\nEvaluation completed successfully!")
