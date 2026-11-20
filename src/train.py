
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(PROJECT_DIR / "data" / "breast_cancer.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

X = df.drop("target", axis=1)

y = df["target"]

print("Features:", X.shape)
print("Target:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual values:", y_test.values[:10])

print("Predicted values:", y_pred[:10])

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "cancer_model.pkl")

print("Model saved successfully!")