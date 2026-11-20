
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# Load dataset
data = load_breast_cancer()

# Create DataFrame
X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Add target column
X["target"] = data.target

# Create data folder path
project_path = Path(__file__).resolve().parents[1]
data_path = project_path / "data"

# Save dataset as CSV
X.to_csv(data_path / "breast_cancer.csv", index=False)

# Display dataset information
print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)

print("\nFirst 5 rows:")
print(X.head())

print("\nTarget classes:")
print(data.target_names)

print("\nTarget distribution:")
print(X["target"].value_counts())

print("\nMissing values:")
print(X.isnull().sum().sum())
