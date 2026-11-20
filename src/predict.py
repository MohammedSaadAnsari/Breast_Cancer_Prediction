
import pandas as pd
import joblib

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load trained model
model_path = PROJECT_DIR / "models" / "best_cancer_model.pkl"

model = joblib.load(model_path)

print("Model loaded successfully!")

# Load dataset to get the correct feature names
df = pd.read_csv(
    PROJECT_DIR / "data" / "breast_cancer.csv"
)

X = df.drop("target", axis=1)

# Select one sample for demonstration
sample = X.iloc[[0]]

print("\nInput data:")
print(sample)

# Make prediction
prediction = model.predict(sample)[0]

# Get prediction probability
probabilities = model.predict_proba(sample)[0]

if prediction == 0:
    result = "Malignant"
else:
    result = "Benign"

print("\nPrediction Result:", result)

print(
    "Malignant Probability:",
    round(probabilities[0] * 100, 2), "%"
)

print(
    "Benign Probability:",
    round(probabilities[1] * 100, 2), "%"
)

print("\nPrediction completed successfully!")
