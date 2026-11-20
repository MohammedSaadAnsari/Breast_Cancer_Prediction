
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load trained model
MODEL_PATH = PROJECT_DIR / "models" / "best_cancer_model.pkl"
model = joblib.load(MODEL_PATH)

# Load feature names in the original order
df = pd.read_csv(
    PROJECT_DIR / "data" / "breast_cancer.csv"
)

FEATURE_NAMES = df.drop("target", axis=1).columns.tolist()

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Educational ML prediction API",
    version="1.0.0"
)


# Input validation
class PredictionInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 breast cancer features in dataset order"
    )


@app.get("/")
def home():
    return {
        "message": "Breast Cancer Prediction API is running!"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    if len(data.features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail="Exactly 30 feature values are required."
        )

    input_data = pd.DataFrame(
        [data.features],
        columns=FEATURE_NAMES
    )

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]

    result = "Malignant" if prediction == 0 else "Benign"

    return {
        "prediction": prediction,
        "result": result,
        "malignant_probability": round(
            float(probabilities[0]), 4
        ),
        "benign_probability": round(
            float(probabilities[1]), 4
        )
    }
