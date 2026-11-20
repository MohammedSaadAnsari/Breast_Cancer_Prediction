
import pandas as pd

from pathlib import Path
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_DIR / "data" / "breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Four models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Cross-validation setup
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Focus on malignant cases (class 0)
malignant_recall = make_scorer(recall_score, pos_label=0)
malignant_f1 = make_scorer(f1_score, pos_label=0)

scoring = {
    "accuracy": "accuracy",
    "malignant_recall": malignant_recall,
    "malignant_f1": malignant_f1
}

results = []

# Compare each model
for name, classifier in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Malignant Recall": scores["test_malignant_recall"].mean(),
        "Malignant F1": scores["test_malignant_f1"].mean()
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nModel comparison completed!")
