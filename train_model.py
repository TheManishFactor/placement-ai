import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/student_placement_synthetic.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. INPUT FEATURES AND TARGET
# ==========================================

# We want to predict placement_status
# salary_package_lpa is removed because salary is
# known only after placement.

X = df.drop(
    columns=["placement_status", "salary_package_lpa"]
)

y = df["placement_status"]


# ==========================================
# 3. DEFINE COLUMN TYPES
# ==========================================

categorical_columns = [
    "branch",
    "college_tier"
]

numeric_columns = [
    "cgpa",
    "backlogs",
    "coding_skills",
    "dsa_score",
    "aptitude_score",
    "communication_skills",
    "ml_knowledge",
    "system_design",
    "internships",
    "projects_count",
    "certifications",
    "hackathons",
    "open_source_contributions",
    "extracurriculars"
]


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),

        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns
        )
    ]
)


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. CREATE ML MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 7. TRAIN AND EVALUATE MODELS
# ==========================================

results = {}

for name, model in models.items():

    print("\n--------------------------------")
    print("Training:", name)
    print("--------------------------------")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    # Store results
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

print("\n\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

for name, metrics in results.items():

    print("\n", name)

    print(
        "Accuracy :",
        round(metrics["accuracy"], 4)
    )

    print(
        "Precision:",
        round(metrics["precision"], 4)
    )

    print(
        "Recall   :",
        round(metrics["recall"], 4)
    )

    print(
        "F1 Score :",
        round(metrics["f1"], 4)
    )


# ==========================================
# 9. SELECT BEST MODEL
# ==========================================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

print("\n\n==========================================")
print("BEST MODEL")
print("==========================================")

print(best_model_name)


# ==========================================
# 10. TRAIN BEST MODEL
# ==========================================

best_model = models[best_model_name]

best_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", best_model)
    ]
)

best_pipeline.fit(
    X_train,
    y_train
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(
    best_pipeline,
    "placement_model.pkl"
)

print("\nModel saved successfully!")
print("File: placement_model.pkl")