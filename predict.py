import pandas as pd
import joblib


# Load trained model
model = joblib.load("placement_model.pkl")


# Create a new student's data
student = pd.DataFrame([{
    "branch": "CSE",
    "college_tier": "Tier 2",
    "cgpa": 8.8,
    "backlogs": 0,
    "coding_skills": 8.5,
    "dsa_score": 8.0,
    "aptitude_score": 8.0,
    "communication_skills": 8.0,
    "ml_knowledge": 7.0,
    "system_design": 6.0,
    "internships": 2,
    "projects_count": 4,
    "certifications": 3,
    "hackathons": 2,
    "open_source_contributions": 1,
    "extracurriculars": 3
}])


# Predict placement
prediction = model.predict(student)

# Get probability
probability = model.predict_proba(student)[0][1]


# Display result
print("\n==============================")
print("AI PLACEMENT PREDICTION")
print("==============================")

if prediction[0] == 1:
    print("Placement Prediction: PLACED")
else:
    print("Placement Prediction: NOT PLACED")

print(f"Placement Probability: {probability * 100:.2f}%")