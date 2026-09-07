import pandas as pd
import joblib

model = joblib.load("salary_model.pkl")

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

predicted_salary = model.predict(student)[0]

print("\n==============================")
print("AI SALARY PREDICTION")
print("==============================")

print(f"Predicted Salary: {predicted_salary:.2f} LPA")