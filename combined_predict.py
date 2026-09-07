import pandas as pd
import joblib

# Load models
placement_model = joblib.load("placement_model.pkl")
salary_model = joblib.load("salary_model.pkl")


# -----------------------------
# STUDENT INPUT
# -----------------------------

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


# -----------------------------
# PLACEMENT PREDICTION
# -----------------------------

placement_prediction = placement_model.predict(student)[0]

placement_probability = (
    placement_model.predict_proba(student)[0][1] * 100
)


# -----------------------------
# SALARY PREDICTION
# -----------------------------

if placement_prediction == 1:

    predicted_salary = salary_model.predict(student)[0]

else:

    predicted_salary = 0


# -----------------------------
# CAREER RECOMMENDATION
# -----------------------------

if student["ml_knowledge"][0] >= 7:
    career = "Machine Learning Engineer"

elif student["coding_skills"][0] >= 8:
    career = "Software Engineer"

elif student["system_design"][0] >= 7:
    career = "Backend Engineer"

else:
    career = "Software / Data Professional"


# -----------------------------
# SKILL IMPROVEMENT
# -----------------------------

improvements = []

if student["dsa_score"][0] < 7:
    improvements.append("Improve DSA")

if student["communication_skills"][0] < 7:
    improvements.append("Improve communication")

if student["system_design"][0] < 7:
    improvements.append("Learn system design")

if student["ml_knowledge"][0] < 7:
    improvements.append("Improve Machine Learning")

if student["projects_count"][0] < 3:
    improvements.append("Build more projects")


# -----------------------------
# FINAL REPORT
# -----------------------------

print("\n")
print("=" * 50)
print("        AI PLACEMENT INTELLIGENCE")
print("=" * 50)

print("\nPLACEMENT RESULT")
print("-" * 50)

if placement_prediction == 1:
    print("Prediction        : PLACED")
else:
    print("Prediction        : NOT PLACED")

print(f"Placement Probability : {placement_probability:.2f}%")

print("\nSALARY PREDICTION")
print("-" * 50)

if placement_prediction == 1:
    print(f"Estimated Package : {predicted_salary:.2f} LPA")
else:
    print("Salary prediction : Not applicable")


print("\nCAREER RECOMMENDATION")
print("-" * 50)

print(f"Recommended Career : {career}")


print("\nSKILL IMPROVEMENT")
print("-" * 50)

if improvements:

    for item in improvements:
        print(f"- {item}")

else:
    print("Your current skill profile is strong.")


print("\n" + "=" * 50)
print("Prediction generated successfully!")
print("=" * 50)