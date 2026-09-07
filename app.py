from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML models
placement_model = joblib.load("placement_model.pkl")
salary_model = joblib.load("salary_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        try:
            # Get data from form
            student = pd.DataFrame([{
                "branch": request.form["branch"],
                "college_tier": request.form["college_tier"],
                "cgpa": float(request.form["cgpa"]),
                "backlogs": int(request.form["backlogs"]),
                "coding_skills": float(request.form["coding_skills"]),
                "dsa_score": float(request.form["dsa_score"]),
                "aptitude_score": float(request.form["aptitude_score"]),
                "communication_skills": float(
                    request.form["communication_skills"]
                ),
                "ml_knowledge": float(request.form["ml_knowledge"]),
                "system_design": float(request.form["system_design"]),
                "internships": int(request.form["internships"]),
                "projects_count": int(request.form["projects_count"]),
                "certifications": int(request.form["certifications"]),
                "hackathons": int(request.form["hackathons"]),
                "open_source_contributions": int(
                    request.form["open_source_contributions"]
                ),
                "extracurriculars": int(
                    request.form["extracurriculars"]
                )
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

            ml = student["ml_knowledge"][0]
            coding = student["coding_skills"][0]
            system = student["system_design"][0]

            if ml >= 7:
                career = "Machine Learning Engineer"

            elif coding >= 8:
                career = "Software Engineer"

            elif system >= 7:
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
                improvements.append("Improve Communication")

            if system < 7:
                improvements.append("Learn System Design")

            if ml < 7:
                improvements.append("Improve Machine Learning")

            if student["projects_count"][0] < 3:
                improvements.append("Build More Projects")

            if not improvements:
                improvements.append(
                    "Your current skill profile is strong"
                )

            result = {
                "placed": placement_prediction == 1,
                "probability": round(placement_probability, 2),
                "salary": round(predicted_salary, 2),
                "career": career,
                "improvements": improvements
            }

        except Exception as e:
            result = {
                "error": str(e)
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)