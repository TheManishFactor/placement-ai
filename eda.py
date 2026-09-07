import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/student_placement_synthetic.csv")

# 1. Placement distribution
print("PLACEMENT STATUS")
print(df["placement_status"].value_counts())

# 2. Average CGPA for placed vs not placed
print("\nAVERAGE CGPA")
print(df.groupby("placement_status")["cgpa"].mean())

# 3. Average coding skills
print("\nAVERAGE CODING SKILLS")
print(df.groupby("placement_status")["coding_skills"].mean())

# 4. Average number of internships
print("\nAVERAGE INTERNSHIPS")
print(df.groupby("placement_status")["internships"].mean())

# 5. Average number of projects
print("\nAVERAGE PROJECTS")
print(df.groupby("placement_status")["projects_count"].mean())

# 6. Salary statistics - only placed students
placed_students = df[df["placement_status"] == 1]

print("\nSALARY STATISTICS")
print(placed_students["salary_package_lpa"].describe())

# 7. Plot placement distribution
df["placement_status"].value_counts().plot(kind="bar")

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.xticks([0, 1], ["Not Placed", "Placed"], rotation=0)

plt.show()