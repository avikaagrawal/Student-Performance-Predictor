import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import joblib

# Load data
df = pd.read_csv("data/student_data.csv")

X = df[["StudyHours", "Attendance", "PreviousScore"]]
y = df["FinalGrade"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Linear Regression R2 Score:",
      r2_score(y_test, lr_pred))

print("Decision Tree R2 Score:",
      r2_score(y_test, dt_pred))

import matplotlib.pyplot as plt

plt.scatter(y_test, lr_pred)

plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted")

plt.show()
# Save best model
import os

os.makedirs("model", exist_ok=True)
joblib.dump(lr, "model/student_model.pkl")

print("Model Saved!")