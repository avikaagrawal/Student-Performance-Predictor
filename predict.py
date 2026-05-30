import joblib

model = joblib.load("model/student_model.pkl")

study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance %: "))
previous_score = float(input("Previous Score: "))

prediction = model.predict(
    [[study_hours, attendance, previous_score]]
)

print(f"Predicted Final Grade: {prediction[0]:.2f}")