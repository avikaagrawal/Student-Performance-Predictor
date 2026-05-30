import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load model
model = joblib.load("model/student_model.pkl")

st.title("🎓 Student Performance Predictor")
st.markdown("Predict a student's final grade using Machine Learning.")

st.divider()

study_hours = st.slider("📚 Study Hours", 0, 15, 5)
attendance = st.slider("📝 Attendance (%)", 0, 100, 80)
previous_score = st.slider("📊 Previous Score", 0, 100, 70)

if st.button("🚀 Predict Performance"):

    prediction = model.predict(
        [[study_hours, attendance, previous_score]]
    )[0]

    st.metric(
        label="Predicted Final Grade",
        value=f"{prediction:.2f}"
    )

    st.progress(min(int(prediction), 100))

    if prediction >= 90:
        st.success("🌟 Excellent Performance")
    elif prediction >= 75:
        st.info("👍 Good Performance")
    elif prediction >= 60:
        st.warning("📖 Needs Improvement")
    else:
        st.error("⚠️ At Risk")