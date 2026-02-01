import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Health Status Predictor", layout="wide")

# ------------------------------------
# Load & Prepare Data
# ------------------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    df = pd.read_csv("health_tracker_500.csv")

    # -------- Ordinal Encoding --------
    activity_map = {
        'Sedentary': 0,
        'Lightly Active': 1,
        'Moderately Active': 2,
        'Very Active': 3
    }

    diet_map = {
        'Balanced': 2,
        'High Protein': 2,
        'Vegetarian': 2,
        'High Carb': 1,
        'Keto': 1
    }

    df['ActivityLevel'] = df['ActivityLevel'].map(activity_map)
    df['DietType'] = df['DietType'].map(diet_map)

    # -------- Health Scoring Logic --------
    def compute_health_status(row):
        score = 0

        # BMI logic (Overweight allowed if active)
        if row['BMI_Category'] == 'Normal':
            score += 1
        elif row['BMI_Category'] == 'Overweight' and row['ActivityLevel'] >= 2:
            score += 1

        # Blood Pressure
        if row['BP_Category'] in ['Normal', 'Elevated']:
            score += 1

        # Cholesterol
        if row['Cholesterol_Category'] == 'Desirable':
            score += 1

        # Activity (fitness signal)
        if row['ActivityLevel'] >= 2:
            score += 1

        return 'Healthy and Fit' if score >= 3 else 'Unhealthy and Not Fit'

    df['HealthyStatus'] = df.apply(compute_health_status, axis=1)
    return df

# ------------------------------------
# Train Model
# ------------------------------------
@st.cache_resource(show_spinner=True)
def train_model(df):
    features = [
        'Height_cm', 'Weight_kg', 'BMI',
        'BP_Systolic', 'BP_Diastolic',
        'Cholesterol_mg_dL',
        'ActivityLevel', 'DietType'
    ]

    X = df[features]
    y = df['HealthyStatus'].map({
        'Unhealthy and Not Fit': 0,
        'Healthy and Fit': 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model, features

# ------------------------------------
# BMI Gauge
# ------------------------------------
def bmi_gauge(bmi):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={'text': "BMI"},
        gauge={
            'axis': {'range': [10, 40]},
            'steps': [
                {'range': [10, 18.5], 'color': "lightblue"},
                {'range': [18.5, 24.9], 'color': "green"},
                {'range': [25, 29.9], 'color': "orange"},
                {'range': [30, 40], 'color': "red"}
            ],
            'bar': {'color': "darkblue"}
        }
    ))
    return fig

# ------------------------------------
# Health Tips
# ------------------------------------
def generate_tips(bmi, bp_sys, bp_dia, chol, activity):
    tips = []

    if bmi >= 30:
        tips.append("Focus on gradual weight loss through diet and exercise.")
    elif bmi >= 25:
        tips.append("Maintain regular physical activity to stay metabolically healthy.")

    if bp_sys >= 130 or bp_dia >= 85:
        tips.append("Reduce sodium intake and monitor blood pressure regularly.")

    if chol >= 200:
        tips.append("Increase fiber intake and reduce saturated fats.")

    if activity < 2:
        tips.append("Aim for at least 150 minutes of moderate exercise weekly.")

    if not tips:
        tips.append("Great job! Keep maintaining your healthy lifestyle.")

    return tips

# ------------------------------------
# Streamlit UI
# ------------------------------------
def main():
    st.title("Health Status Predictor")
    st.markdown("Predicts **Healthy & Fit** vs **Unhealthy & Not Fit** using smarter health logic.")

    df = load_and_prepare_data()
    model, features = train_model(df)

    st.header("Single Prediction")

    height = st.number_input("Height (cm)", 100, 220, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    bp_sys = st.number_input("BP Systolic", 80, 200, 120)
    bp_dia = st.number_input("BP Diastolic", 50, 130, 80)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 180)

    activity_label = st.selectbox(
        "Activity Level",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
    )

    diet_label = st.selectbox(
        "Diet Type",
        ["Balanced", "High Protein", "Vegetarian", "High Carb", "Keto"]
    )

    # Encoding user input
    activity_map = {'Sedentary': 0, 'Lightly Active': 1, 'Moderately Active': 2, 'Very Active': 3}
    diet_map = {'Balanced': 2, 'High Protein': 2, 'Vegetarian': 2, 'High Carb': 1, 'Keto': 1}

    bmi = round(weight / ((height / 100) ** 2), 1)

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            'Height_cm': height,
            'Weight_kg': weight,
            'BMI': bmi,
            'BP_Systolic': bp_sys,
            'BP_Diastolic': bp_dia,
            'Cholesterol_mg_dL': cholesterol,
            'ActivityLevel': activity_map[activity_label],
            'DietType': diet_map[diet_label]
        }])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][pred]

        status = "Healthy and Fit" if pred == 1 else "Unhealthy and Not Fit"
        color = "green" if pred == 1 else "red"

        st.markdown(
            f"""
            <div style="background-color:{color};padding:15px;border-radius:10px">
                <h3 style="color:white;">Prediction: {status}</h3>
                <p style="color:white;">Confidence: {proba:.2%}</p>
                <p style="color:white;">BMI: {bmi}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Health Tips")
        for tip in generate_tips(bmi, bp_sys, bp_dia, cholesterol, activity_map[activity_label]):
            st.markdown(f"- {tip}")

        st.subheader("BMI Visualization")
        st.plotly_chart(bmi_gauge(bmi), use_container_width=True)

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & scikit-learn — By Daniel Joe Gasper")

if __name__ == "__main__":
    main()

