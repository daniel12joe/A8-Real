import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Comprehensive Health Status Predictor", layout="wide")

# --------- Step 1: Data preparation ---------
def recompute_score(row):
    bmi = row['Weight_kg'] / ((row['Height_cm'] / 100) ** 2)
    score = 100 - (bmi - 22)**2 - abs(row['BP_Systolic'] - 120)/2 - (row['Cholesterol_mg_dL'] - 180)/5
    if row['ActivityLevel'] == "Very Active":
        score += 5
    elif row['ActivityLevel'] == "Moderately Active":
        score += 2
    if row['DietType'] == "Balanced":
        score += 3
    return max(0, min(100, round(score, 1)))

def prepare_dataset():
    if not os.path.exists("health_tracker_500_clean.csv"):
        df = pd.read_csv("health_tracker_500.csv")
        df['HealthScore'] = df.apply(recompute_score, axis=1)
        df.to_csv("health_tracker_500_clean.csv", index=False)
        print("Clean dataset created: health_tracker_500_clean.csv")
    return pd.read_csv("health_tracker_500_clean.csv")

# --------- Step 2: Model training ---------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    df = prepare_dataset()
    healthy_bmi = ['Normal', 'Underweight']
    healthy_bp = ['Normal', 'Elevated']
    healthy_chol = ['Desirable', 'Borderline']

    df['HealthyStatus'] = np.where(
        (df['BMI_Category'].isin(healthy_bmi)) &
        (df['BP_Category'].isin(healthy_bp)) &
        (df['Cholesterol_Category'].isin(healthy_chol)) &
        (df['HealthScore'] >= 60),
        'Healthy and Fit',
        'Unhealthy and Not Fit'
    )
    return df

@st.cache_resource(show_spinner=True)
def train_model(df):
    features = [
        'Height_cm', 'Weight_kg', 'BMI', 'BP_Systolic', 'BP_Diastolic', 
        'Cholesterol_mg_dL', 'ActivityLevel', 'DietType', 'HealthScore'
    ]
    target = 'HealthyStatus'
    encoders = {}
    for col in ['ActivityLevel', 'DietType']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    target_le = LabelEncoder()
    df[target] = target_le.fit_transform(df[target])
    encoders['target'] = target_le
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, encoders, features, target_le

def preprocess_input(input_df, encoders, features):
    for col in encoders:
        if col == 'target':
            continue
        le = encoders[col]
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])
        else:
            input_df[col] = le.transform([le.classes_[0]])[0]
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[features]

# --------- HealthScore calculation ---------
def calculate_health_score(height, weight, bp_sys, cholesterol, activity, diet):
    bmi = weight / ((height / 100) ** 2)
    score = 100 - (bmi - 22)**2 - abs(bp_sys - 120)/2 - (cholesterol - 180)/5
    if activity == "Very Active":
        score += 5
    elif activity == "Moderately Active":
        score += 2
    if diet == "Balanced":
        score += 3
    return max(0, min(100, round(score, 1))), round(bmi, 1)

def health_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Health Score"},
        gauge = {
            'axis': {'range': [0,100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0,60], 'color': "red"},
                {'range': [60,80], 'color': "orange"},
                {'range': [80,100], 'color': "green"}
            ]
        }
    ))
    return fig

# --------- Streamlit UI ---------
def main():
    st.title("Comprehensive Health Status Predictor")
    st.markdown("""
        This app predicts if a person is **Healthy and Fit** or **Unhealthy and Not Fit**  
        using BMI, blood pressure, cholesterol, diet, activity, and an auto‑calculated health score.
    """)

    df = load_and_prepare_data()
    model, encoders, features, target_le = train_model(df)

    choice = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Batch Prediction"])

    if choice == "Single Prediction":
        st.header("Input health details:")
        height = st.number_input("Height (cm)", 100, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        bp_sys = st.number_input("BP Systolic", 80, 200, 120)
        bp_dia = st.number_input("BP Diastolic", 50, 130, 80)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 180)
        activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
        diet = st.selectbox("Diet Type", ["Balanced", "High Protein", "High Carb", "Keto", "Vegetarian"])

        if st.button("Predict"):
            health_score, bmi = calculate_health_score(height, weight, bp_sys, cholesterol, activity, diet)

            input_data = {
                'Height_cm': height,
                'Weight_kg': weight,
                'BMI': bmi,
                'BP_Systolic': bp_sys,
                'BP_Diastolic': bp_dia,
                'Cholesterol_mg_dL': cholesterol,
                'ActivityLevel': activity,
                'DietType': diet,
                'HealthScore': health_score
            }
            input_df = pd.DataFrame([input_data])
            input_encoded = preprocess_input(input_df, encoders, features)
            pred = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0]
            class_name = target_le.inverse_transform([pred])[0]
            prob = proba[pred]

            # Color coding
            if class_name == "Healthy and Fit":
                color = "green"
            elif health_score >= 60:
                color = "orange"
            else:
                color = "red"

            st.markdown(
                f"""
                <div style="background-color:{color};padding:15px;border-radius:10px">
                    <h3 style="color:white;">Prediction: {class_name}</h3>
                    <p style="color:white;">Confidence: {prob:.2%}</p>
                    <p style="color:white;">Your BMI: {bmi}</p>
                    <p style="color:white;">Your Health Score: {health_score}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.plotly_chart(health_score_gauge(health_score))

    else:
        st.header("Batch prediction by uploading CSV file")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(batch_df.head())

            input_encoded = preprocess_input(batch_df.copy(), encoders, features)
            preds = model.predict(input_encoded)
            probas = model.predict_proba(input_encoded)
            pred_labels = target_le.inverse_transform(preds)
            pred_probas = [probas[i][preds[i]] for i in range(len(preds))]
            batch_df['Prediction'] = pred_labels
            batch_df['Confidence'] = pred_probas

            st.subheader("Prediction Results:")
            st.dataframe(batch_df)

            csv = batch_df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", data=csv, file_name='health_predictions.csv')

            st.subheader("Prediction Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Prediction', data=batch_df, order=target_le.classes_, ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, scikit-learn, and Prophet - By: Daniel Joe Gasper ")
