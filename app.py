import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Health Status Predictor", layout="wide")

# --------- Step 1: Load data & add Healthy/Unhealthy label ---------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    df = pd.read_csv("health_tracker_500.csv")

    # Define health categories
    healthy_bmi = ['Normal', 'Underweight']
    healthy_bp = ['Normal', 'Elevated']
    healthy_chol = ['Desirable', 'Borderline']

    df['HealthyStatus'] = np.where(
        (df['BMI_Category'].isin(healthy_bmi)) &
        (df['BP_Category'].isin(healthy_bp)) &
        (df['Cholesterol_Category'].isin(healthy_chol)),
        'Healthy and Fit',
        'Unhealthy and Not Fit'
    )
    return df

# --------- Step 2: Train model ---------
@st.cache_resource(show_spinner=True)
def train_model(df):
    features = [
        'Height_cm', 'Weight_kg', 'BMI', 'BP_Systolic', 'BP_Diastolic', 
        'Cholesterol_mg_dL', 'ActivityLevel', 'DietType'
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

# --------- Streamlit UI ---------
def main():
    st.title("Health Status Predictor")
    st.markdown("""
        This app predicts whether a person is **Healthy and Fit** or **Unhealthy and Not Fit**  
        using BMI, blood pressure, cholesterol, activity, and diet.
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

        # Calculate BMI
        bmi = round(weight / ((height / 100) ** 2), 1)

        if st.button("Predict"):
            input_data = {
                'Height_cm': height,
                'Weight_kg': weight,
                'BMI': bmi,
                'BP_Systolic': bp_sys,
                'BP_Diastolic': bp_dia,
                'Cholesterol_mg_dL': cholesterol,
                'ActivityLevel': activity,
                'DietType': diet
            }
            input_df = pd.DataFrame([input_data])
            input_encoded = preprocess_input(input_df, encoders, features)
            pred = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0]
            class_name = target_le.inverse_transform([pred])[0]
            prob = proba[pred]

            # Color coding
            color = "green" if class_name == "Healthy and Fit" else "red"

            st.markdown(
                f"""
                <div style="background-color:{color};padding:15px;border-radius:10px">
                    <h3 style="color:white;">Prediction: {class_name}</h3>
                    <p style="color:white;">Confidence: {prob:.2%}</p>
                    <p style="color:white;">Your BMI: {bmi}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

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
