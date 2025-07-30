import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Comprehensive Health Status Predictor", layout="wide")

@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    df = pd.read_csv('health_tracker_500.csv')

    # Define binary health status combining multiple indicators
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
    # Features to use
    features = [
        'Height_cm', 'Weight_kg', 'BMI', 'BP_Systolic', 'BP_Diastolic', 
        'Cholesterol_mg_dL', 'ActivityLevel', 'DietType', 'HealthScore'
    ]
    target = 'HealthyStatus'

    # Encode categorical features
    encoders = {}
    for col in ['ActivityLevel', 'DietType']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode target labels
    target_le = LabelEncoder()
    df[target] = target_le.fit_transform(df[target])
    encoders['target'] = target_le

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation (optional for console logs)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Validation Accuracy: {acc*100:.2f}%")

    return model, encoders, features, target_le

def preprocess_input(input_df, encoders, features):
    # Encode categorical columns
    for col in encoders:
        if col == 'target':
            continue
        le = encoders[col]
        if col in input_df.columns:
            # Replace unseen labels with first known in encoder classes
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])
        else:
            input_df[col] = le.transform([le.classes_[0]])[0]

    # Add missing numeric columns if any
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0  # or could use median or other default

    return input_df[features]

def main():
    st.title("Comprehensive Health Status Predictor")

    st.markdown("""
        This app predicts if a person is **Healthy and Fit** or **Unhealthy and Not Fit**  
        using multiple health indicators including BMI, blood pressure, cholesterol, diet, activity, and health score.
    """)

    df = load_and_prepare_data()
    model, encoders, features, target_le = train_model(df)

    choice = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Batch Prediction"])

    if choice == "Single Prediction":
        st.header("Input health details:")

        input_data = {}
        for f in features:
            if f in encoders:
                options = list(encoders[f].classes_)
                input_data[f] = st.selectbox(f"{f} (categorical)", options)
            elif f == 'HealthScore':
                input_data[f] = st.slider(f"{f}", 0.0, 100.0, 50.0)
            else:
                min_val = float(np.floor(df[f].min()))
                max_val = float(np.ceil(df[f].max()))
                median_val = float(np.median(df[f]))
                input_data[f] = st.number_input(f"{f} (numeric)", min_val, max_val, median_val)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_encoded = preprocess_input(input_df, encoders, features)
            pred = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0]

            class_name = target_le.inverse_transform([pred])[0]
            prob = proba[pred]

            st.success(f"Prediction: **{class_name}**")
            st.info(f"Confidence: **{prob:.2%}**")

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

            numeric_feats = [f for f in features if f not in encoders]
            st.subheader("Sample Feature Distributions")
            for f in numeric_feats[:4]:
                fig2, ax2 = plt.subplots()
                sns.histplot(batch_df[f], kde=True, ax=ax2)
                ax2.set_title(f"Distribution of {f}")
                st.pyplot(fig2)

if __name__ == "__main__":
    main()
