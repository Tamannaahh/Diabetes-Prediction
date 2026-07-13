"""
Streamlit app for the Predicting Diabetes project.

Loads the trained Random Forest model and the MinMaxScaler saved by the
notebook (diabetes_model.joblib / scaler.joblib) and lets a user enter a
patient's health measurements to get a diabetes risk prediction.

Run with:
    streamlit run app.py

Make sure diabetes_model.joblib and scaler.joblib are in the same folder
(they're created by running the last few cells of Predicting_Diabetes.ipynb).
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# These must match the columns the notebook log-transforms and scales.
LOG_COLS = ["SkinThickness", "Insulin"]
SCALED_COLS = ["Insulin", "SkinThickness", "DiabetesPedigreeFunction"]


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("diabetes_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


def preprocess(raw_input: dict, scaler) -> pd.DataFrame:
    """Apply the same log-transform + scaling steps used during training."""
    df = pd.DataFrame([raw_input], columns=FEATURE_ORDER)

    # log-transform skewed columns, same as in the notebook
    for col in LOG_COLS:
        df[col] = np.log(df[col] + 1)

    # scale the same columns the notebook scaled, using the *fitted* scaler
    df[SCALED_COLS] = scaler.transform(df[SCALED_COLS])

    return df


def main():
    st.title("🩺 Diabetes Risk Predictor")
    st.write(
        "Enter a patient's health measurements below to estimate their "
        "risk of diabetes, using a Random Forest model trained on the "
        "Pima Indians Diabetes dataset."
    )

    try:
        model, scaler = load_model_and_scaler()
    except FileNotFoundError:
        st.error(
            "Model files not found. Run the notebook (Predicting_Diabetes.ipynb) "
            "first to generate `diabetes_model.joblib` and `scaler.joblib`, "
            "then place them in this same folder."
        )
        st.stop()

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4, step=0.01)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw_input = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }

        X = preprocess(raw_input, scaler)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        st.subheader("Result")
        if prediction == 1:
            st.error(f"⚠️ Higher risk of diabetes  (model confidence: {probability:.1%})")
        else:
            st.success(f"✅ Lower risk of diabetes  (model confidence: {1 - probability:.1%})")

        st.caption(
            "This is a machine learning estimate based on a sample dataset, "
            "not a medical diagnosis. Please consult a doctor for an actual evaluation."
        )

        with st.expander("See the values used for this prediction"):
            st.json(raw_input)


if __name__ == "__main__":
    main()
    