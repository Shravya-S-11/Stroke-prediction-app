import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Stroke Prediction App", layout="centered")

model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title(" Stroke Prediction App")
st.write("Enter the patient details below to assess stroke risk:")

age = st.slider("Age", min_value=0, max_value=120, value=30)

heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
heart_disease = 1 if heart_disease == "Yes" else 0

hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
hypertension = 1 if hypertension == "Yes" else 0

avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, value=100.0)

bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)

smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "smokes", "never smoked"])
smoking_map = {"formerly smoked": 0, "smokes": 2, "never smoked": 1}
smoking_status = smoking_map[smoking_status]

if st.button("Predict Stroke Risk"):
    input_data = pd.DataFrame([[age, heart_disease, avg_glucose_level, hypertension, bmi, smoking_status]],
                              columns=['age', 'heart_disease', 'avg_glucose_level', 'hypertension', 'bmi', 'smoking_status'])
    
    input_scaled = scaler.transform(input_data)

    threshold = 0.25
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(probability >= threshold)  

    st.markdown("---")
    st.markdown(f"###  Probability of Stroke: {probability*100:.2f}%")

    if 40 <= probability * 100 < 60:
       st.warning("Borderline case — risk indicators present.\n\n"
                  f"*Stroke Prediction:* {'Yes' if prediction == 1 else 'No'}")
    elif prediction == 1:
       st.error("⚠*High Risk of Stroke* — Please consult a doctor.\n\n"
                f"*Stroke Prediction:* Yes")
    else:
        st.success("*Low Risk of Stroke* — No immediate concern.\n\n"
                  f"*Stroke Prediction:* No")
