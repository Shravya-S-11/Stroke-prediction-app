import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
import base64
import os

st.set_page_config(page_title="Stroke Prediction App", layout="centered")

model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Stroke Prediction App")
st.write("Enter the patient details below to assess stroke risk:")

# Ask for name
name = st.text_input("Patient Name")
age = st.slider("Age", min_value=0, max_value=120, value=30)

heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
heart_disease_val = 1 if heart_disease == "Yes" else 0

hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
hypertension_val = 1 if hypertension == "Yes" else 0

avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, value=100.0)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)

smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "smokes", "never smoked"])
smoking_map = {"formerly smoked": 0, "smokes": 2, "never smoked": 1}
smoking_val = smoking_map[smoking_status]

if st.button("Predict Stroke Risk"):
    if name.strip() == "":
        st.warning("⚠ Please enter the patient's name before proceeding.")
    else:
        input_data = pd.DataFrame([[age, heart_disease_val, avg_glucose_level, hypertension_val, bmi, smoking_val]],
                                  columns=['age', 'heart_disease', 'avg_glucose_level', 'hypertension', 'bmi', 'smoking_status'])
        input_scaled = scaler.transform(input_data)

        threshold = 0.25
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = int(probability >= threshold)

        st.markdown("---")
        st.markdown(f"### Probability of Stroke: `{probability*100:.2f}%`")

        if 40 <= probability * 100 < 60:
            result_text = "Borderline case — risk indicators present."
            st.warning(result_text + f"\n\nStroke Prediction: {'Yes' if prediction == 1 else 'No'}")
        elif prediction == 1:
            result_text = "⚠ High Risk of Stroke — Please consult a doctor."
            st.error(result_text + "\n\nStroke Prediction: Yes")
        else:
            result_text = "Low Risk of Stroke — No immediate concern."
            st.success(result_text + "\n\nStroke Prediction: No")

        # ---------------- PDF Generation ----------------
        pdf = FPDF()
        pdf.add_page()

        # Load Unicode font
        font_path = "DejaVuSans.ttf"
        if not os.path.isfile(font_path):
            st.error("Font file 'DejaVuSans.ttf' not found in project directory. Please upload it.")
        else:
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", '', 14)
            pdf.cell(0, 10, "Stroke Prediction Report", ln=True, align='C')
            pdf.set_font("DejaVu", '', 12)
            pdf.ln(10)

            def add_row(label, value):
                pdf.cell(70, 10, str(label), border=1)
                pdf.cell(120, 10, str(value), border=1)
                pdf.ln()

            # Table content
            add_row("Patient Name", name)
            add_row("Age", age)
            add_row("Heart Disease", "Yes" if heart_disease_val else "No")
            add_row("Hypertension", "Yes" if hypertension_val else "No")
            add_row("Avg Glucose Level", avg_glucose_level)
            add_row("BMI", bmi)
            add_row("Smoking Status", smoking_status)
            add_row("Stroke Probability", f"{probability*100:.2f}%")
            add_row("Prediction", "Yes" if prediction else "No")

            pdf.multi_cell(0, 10, f"\nSummary: {result_text}", border=1)

            # Save & provide download
            pdf_output_path = "/tmp/stroke_report.pdf"
            pdf.output(pdf_output_path)

            with open(pdf_output_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="stroke_report.pdf">📄 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
