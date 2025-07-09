import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
import base64

# ------------------------ Page Setup ------------------------
st.set_page_config(
    page_title="Stroke Prediction App",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------ Load Model ------------------------
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------ Sidebar ------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4727/4727482.png", width=100)
    st.markdown("## About")
    st.write("This app predicts the risk of stroke based on patient information.")
    st.write("Developed by [Your Name]")
    st.write("📧 contact@example.com")

# ------------------------ Header ------------------------
st.markdown("<h1 style='text-align: center; color: white;'>🧠 Stroke Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
st.write("Please fill in the patient details below:")

# ------------------------ Input Form ------------------------
with st.form("stroke_form"):
    name = st.text_input("👤 Patient Name")
    age = st.slider("🎂 Age", min_value=0, max_value=120, value=30)
    heart_disease = st.selectbox("❤️ Do you have heart disease?", ["No", "Yes"])
    heart_disease_val = 1 if heart_disease == "Yes" else 0

    hypertension = st.selectbox("💉 Do you have hypertension?", ["No", "Yes"])
    hypertension_val = 1 if hypertension == "Yes" else 0

    avg_glucose_level = st.number_input("🧪 Average Glucose Level (mg/dL)", min_value=0.0, value=100.0)
    bmi = st.number_input("📏 BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0)

    smoking_status = st.selectbox("🚬 Smoking Status", ["formerly smoked", "smokes", "never smoked"])
    smoking_map = {"formerly smoked": 0, "smokes": 2, "never smoked": 1}
    smoking_val = smoking_map[smoking_status]

    submit = st.form_submit_button("🔍 Predict Stroke Risk")

# ------------------------ Prediction ------------------------
if submit:
    if not name.strip():
        st.warning("⚠️ Please enter the patient's name before proceeding.")
    else:
        # Prepare input
        input_data = pd.DataFrame([[age, heart_disease_val, avg_glucose_level, hypertension_val, bmi, smoking_val]],
                                  columns=['age', 'heart_disease', 'avg_glucose_level', 'hypertension', 'bmi', 'smoking_status'])
        input_scaled = scaler.transform(input_data)

        # Predict
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = int(probability >= 0.25)

        st.markdown("---")
        st.markdown(f"### 🧾 Stroke Probability: **{probability * 100:.2f}%**")

        if 40 <= probability * 100 < 60:
            result_text = "Borderline case - risk indicators present."
            st.warning(result_text)
        elif prediction == 1:
            result_text = "⚠ High Risk of Stroke — Please consult a doctor."
            st.error(result_text)
        else:
            result_text = "✅ Low Risk of Stroke — No immediate concern."
            st.success(result_text)

        # ------------------------ PDF Report ------------------------
        diet = (
            "- Eat more fruits, vegetables, and whole grains.\n"
            "- Limit saturated fats and sodium.\n"
            "- Drink enough water daily.\n"
            "- Avoid smoking and alcohol.\n"
            "- Include light physical activity like walking."
        )

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Stroke Prediction Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        # Table
        def add_row(label, value):
            pdf.cell(70, 10, label, border=1)
            pdf.cell(120, 10, str(value), border=1)
            pdf.ln()

        add_row("Patient Name", name)
        add_row("Age", age)
        add_row("Heart Disease", heart_disease)
        add_row("Hypertension", hypertension)
        add_row("Avg Glucose Level", avg_glucose_level)
        add_row("BMI", bmi)
        add_row("Smoking Status", smoking_status)
        add_row("Stroke Probability", f"{probability * 100:.2f}%")
        add_row("Prediction", "Yes" if prediction == 1 else "No")

        pdf.multi_cell(0, 10, f"\nSummary: {result_text}", border=1)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Basic Health & Diet Tips", ln=True)
        pdf.set_font("Arial", size=11)
        for line in diet.split('\n'):
            pdf.cell(0, 8, line.strip(), ln=True)

        pdf_output_path = "/tmp/stroke_report.pdf"
        pdf.output(pdf_output_path)

        # ------------------------ Download Link ------------------------
        with open(pdf_output_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="stroke_report.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
