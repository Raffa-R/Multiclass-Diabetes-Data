import os
import subprocess
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
  
# Load Model dan Scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Session State untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "input"

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# Sidebar Button
if st.sidebar.button("📝 Input Data Patient"):
    st.session_state.page = "input"

# Sidebar Expander untuk Machine Learning
with st.sidebar.expander("📊 Machine Learning", expanded=True):
    # Button untuk tiap jenis tampilan hasil
    if st.button("Result by Table Data", key="btn_table"):
        st.session_state.page = "table"
    if st.button("Result by Line Chart", key="btn_line"):
        st.session_state.page = "line_chart"
    if st.button("Result by Pie Chart", key="btn_pie"):
        st.session_state.page = "pie_chart"
    if st.button("Result by Bar Chart", key="btn_bar"):
        st.session_state.page = "bar_chart"
    
# Fungsi Prediksi
def predict_diabetic(model, scaler, input_data):
    """Functions for data scaling, prediction, and probability"""

    # Scaled
    input_scaled = scaler.transform(input_data)

    # Prediksi model
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    return prediction, probability

# Input Data Pasien
if st.session_state.page == "input":
    st.title("Predicting Diabetes Risk with Machine Learning")
    st.write("Enter patient data to predict the likelihood of diabetes. "
             "This prediction is **not a medical diagnosis**.")

    gender = st.radio("Gender", ['Female', 'Male'], index=0)
    age = st.number_input("Age (Year)", min_value=25, max_value=77, value=30, step=1)
    urea = st.number_input("Blood Urea Levels", min_value=1.1, max_value=26.40, value=5.0, step=0.1)
    cr = st.number_input("Blood Creatinine Levels", min_value=6, max_value=800, value=7, step=1)
    hba1c = st.number_input("Glycated haemoglobin", min_value=0.9, max_value=14.6, value=5.5, step=0.1)
    chol = st.number_input("Total Cholesterol", min_value=0.5, max_value=9.5, value=3.0, step=0.1)
    hdl = st.number_input("High-Density Lipoprotein", min_value=0.4, max_value=4.0, value=2.0, step=0.1)
    ldl = st.number_input("Low-Density Lipoprotein", min_value=0.3, max_value=5.6, value=2.0, step=0.1)
    vldl = st.number_input("Very Low-Density Lipoprotein", min_value=0.2, max_value=31.8, value=2.0, step=0.1)
    bmi = st.number_input("BMI (kg/m²)", min_value=19.0, max_value=43.25, value=23.0, step=0.1)
    urea_cr_ratio = st.number_input("Urea/Creatinine Ratio", min_value=0.0124, max_value=0.6500, value=0.0480, step=0.01)
    bmi_hba1c = st.number_input("BMI x HbA1c", min_value=19.8, max_value=475.2, value=175.2, step=0.1)
    age_bmi = st.number_input("Usia x BMI", min_value=550.0, max_value=2553.0, value=750.0, step=1.0)

    gender_encoded = 0 if gender == "Female" else 1

    st.session_state.input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, 
                                             hdl, ldl, vldl, bmi, urea_cr_ratio,
                                             bmi_hba1c, age_bmi]])
                                             
# Table Page
elif st.session_state.page == "table":
    st.title("📋 Tabel Data Patient")
    if st.session_state.input_data is None:
        st.warning("⚠️ Please fill in the patient's details first in the *Patient Data Input* menu..")
    else:
        st.session_state.data = {
            "Gender": ["Male" if st.session_state.input_data[0][0] == 1 else "Female"],
            "Age": [round(float(st.session_state.input_data[0][1]), 2)],
            "Urea": [round(float(st.session_state.input_data[0][2]), 2)],
            "Cr": [round(float(st.session_state.input_data[0][3]), 2)],
            "HbA1c": [round(float(st.session_state.input_data[0][4]), 2)],
            "Chol": [round(float(st.session_state.input_data[0][5]), 2)],
            "HDL": [round(float(st.session_state.input_data[0][6]), 2)],
            "LDL": [round(float(st.session_state.input_data[0][7]), 2)],
            "VLDL": [round(float(st.session_state.input_data[0][8]), 2)],
            "BMI": [round(float(st.session_state.input_data[0][9]), 2)],
            "Urea/Cr": [round(float(st.session_state.input_data[0][10]), 2)],
            "BMI x HbA1c": [round(float(st.session_state.input_data[0][11]), 2)],
            "AGE x BMI": [round(float(st.session_state.input_data[0][12]), 2)]
        }

        data_input = pd.DataFrame(st.session_state.data)
        st.session_state.data_input = data_input
        data_input_transpose = data_input.T
        data_input_transpose.rename(columns={0: "Data Patient"}, inplace=True)
        st.dataframe(data_input_transpose)

 # Prediction
        prediction, prob = predict_diabetic(model, scaler, st.session_state.input_data)

        st.title("Prediction Results")
        if prediction[0] == 0:
            st.success("✅ The patient is Non Diabetic")
        elif prediction[0] == 1:
            st.warning("⚠️ Patients at HIGH RISK of developing Predict Diabetic")
        else:
            st.error("⚠️ Patients at HIGH RISK of developing Diabetic")


# Line Chart Page
elif st.session_state.page == "line_chart":
    st.title("Prediction Results & Probability")
  
    # Validasi input data
    if st.session_state.input_data is None or 'data' not in st.session_state:
        st.warning("⚠️ Please fill in the patient's details first in the Patient Data Input menu..")
    else:
        data_input = st.session_state.data_input

        # Displaying patient data: Gender, Age, BMI
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gender", data_input["Gender"][0])
        with col2:
            st.metric("Age", f"{int(data_input['Age'][0])} year")
        with col3:
            st.metric("BMI", f"{data_input['BMI'][0]:.1f} kg/m²")

        # Line Chart
        features = ["Urea", "Cr", "HbA1c", "Chol", "HDL", "LDL", "VLDL"]
        values = [data_input[feat][0] for feat in features]

        data_line = pd.DataFrame({
            "Fitur": features,
            "Nilai": values
        })

        st.subheader("📈 Patient Data Visualisation (Line Chart)")
        st.line_chart(
            data_line.set_index("Fitur"),
            y="Nilai",
            height=300
        )

        # Display data Urea/Cr, BMI x HbA1c, AGE x BMI'
        st.header("🧬 Derivative Features from Clinical Calculations")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Urea/Cr', f"{data_input['Urea/Cr'][0]:.1f} kg/m²")
        with col2:
            st.metric('BMI x HbA1c', f"{data_input['BMI x HbA1c'][0]:.1f} kg/m²")
        with col3:
            st.metric('AGE x BMI', f"{data_input['AGE x BMI'][0]:.1f} kg/m²")

        # Prediction
        prediction, prob = predict_diabetic(model, scaler, st.session_state.input_data)

        if prediction[0] == 0:
            st.success("✅ The patient is Non Diabetic")
        elif prediction[0] == 1:
            st.warning("⚠️ Patients at HIGH RISK of Predict Diabetic")
        else:
            st.error("⚠️ The patient is Diabetic")

# Pie Chart Page
elif st.session_state.page == "pie_chart":
    st.title("Prediction Results & Probability")

    if st.session_state.input_data is None or 'data' not in st.session_state:
        st.warning("⚠️ Please fill in the patient's details first in the Patient Data Input menu..")
    else:
        data_input = st.session_state.data_input

        # Display data nama, year, BMI
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gender", data_input["Gender"][0])
        with col2:
            st.metric("Age", f"{int(data_input['Age'][0])} year")
        with col3:
            st.metric("BMI", f"{data_input['BMI'][0]:.1f} kg/m²")

        # Prediction
        prediction, prob = predict_diabetic(model, scaler, st.session_state.input_data)

        # Pie chart Probability
        all_labels = ["Non Diabetic", "Predict Diabetic", "Diabetic"]
        all_colors = ["forestgreen", "darkorange", "firebrick"]

        # Filter hanya yang lebih dari 1%
        threshold = 0.01
        labels = [l for l, p in zip(all_labels, prob) if p > threshold]
        colors = [c for c, p in zip(all_colors, prob) if p > threshold]
        probability = [p for p in prob if p > threshold]

        # Pesan Prediction Results
        if prediction[0] == 0:
            st.success("✅ The patient is Non Diabetic")
        elif prediction[0] == 1:
            st.warning("⚠️ Patients at HIGH RISK of Predict Diabetic")
        else:
            st.error("⚠️ The patient is Diabetic")

        # Fungsi untuk Display persentase hanya jika > 0
        def autopct_format(pct):
            return ("%1.1f%%" % pct) if pct > 0 else ""

        # Pie Chart
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            probability, labels=labels, colors=colors,
            autopct=autopct_format, startangle=60,
            counterclock=False, radius=0.6,
            labeldistance=1.1, pctdistance=0.5,
            textprops={'fontsize': 8}
        )
        ax.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)

# Bar Chart Page
elif st.session_state.page == "bar_chart":
    st.title("Prediction Results & Probability")
    
    if st.session_state.input_data is None or 'data' not in st.session_state:
        st.warning("⚠️ Please fill in the patient's details first in the Patient Data Input menu..")
    else:
        data_input = st.session_state.data_input

        # Display data nama, year, BMI
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gender", data_input["Gender"][0])
        with col2:
            st.metric("Age", f"{int(data_input['Age'][0])} year")
        with col3:
            st.metric("BMI", f"{data_input['BMI'][0]:.1f} kg/m²")

        # Bar Chart Terkelompok
        # 1. "Kidney Function Profile & Diabetes Risk"
        fitur_ginjal = ["Urea", "Cr", "HbA1c"]
        nilai_ginjal = [data_input[feat][0] for feat in fitur_ginjal]
        df_ginjal = pd.DataFrame({"Fitur": fitur_ginjal, "Nilai": nilai_ginjal})
        st.subheader("**⚖️ Kidney Function Profile & Diabetes Risk**")
        st.bar_chart(df_ginjal.set_index("Fitur"), y="Nilai", height=250)

        # 2. Lipid Profile
        fitur_lipid = ["Chol", "HDL", "LDL", "VLDL"]
        nilai_lipid = [data_input[feat][0] for feat in fitur_lipid]
        df_lipid = pd.DataFrame({"Fitur": fitur_lipid, "Nilai": nilai_lipid})
        st.subheader("**🧪Lipid Profile**")
        st.bar_chart(df_lipid.set_index("Fitur"), y="Nilai", height=250)

        # Display data Urea/Cr, BMI x HbA1c, AGE x BMI'
        st.header("🧬 Derivative Features from Clinical Calculations")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Urea/Cr', f"{data_input['Urea/Cr'][0]:.1f} kg/m²")
        with col2:
            st.metric('BMI x HbA1c', f"{data_input['BMI x HbA1c'][0]:.1f} kg/m²")
        with col3:
            st.metric('AGE x BMI', f"{data_input['AGE x BMI'][0]:.1f} kg/m²")

       # Prediction
        prediction, prob = predict_diabetic(model, scaler, st.session_state.input_data)

        if prediction[0] == 0:
            st.success("✅ The patient is Non Diabetic")
        elif prediction[0] == 1:
            st.warning("⚠️ Patients at HIGH RISK of Predict Diabetic")
        else:
            st.error("⚠️ The patient is Diabetic")
