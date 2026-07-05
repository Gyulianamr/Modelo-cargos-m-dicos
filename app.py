"""
Proyecto: Predicción de Riesgo Actuarial en Seguros Médicos

Nombre: Genesis Yuliaan Medina Ramos
Cuenta: 20231900117

Aplicación desarrollada con Streamlit
"""

import streamlit as st
import pandas as pd
import joblib

# ===============================
# Información del Proyecto
# ===============================
st.set_page_config(
    page_title="Predicción de Riesgo Actuarial",
    page_icon="📊",
    layout="wide"
)

st.title("Predicción de Riesgo Actuarial en Seguros Médicos")
st.markdown("### Genesis Yuliaan Medina Ramos")
st.markdown("**Cuenta:** 20231900117")

st.divider()

# Cargar modelos
kmeans = joblib.load("models/kmeans_riesgo_actuarial.pkl")
svm = joblib.load("models/svm_riesgo_actuarial.pkl")

# Aquí continúa el resto de tu código...

st.title("Predicción de Riesgo de Seguro Médico")

st.header("Ingrese los datos del cliente")

age = st.number_input("Edad", 18, 100, 30)
sex = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.number_input("Número de hijos", 0, 10, 0)
smoker = st.selectbox("Fumador", ["yes", "no"])
region = st.selectbox(
    "Región",
    ["southwest", "southeast", "northwest", "northeast"]
)
charges = st.number_input("Costo médico", 0.0, 100000.0, 10000.0)

if st.button("Predecir"):

    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])

    cluster = kmeans.predict(cliente)[0]

    cliente["cluster"] = cluster

    riesgo = svm.predict(cliente)[0]

    st.success(f"Cluster: {cluster}")
    st.success(f"Nivel de riesgo: {riesgo}")