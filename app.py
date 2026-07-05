
import streamlit as st
import pandas as pd
import joblib
import os
 
st.set_page_config(
    page_title="Predicción de Riesgo Actuarial",
    page_icon="📊",
    layout="centered"
)
 
st.title("Predicción de Riesgo Actuarial en Seguros Médicos")
st.markdown("### Genesis Yuliana Medina Ramos")
st.markdown("**Cuenta:** 20231900117")
st.divider()
 
# ==========================
# Cargar modelos
# ==========================
try:
    # Pipeline completo: preprocesador (scaler + one-hot) + KMeans
    kmeans_pipeline = joblib.load("modelo/kmeans_riesgo_actuarial.pkl")
 
    svm_data = joblib.load("modelo/svm_riesgo_actuarial.pkl")
 
    if isinstance(svm_data, dict):
        if "rbf" in svm_data:
            svm = svm_data["rbf"]
        elif "linear" in svm_data:
            svm = svm_data["linear"]
        else:
            svm = list(svm_data.values())[0]
    else:
        svm = svm_data
 
    # El SVM fue entrenado sobre PC1, PC2 (salida de un PCA), no sobre
    # las columnas originales. Ese PCA debe haberse guardado por separado.
    pca_path = "modelo/pca_riesgo_actuarial.pkl"
    pca = None
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        st.warning(
            "⚠️ No se encontró `modelo/pca_riesgo_actuarial.pkl`. "
            "El SVM fue entrenado con 2 componentes principales (PC1, PC2), "
            "así que sin el objeto PCA original no es posible generar una "
            "predicción correcta. Exporta el PCA desde tu notebook de "
            "entrenamiento (joblib.dump(pca, 'pca_riesgo_actuarial.pkl')) "
            "y colócalo en la carpeta modelo/."
        )
 
except Exception as e:
    st.error(f"Error cargando los modelos:\n\n{e}")
    st.stop()
 
# ==========================
# Entradas
# ==========================
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
 
# ==========================
# Predicción
# ==========================
if st.button("Predecir"):
 
    cliente = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region],
        "charges": [charges]
    })
 
    try:
        # 1) Cluster: el pipeline aplica su propio preprocesador internamente
        cluster = kmeans_pipeline.predict(cliente)[0]
        st.success(f"Cluster asignado: {cluster}")
 
        # 2) Riesgo: el SVM necesita PC1 y PC2, generados aplicando el MISMO
        #    preprocesador del pipeline de kmeans y luego el PCA guardado.
        if pca is None:
            st.error(
                "No se puede calcular el riesgo: falta el objeto PCA "
                "(modelo/pca_riesgo_actuarial.pkl). Ver advertencia arriba."
            )
        else:
            preprocessor = kmeans_pipeline.named_steps["preprocessor"]
            cliente_preprocesado = preprocessor.transform(cliente)
 
            componentes = pca.transform(cliente_preprocesado)
            cliente_pca = pd.DataFrame(componentes, columns=["PC1", "PC2"])
 
            # El SVM fue entrenado con las etiquetas de texto
            # ("Alto"/"Bajo"/"Medio"), así que ya devuelve el nombre listo.
            riesgo = svm.predict(cliente_pca)[0]
            st.success(f"Nivel de riesgo: {riesgo}")
 
    except Exception as e:
        st.error(f"Error realizando la predicción:\n\n{e}")