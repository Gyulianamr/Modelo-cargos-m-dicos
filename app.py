import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# Configuración de página
st.set_page_config(
    page_title="Predicción de Riesgo Actuarial",
    page_icon="📊",
    layout="wide"
)

# Estilo para mejorar contenedores (opcional)
st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    </style>
""", unsafe_allow_html=True)

# Encabezado
st.title("📊 Predicción de Riesgo Actuarial")
st.markdown("### Genesis Yuliana Medina Ramos | **Cuenta:** 20231900117")
st.divider()

# ==========================
# Carga de Modelos y Datos
# ==========================
@st.cache_resource
def load_assets():
    # Asumimos que los archivos existen en las rutas especificadas
    kmeans_pipeline = joblib.load("modelo/kmeans_riesgo_actuarial.pkl")
    svm_data = joblib.load("modelo/svm_riesgo_actuarial.pkl")
    
    if isinstance(svm_data, dict):
        svm = svm_data.get("rbf", list(svm_data.values())[0])
    else:
        svm = svm_data
        
    pca = joblib.load("modelo/pca_riesgo_actuarial.pkl") if os.path.exists("modelo/pca_riesgo_actuarial.pkl") else None
    preprocessor = kmeans_pipeline.named_steps["preprocessor"]
    
    # Cargar datos
    df = pd.read_csv("insurance.csv")
    df = df.drop_duplicates()
    X = df[["age", "bmi", "children", "charges", "sex", "smoker", "region"]].copy()
    df["cluster"] = kmeans_pipeline.predict(X)
    
    # Calcular resumen
    orden = df.groupby("cluster")["charges"].mean().sort_values().index.tolist()
    mapa = {orden[0]: "Bajo", orden[1]: "Medio", orden[2]: "Alto"}
    df["riesgo"] = df["cluster"].map(mapa)
    
    resumen = df.groupby("riesgo").agg(
        clientes=("riesgo", "count"),
        edad_prom=("age", "mean"),
        bmi_prom=("bmi", "mean"),
        costo_prom=("charges", "mean")
    ).round(1).reindex(["Bajo", "Medio", "Alto"])
    
    return kmeans_pipeline, svm, pca, preprocessor, df, resumen

try:
    kmeans, svm, pca, preprocessor, df_full, resumen = load_assets()
except Exception as e:
    st.error(f"Error al cargar recursos: {e}")
    st.stop()

# ==========================
# Interfaz con Pestañas
# ==========================
tab1, tab2 = st.tabs(["📈 Análisis de Cartera", "🔍 Nueva Predicción"])

with tab1:
    st.header("Resumen del Modelo")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(resumen, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        colores = {"Bajo": "#4CAF50", "Medio": "#FFC107", "Alto": "#F44336"}
        resumen["costo_prom"].plot(kind='bar', color=[colores.get(r, "gray") for r in resumen.index], ax=ax)
        ax.set_title("Costo Médico Promedio")
        st.pyplot(fig)

with tab2:
    st.header("Ingrese los datos del cliente")
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Edad", 18, 100, 30)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        with c2:
            sex = st.selectbox("Sexo", ["male", "female"])
            smoker = st.selectbox("Fumador", ["yes", "no"])
        with c3:
            children = st.number_input("Hijos", 0, 10, 0)
            region = st.selectbox("Región", ["southwest", "southeast", "northwest", "northeast"])
        
        charges = st.number_input("Costo médico", 0.0, 100000.0, 10000.0)
        submitted = st.form_submit_button("Predecir Riesgo")

    if submitted:
        # Lógica de predicción
        cliente = pd.DataFrame([[age, sex, bmi, children, smoker, region, charges]], 
                               columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"])
        
        cluster = kmeans.predict(cliente)[0]
        cliente_pre = preprocessor.transform(cliente)
        componentes = pca.transform(cliente_pre)
        riesgo = svm.predict(pd.DataFrame(componentes, columns=["PC1", "PC2"]))[0]
        
        # Resultados destacados
        m1, m2 = st.columns(2)
        m1.metric("Cluster Asignado", cluster)
        m2.metric("Nivel de Riesgo", riesgo)
        
        with st.expander("Ver detalles y análisis visual"):
            st.info(f"**Explicación:** El cliente ha sido clasificado como riesgo {riesgo}.")
            # Gráfico de ubicación
            fig3, ax3 = plt.subplots()
            ax3.scatter(componentes[0, 0], componentes[0, 1], color="black", marker="*", s=200, label="Cliente")
            ax3.set_title("Ubicación del cliente en el mapa de riesgo")
            ax3.legend()
            st.pyplot(fig3)
