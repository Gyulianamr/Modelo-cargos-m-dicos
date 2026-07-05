import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

NUMERIC_FEATURES = ["age", "bmi", "children", "charges"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]

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
    # las columnas originales.
    pca_path = "modelo/pca_riesgo_actuarial.pkl"
    pca = None
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        st.warning(
            "⚠️ No se encontró `models/pca_riesgo_actuarial.pkl`. "
            "El SVM fue entrenado con 2 componentes principales (PC1, PC2), "
            "así que sin el objeto PCA original no es posible generar una "
            "predicción correcta."
        )

except Exception as e:
    st.error(f"Error cargando los modelos:\n\n{e}")
    st.stop()

preprocessor = kmeans_pipeline.named_steps["preprocessor"]

# ==========================
# Cargar dataset para el resumen / visualización de clusters
# ==========================
@st.cache_data
def cargar_resumen_clusters():
    df = pd.read_csv("insurance.csv")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df = df.drop_duplicates()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    df["cluster"] = kmeans_pipeline.predict(X)

    # Mismo criterio del notebook: el riesgo se ordena por costo promedio
    orden_riesgo = df.groupby("cluster")["charges"].mean().sort_values().index.tolist()
    mapa_riesgo = {orden_riesgo[0]: "Bajo", orden_riesgo[1]: "Medio", orden_riesgo[2]: "Alto"}
    df["riesgo"] = df["cluster"].map(mapa_riesgo)

    resumen = df.groupby("riesgo").agg(
        clientes=("riesgo", "count"),
        edad_promedio=("age", "mean"),
        bmi_promedio=("bmi", "mean"),
        costo_promedio=("charges", "mean"),
        pct_fumadores=("smoker", lambda x: (x == "yes").mean() * 100),
    ).round(1).reindex(["Bajo", "Medio", "Alto"])

    X_prep = preprocessor.transform(X)
    pcs = pca.transform(X_prep) if pca is not None else None

    return df, resumen, pcs


df_full, resumen_clusters, pcs_full = None, None, None
try:
    df_full, resumen_clusters, pcs_full = cargar_resumen_clusters()
except Exception as e:
    st.info(
        "No se pudo generar el resumen de clusters (¿falta `insurance.csv` "
        f"en la raíz del proyecto?). Detalle: {e}"
    )

# ==========================
# Resumen / visualización general de los clusters (dataset completo)
# ==========================
if resumen_clusters is not None:
    st.header("Resumen general de los clusters (dataset)")
    st.dataframe(resumen_clusters, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    colores = {"Bajo": "#4CAF50", "Medio": "#FFC107", "Alto": "#F44336"}
    ax.bar(
        resumen_clusters.index,
        resumen_clusters["costo_promedio"],
        color=[colores.get(r, "gray") for r in resumen_clusters.index]
    )
    ax.set_ylabel("Costo médico promedio")
    ax.set_title("Costo médico promedio por nivel de riesgo")
    st.pyplot(fig)

    if pcs_full is not None:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        for riesgo, color in colores.items():
            mask = df_full["riesgo"] == riesgo
            ax2.scatter(
                pcs_full[mask, 0], pcs_full[mask, 1],
                label=riesgo, alpha=0.5, s=15, color=color
            )
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("Clusters visualizados con PCA")
        ax2.legend()
        st.pyplot(fig2)

st.divider()

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
st.caption(
    "ℹ️ El **costo médico** es, por lejos, la variable que más influye en el "
    "cluster y el nivel de riesgo asignados (el modelo agrupa clientes según "
    "su costo observado y ordena el riesgo por costo promedio de cada grupo). "
    "Edad, sexo, BMI, hijos, fumador y región tienen un peso mucho menor en "
    "comparación."
)

EXPLICACION_RIESGO = {
    "Bajo": "Cliente agrupado con perfiles de menor costo médico promedio.",
    "Medio": "Cliente agrupado con perfiles de costo y factores de riesgo intermedios.",
    "Alto": "Cliente agrupado con perfiles de mayor costo médico promedio y/o factores de riesgo relevantes.",
}

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

        # 2) Riesgo: el SVM necesita PC1 y PC2, generados aplicando el MISMO
        #    preprocesador del pipeline de kmeans y luego el PCA guardado.
        if pca is None:
            st.error("No se puede calcular el riesgo: falta el objeto PCA.")
        else:
            cliente_preprocesado = preprocessor.transform(cliente)
            componentes = pca.transform(cliente_preprocesado)
            cliente_pca = pd.DataFrame(componentes, columns=["PC1", "PC2"])

            # El SVM fue entrenado con las etiquetas de texto
            # ("Alto"/"Bajo"/"Medio"), así que ya devuelve el nombre listo.
            riesgo = svm.predict(cliente_pca)[0]

            st.success(f"**Cluster asignado:** {cluster}")
            st.success(f"**Nivel de riesgo actuarial:** {riesgo}")
            st.info(f"**Explicación:** {EXPLICACION_RIESGO.get(riesgo, '')}")

            # Ubicar al cliente dentro del mapa de PCA general, si existe
            if pcs_full is not None:
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                colores = {"Bajo": "#4CAF50", "Medio": "#FFC107", "Alto": "#F44336"}
                for r, color in colores.items():
                    mask = df_full["riesgo"] == r
                    ax3.scatter(
                        pcs_full[mask, 0], pcs_full[mask, 1],
                        label=r, alpha=0.3, s=12, color=color
                    )
                ax3.scatter(
                    componentes[0, 0], componentes[0, 1],
                    color="black", marker="*", s=250, label="Cliente ingresado"
                )
                ax3.set_xlabel("PC1")
                ax3.set_ylabel("PC2")
                ax3.set_title("Ubicación del cliente respecto a la cartera")
                ax3.legend()
                st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error realizando la predicción:\n\n{e}")
