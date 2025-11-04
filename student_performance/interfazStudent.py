import streamlit as st
import pandas as pd
import plotly.express as px
from student_performance.student_performance import StudentPerformanceProcessor

def show_student():
    """
    Interfaz de Streamlit para el procesamiento del dataset Student Performance
    """

    st.markdown("## Procesamiento del Dataset 'Student Performance'")

    # Ruta al dataset por defecto
    filepath = "student_performance/student_por.csv"

    # Crear instancia del procesador (guardada en sesión)
    if 'student_processor' not in st.session_state:
        st.session_state.student_processor = StudentPerformanceProcessor(filepath)

    processor = st.session_state.student_processor

    # ===== TABS PARA ORGANIZAR EL CONTENIDO =====
    tabs = st.tabs([
        "📊 Dataset Original",
        "⚙️ Procesamiento",
        "✅ Resultados Finales"
    ])

    # ===== TAB 1: DATASET ORIGINAL =====
    with tabs[0]:
        st.markdown("### 📥 Carga de Datos")

        if st.button("🔄 Cargar Dataset", type="primary"):
            result = processor.load_data()

            if result["success"]:
                st.success(result["message"])

                df_original = processor.get_original_data()

                # Mostrar métricas básicas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📋 Registros", df_original.shape[0])
                with col2:
                    st.metric("📊 Columnas", df_original.shape[1])
                with col3:
                    st.metric("❓ Valores Nulos", df_original.isnull().sum().sum())

                st.markdown("#### Vista previa del dataset original")
                st.dataframe(df_original.head(10), use_container_width=True)

                st.markdown("#### Información de columnas")
                st.dataframe(processor.get_column_info(), use_container_width=True)
            else:
                st.error(result["message"])

    # ===== TAB 2: PROCESAMIENTO =====
    with tabs[1]:
        st.markdown("### ⚙️ Pipeline de Procesamiento")

        if not processor.is_loaded():
            st.warning("⚠️ Primero debes cargar el dataset en la pestaña 'Dataset Original'")
        else:
            if st.button("▶️ Ejecutar Procesamiento Completo", type="primary"):
                with st.spinner("Procesando datos..."):
                    results = processor.process_all()

                st.success("✅ Procesamiento completado exitosamente!")

                # === 1️⃣ Variables Categóricas ===
                st.markdown("#### 1️⃣ Análisis de Variables Categóricas")
                st.write(f"Total de variables categóricas: **{results['categorical_analysis']['total']}**")
                st.write("Columnas categóricas:", results['categorical_analysis']['columns'])

                # === 2️⃣ Limpieza ===
                st.markdown("#### 2️⃣ Limpieza de Datos")
                st.json(results["cleaning"])

                # === 3️⃣ One Hot Encoding ===
                st.markdown("#### 3️⃣ One Hot Encoding")
                st.json(results["encoding"])

                # === 4️⃣ Normalización ===
                st.markdown("#### 4️⃣ Normalización de Variables Numéricas")
                st.json(results["normalization"])

                # === 5️⃣ Separación X / y ===
                st.markdown("#### 5️⃣ Separación de Características (X) y Variable Objetivo (y)")
                st.json(results["separation"])

                # === 6️⃣ División Train / Test ===
                st.markdown("#### 6️⃣ División de Datos (80% - 20%)")
                st.json(results["split"])

                # === 7️⃣ Correlación ===
                st.markdown("#### 7️⃣ Correlación entre Notas (G1, G2, G3)")
                corr = results["correlation"]
                st.dataframe(corr["correlation_matrix"], use_container_width=True)
                st.info(f"📈 Correlación más fuerte: **{corr['strongest_correlation']['pair']}** = {corr['strongest_correlation']['value']:.2f}")

                # Gráfico de correlación
                fig = px.imshow(
                    corr["correlation_matrix"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title="Matriz de Correlación entre Notas"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 3: RESULTADOS =====
    with tabs[2]:
        st.markdown("### ✅ Dataset Procesado - Vista Final")

        if not processor.is_processed():
            st.warning("⚠️ Primero ejecuta el procesamiento en la pestaña 'Procesamiento'")
        else:
            df_processed = processor.get_processed_data()

            st.markdown("#### 🔝 Primeros 5 registros del dataset procesado")
            st.dataframe(df_processed.head(), use_container_width=True)

            st.markdown("#### 📊 Estadísticas del Dataset Procesado")
            st.dataframe(processor.get_statistics(), use_container_width=True)

            # Botón para descargar dataset procesado
            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar datos procesados (CSV)",
                data=csv,
                file_name="student_processed.csv",
                mime="text/csv"
            )
