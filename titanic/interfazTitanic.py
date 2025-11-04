# titanic/interfazTitanic.py
import streamlit as st
import pandas as pd
from titanic.titanic import TitanicProcessor

def show_titanic():
    """
    Interfaz de Streamlit para el procesamiento del dataset Titanic
    """
    
    st.markdown("##  Análisis del Dataset Titanic")

    # Ruta al dataset
    filepath = "titanic/titanic_dataset.csv"
    
    # Crear instancia del procesador
    if 'titanic_processor' not in st.session_state:
        st.session_state.titanic_processor = TitanicProcessor(filepath)
    
    processor = st.session_state.titanic_processor
    
    # ===== TABS PARA ORGANIZAR EL CONTENIDO =====
    tabs = st.tabs([
        "📊 Dataset Original",
        "🔧 Procesamiento",
        "✅ Resultados Finales"
    ])
    
    # ===== TAB 1: DATASET ORIGINAL =====
    with tabs[0]:
        st.markdown("### 📥 Carga de Datos")
        
        if st.button("🔄 Cargar Dataset", type="primary"):
            success, message = processor.load_data()
            if success:
                st.success(message)
                
                # Mostrar información del dataset
                df_original = processor.get_original_data()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📋 Total de Registros", df_original.shape[0])
                with col2:
                    st.metric("📊 Total de Columnas", df_original.shape[1])
                with col3:
                    st.metric("❓ Valores Nulos", df_original.isnull().sum().sum())
                
                st.markdown("#### Vista previa del dataset original")
                st.dataframe(df_original.head(10), use_container_width=True)
                
                st.markdown("#### Información de columnas")
                st.dataframe(df_original.dtypes.to_frame(name='Tipo de Dato'), use_container_width=True)
            else:
                st.error(message)
    
    # ===== TAB 2: PROCESAMIENTO =====
    with tabs[1]:
        st.markdown("### ⚙️ Pipeline de Procesamiento")
        
        if processor.df is None:
            st.warning("⚠️ Primero debes cargar el dataset en la pestaña 'Dataset Original'")
        else:
            if st.button("▶️ Ejecutar Procesamiento Completo", type="primary"):
                with st.spinner("Procesando datos..."):
                    results = processor.process_all()
                
                # Mostrar resultados paso a paso
                st.success("✅ Procesamiento completado exitosamente!")
                
                # INSTRUCCIÓN 2: Columnas eliminadas
                st.markdown("#### 2️⃣ Columnas Eliminadas")
                st.info(f"Se eliminaron las columnas: **{', '.join(['Name', 'Ticket', 'Cabin'])}**")
                remaining = results['dropped_columns']
                st.write(f"Columnas restantes ({len(remaining)}): {', '.join(remaining)}")
                
                # INSTRUCCIÓN 3: Valores nulos
                st.markdown("#### 3️⃣ Manejo de Valores Nulos")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Antes del procesamiento:**")
                    missing_before = pd.DataFrame.from_dict(
                        results['missing_values']['before'],
                        orient='index',
                        columns=['Valores Nulos']
                    )
                    st.dataframe(missing_before[missing_before['Valores Nulos'] > 0], use_container_width=True)
                
                with col2:
                    st.markdown("**Después del procesamiento:**")
                    missing_after = pd.DataFrame.from_dict(
                        results['missing_values']['after'],
                        orient='index',
                        columns=['Valores Nulos']
                    )
                    st.dataframe(missing_after, use_container_width=True)
                
                # INSTRUCCIÓN 4: Codificación
                st.markdown("#### 4️⃣ Codificación de Variables Categóricas")
                encoding_info = results['encoding']
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Sex' in encoding_info:
                        st.markdown("**Sex:**")
                        st.json(encoding_info['Sex'])
                
                with col2:
                    if 'Embarked' in encoding_info:
                        st.markdown("**Embarked:**")
                        st.json(encoding_info['Embarked'])
                
                # INSTRUCCIÓN 5: Estandarización
                st.markdown("#### 5️⃣ Estandarización de Variables Numéricas")
                scaled_features = results['scaled_features']
                st.info(f"Variables estandarizadas: **{', '.join(scaled_features)}**")
                
                # INSTRUCCIÓN 6: División de datos
                st.markdown("#### 6️⃣ División de Datos (70% - 30%)")
                split_info = results['split_info']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Conjunto de Entrenamiento (70%)**")
                    st.metric("X_train shape", str(split_info['X_train_shape']))
                    st.metric("y_train shape", str(split_info['y_train_shape']))
                
                with col2:
                    st.markdown("**🧪 Conjunto de Prueba (30%)**")
                    st.metric("X_test shape", str(split_info['X_test_shape']))
                    st.metric("y_test shape", str(split_info['y_test_shape']))
    
    # ===== TAB 3: RESULTADOS FINALES =====
    with tabs[2]:
        st.markdown("### 📋 Datos Procesados - Vista Final")
        
        if processor.df_processed is None:
            st.warning("⚠️ Primero debes ejecutar el procesamiento en la pestaña 'Procesamiento'")
        else:
            df_processed = processor.get_processed_data()
            
            st.markdown("#### 🔝 Primeros 5 registros procesados")
            st.dataframe(df_processed.head(), use_container_width=True)
            
            st.markdown("#### 📊 Estadísticas del Dataset Procesado")
            st.dataframe(df_processed.describe(), use_container_width=True)
            
            # Botón de descarga
            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar datos procesados (CSV)",
                data=csv,
                file_name="titanic_processed.csv",
                mime="text/csv"
            )