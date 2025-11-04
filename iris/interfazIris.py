# iris/interfazIris.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from iris.iris import IrisProcessor

def show_iris():
    """
    Interfaz de Streamlit para el dataset Iris
    SOLO muestra resultados llamando funciones del procesador
    """
    
    st.markdown("## Análisis del Dataset Iris")
    
    # Inicializar procesador en session state
    if 'iris_processor' not in st.session_state:
        st.session_state.iris_processor = IrisProcessor()
    
    processor = st.session_state.iris_processor
    
    # ===== TABS =====
    tabs = st.tabs([
        "📊 Dataset Original",
        "📈 Estadísticas",
        "🔧 Estandarización",
        "✂️ División de Datos",
        "📉 Visualizaciones",
        "🎨 Análisis Avanzado"
    ])
    
    # ==================== TAB 1: DATASET ORIGINAL ====================
    with tabs[0]:
        st.markdown("### 📥 Carga del Dataset Iris")
        st.markdown("*Dataset disponible en sklearn.datasets*")
        
        if st.button("🔄 Cargar Dataset Iris", type="primary", key="load_btn"):
            # LLAMAR función de carga
            result = processor.load_data()
            
            if result['success']:
                st.success(result['message'])
                
                # Mostrar métricas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📋 Total de Muestras", result['rows'])
                with col2:
                    st.metric("📊 Características", result['columns'])
                with col3:
                    st.metric("🏷️ Clases", len(result['classes']))
                with col4:
                    st.metric("⚖️ Muestras por Clase", "50")
                
                # Información de características
                st.markdown("#### 📋 Características del Dataset")
                st.info(f"**Características:** {', '.join(result['features'])}")
                st.info(f"**Clases:** {', '.join(result['classes'])}")
                
                # Distribución de clases
                st.markdown("#### 🏷️ Distribución de Clases")
                samples_per_class = result['samples_per_class']
                
                col1, col2, col3 = st.columns(3)
                for idx, (class_name, count) in enumerate(zip(result['classes'], samples_per_class.values())):
                    with [col1, col2, col3][idx]:
                        st.metric(class_name, count)
                
                # Mostrar datos originales
                df_original = processor.get_original_data()
                
                st.markdown("#### 📋 Vista previa del dataset")
                st.dataframe(df_original.head(15), use_container_width=True)
                
                st.markdown("#### 📊 Dataset completo")
                st.dataframe(df_original, use_container_width=True)
            else:
                st.error(result['message'])
    
    # ==================== TAB 2: ESTADÍSTICAS ====================
    with tabs[1]:
        st.markdown("### 📈 Estadísticas Descriptivas")
        
        if not processor.is_loaded():
            st.warning("⚠️ Primero debes cargar el dataset en la pestaña 'Dataset Original'")
        else:
            # LLAMAR función de estadísticas
            stats = processor.get_basic_statistics()
            
            if stats:
                st.markdown("#### 📊 Estadísticas Descriptivas")
                st.dataframe(stats['describe'], use_container_width=True)
                
                st.markdown("#### 🔗 Matriz de Correlación")
                
                # Heatmap de correlación
                fig = px.imshow(
                    stats['correlation'],
                    text_auto='.2f',
                    labels=dict(color="Correlación"),
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                fig.update_layout(
                    title='Correlación entre Características',
                    width=700,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretación de correlaciones
                st.markdown("#### 💡 Observaciones")
                corr = stats['correlation']
                
                # Encontrar la correlación más fuerte (excluyendo diagonal)
                corr_values = []
                for i in range(len(corr)):
                    for j in range(i+1, len(corr)):
                        corr_values.append({
                            'features': f"{corr.index[i]} - {corr.columns[j]}",
                            'value': corr.iloc[i, j]
                        })
                
                strongest = max(corr_values, key=lambda x: abs(x['value']))
                st.info(f"🔍 **Correlación más fuerte:** {strongest['features']} ({strongest['value']:.3f})")
    
    # ==================== TAB 3: ESTANDARIZACIÓN ====================
    with tabs[2]:
        st.markdown("### 🔧 Estandarización con StandardScaler")
        st.markdown("*Aplica transformación Z-score: (x - μ) / σ*")
        
        if not processor.is_loaded():
            st.warning("⚠️ Primero debes cargar el dataset en la pestaña 'Dataset Original'")
        else:
            if st.button("⚡ Aplicar Estandarización", type="primary", key="scale_btn"):
                # LLAMAR función de estandarización
                result = processor.apply_standardization()
                
                if result['success']:
                    st.success(result['message'])
                    
                    st.markdown("#### 📊 Comparación: Antes vs Después")
                    
                    # Mostrar para cada característica
                    for feature in result['features_scaled']:
                        with st.expander(f"📈 {feature}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Antes de Estandarizar:**")
                                st.write(f"Media: {result['stats_before']['mean'][feature]:.3f}")
                                st.write(f"Desv. Estándar: {result['stats_before']['std'][feature]:.3f}")
                                st.write(f"Mínimo: {result['stats_before']['min'][feature]:.3f}")
                                st.write(f"Máximo: {result['stats_before']['max'][feature]:.3f}")
                            
                            with col2:
                                st.markdown("**Después de Estandarizar:**")
                                st.write(f"Media: {result['stats_after']['mean'][feature]:.6f}")
                                st.write(f"Desv. Estándar: {result['stats_after']['std'][feature]:.6f}")
                                st.write(f"Mínimo: {result['stats_after']['min'][feature]:.3f}")
                                st.write(f"Máximo: {result['stats_after']['max'][feature]:.3f}")
                    
                    # Parámetros del scaler
                    st.markdown("#### ⚙️ Parámetros del StandardScaler")
                    scaler_params = pd.DataFrame({
                        'Feature': result['features_scaled'],
                        'Mean (μ)': result['scaler_params']['mean'],
                        'Scale (σ)': result['scaler_params']['scale']
                    })
                    st.dataframe(scaler_params, use_container_width=True)
                    
                    # Mostrar estadísticas del dataset escalado
                    st.markdown("#### 📋 Estadísticas Descriptivas del Dataset Estandarizado")
                    scaled_stats = processor.get_scaled_statistics()
                    st.dataframe(scaled_stats['describe'], use_container_width=True)
                else:
                    st.error(result['message'])
    
    # ==================== TAB 4: DIVISIÓN DE DATOS ====================
    with tabs[3]:
        st.markdown("### ✂️ División de Datos")
        st.markdown("*70% Entrenamiento - 30% Prueba*")
        
        if not processor.is_scaled():
            st.warning("⚠️ Primero debes aplicar estandarización en la pestaña 'Estandarización'")
        else:
            if st.button("✂️ Dividir Datos (70-30)", type="primary", key="split_btn"):
                # LLAMAR función de división
                result = processor.split_data(test_size=0.3, random_state=42)
                
                if result['success']:
                    st.success(result['message'])
                    
                    # Métricas principales
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🎯 Conjunto de Entrenamiento (70%)")
                        st.metric("X_train", str(result['X_train_shape']))
                        st.metric("y_train", str(result['y_train_shape']))
                        st.metric("Total Muestras", result['train_samples'])
                        
                        st.markdown("**Distribución de Clases:**")
                        for class_id, count in result['train_class_distribution'].items():
                            class_name = processor.get_target_names()[int(class_id)]
                            st.write(f"• {class_name}: {count} muestras")
                    
                    with col2:
                        st.markdown("#### 🧪 Conjunto de Prueba (30%)")
                        st.metric("X_test", str(result['X_test_shape']))
                        st.metric("y_test", str(result['y_test_shape']))
                        st.metric("Total Muestras", result['test_samples'])
                        
                        st.markdown("**Distribución de Clases:**")
                        for class_id, count in result['test_class_distribution'].items():
                            class_name = processor.get_target_names()[int(class_id)]
                            st.write(f"• {class_name}: {count} muestras")
                    
                    # Gráfico de división
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=['Entrenamiento (70%)', 'Prueba (30%)'],
                            values=[result['train_samples'], result['test_samples']],
                            marker_colors=['#3498db', '#e74c3c'],
                            hole=0.3
                        )
                    ])
                    fig.update_layout(title='División del Dataset')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(result['message'])
    
    # ==================== TAB 5: VISUALIZACIONES ====================
    with tabs[4]:
        st.markdown("### 📉 Visualizaciones del Dataset")
        
        if not processor.is_loaded():
            st.warning("⚠️ Primero debes cargar el dataset en la pestaña 'Dataset Original'")
        else:
            # GRÁFICO DE DISPERSIÓN
            st.markdown("#### 🎯 Gráfico de Dispersión: Sepal Length vs Petal Length")
            st.markdown("*Diferenciado por clase (target)*")
            
            # LLAMAR función para obtener datos
            scatter_data = processor.get_scatter_data(
                feature_x='sepal length (cm)',
                feature_y='petal length (cm)'
            )
            
            if scatter_data:
                # Crear DataFrame para plotly
                df_scatter = pd.DataFrame({
                    'Sepal Length': scatter_data['x'],
                    'Petal Length': scatter_data['y'],
                    'Species': scatter_data['species']
                })
                
                # Gráfico con Plotly
                fig = px.scatter(
                    df_scatter,
                    x='Sepal Length',
                    y='Petal Length',
                    color='Species',
                    title='Sepal Length vs Petal Length por Especie',
                    labels={
                        'Sepal Length': 'Longitud del Sépalo (cm)',
                        'Petal Length': 'Longitud del Pétalo (cm)'
                    },
                    color_discrete_sequence=['#e74c3c', '#3498db', '#2ecc71']
                )
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico con Matplotlib (alternativa)
                st.markdown("#### 📊 Gráfico Alternativo con Matplotlib")
                
                fig_mpl, ax = plt.subplots(figsize=(10, 6))
                
                colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71'}
                target_names = processor.get_target_names()
                
                for target_value in [0, 1, 2]:
                    mask = scatter_data['target'] == target_value
                    ax.scatter(
                        scatter_data['x'][mask],
                        scatter_data['y'][mask],
                        c=colors[target_value],
                        label=target_names[target_value],
                        s=100,
                        alpha=0.6,
                        edgecolors='white',
                        linewidth=1.5
                    )
                
                ax.set_xlabel('Longitud del Sépalo (cm)', fontsize=12)
                ax.set_ylabel('Longitud del Pétalo (cm)', fontsize=12)
                ax.set_title('Sepal Length vs Petal Length por Especie', fontsize=14, fontweight='bold')
                ax.legend(title='Especie', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig_mpl)
            
            # DISTRIBUCIONES POR CARACTERÍSTICA
            st.markdown("#### 📊 Distribución de Características por Especie")
            
            # LLAMAR función para obtener distribuciones
            distributions = processor.get_feature_distributions()
            
            if distributions:
                feature_names = processor.get_feature_names()
                
                # Crear gráfico de 4 subplots
                for feature in feature_names:
                    data = distributions[feature]
                    
                    df_dist = pd.DataFrame({
                        'Value': data['values'],
                        'Species': data['species']
                    })
                    
                    fig = px.histogram(
                        df_dist,
                        x='Value',
                        color='Species',
                        title=f'Distribución de {feature}',
                        labels={'Value': feature, 'count': 'Frecuencia'},
                        barmode='overlay',
                        opacity=0.7,
                        color_discrete_sequence=['#e74c3c', '#3498db', '#2ecc71']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 6: ANÁLISIS AVANZADO ====================
    with tabs[5]:
        st.markdown("### 🎨 Análisis Avanzado")
        
        if not processor.is_loaded():
            st.warning("⚠️ Primero debes cargar el dataset en la pestaña 'Dataset Original'")
        else:
            # PAIRPLOT
            st.markdown("#### 🔲 Pairplot: Relaciones entre Características")
            st.markdown("*Visualización de todas las combinaciones de características*")
            
            if st.button("🎨 Generar Pairplot", key="pairplot_btn"):
                with st.spinner("Generando pairplot..."):
                    # LLAMAR función para obtener datos
                    df_pairplot = processor.get_pairplot_data()
                    
                    if df_pairplot is not None:
                        # Crear pairplot con seaborn
                        fig = sns.pairplot(
                            df_pairplot,
                            hue='species',
                            palette={'setosa': '#e74c3c', 'versicolor': '#3498db', 'virginica': '#2ecc71'},
                            diag_kind='hist',
                            plot_kws={'alpha': 0.6, 'edgecolor': 'white'},
                            height=2.5
                        )
                        fig.fig.suptitle('Pairplot del Dataset Iris', y=1.01, fontsize=16, fontweight='bold')
                        st.pyplot(fig.fig)
                        
                        st.success("✅ Pairplot generado exitosamente")
                        
                        st.markdown("#### 💡 Interpretación del Pairplot")
                        st.info("""
                        - **Diagonal:** Histogramas de cada característica por especie
                        - **Fuera de diagonal:** Gráficos de dispersión entre pares de características
                        - **Setosa** se distingue claramente por su sépalo corto y ancho
                        - **Petal length y Petal width** son las características más discriminantes
                        """)
            
            st.markdown("---")
            
            # PCA 3D
            st.markdown("#### 🎲 Análisis de Componentes Principales (PCA)")
            st.markdown("*Reducción a 3 dimensiones principales*")
            
            if st.button("📊 Generar PCA 3D", key="pca_btn"):
                with st.spinner("Calculando PCA..."):
                    # LLAMAR función de PCA
                    pca_result = processor.get_pca_data(n_components=3)
                    
                    if pca_result['success']:
                        st.success("✅ PCA calculado exitosamente")
                        
                        # Información de varianza explicada
                        st.markdown("#### 📊 Varianza Explicada")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("PC1", f"{pca_result['explained_variance'][0]*100:.1f}%")
                        with col2:
                            st.metric("PC2", f"{pca_result['explained_variance'][1]*100:.1f}%")
                        with col3:
                            st.metric("PC3", f"{pca_result['explained_variance'][2]*100:.1f}%")
                        with col4:
                            st.metric("Total", f"{pca_result['total_variance']*100:.1f}%")
                        
                        # Gráfico 3D con Plotly
                        st.markdown("#### 🎲 Visualización PCA en 3D")
                        
                        X_pca = pca_result['X_pca']
                        species = pca_result['species']
                        
                        df_pca = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'PC3': X_pca[:, 2],
                            'Species': species
                        })
                        
                        fig = px.scatter_3d(
                            df_pca,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            color='Species',
                            title='Primeras 3 Componentes Principales del Dataset Iris',
                            labels={
                                'PC1': '1er Componente Principal',
                                'PC2': '2do Componente Principal',
                                'PC3': '3er Componente Principal'
                            },
                            color_discrete_sequence=['#e74c3c', '#3498db', '#2ecc71']
                        )
                        
                        fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='white')))
                        fig.update_layout(height=700)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("#### 💡 Interpretación del PCA")
                        st.info(f"""
                        - Las 3 componentes principales explican el **{pca_result['total_variance']*100:.1f}%** de la varianza total
                        - Se observa una clara separación entre las especies en el espacio reducido
                        - **Setosa** es la más fácil de distinguir
                        - **Versicolor** y **Virginica** tienen cierta superposición pero son distinguibles
                        """)