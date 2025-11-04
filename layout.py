# layout.py
import streamlit as st
from streamlit_option_menu import option_menu
from iris.interfazIris import show_iris

# === IMPORTACIONES ===
try:
    from titanic.interfazTitanic import show_titanic
    from student_performance.interfazStudent import show_student
    from iris.interfazIris import show_iris
except ImportError as e:
    st.error(f"❌ Error al importar módulos: {e}")
    st.stop()


def show_layout():
    
    st.set_page_config(
        page_title="ML Dataset Processor",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        /* Fondo principal con gradiente suave */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f5 100%);
        }
        
        /* Sidebar con colores más oscuros para mejor contraste */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2d5016 0%, #1a3d0a 100%);
            box-shadow: 2px 0 10px rgba(0,0,0,0.2);
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
        }
        
        /* Títulos con mejor jerarquía visual */
        h1 {
            color: #1a365d;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            padding-bottom: 20px;
            border-bottom: 3px solid #2d5016;
        }
        
        h2, h3 {
            color: #2b2d42;
            font-weight: 600;
        }
        
        h4 {
            color: #4a5568;
        }
        
        /* Tarjetas de descripción mejoradas */
        .descripcion {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            border-left: 4px solid #2d5016;
            margin: 15px 0;
        }
        
        /* Botones mejorados */
        .stButton > button {
            background: linear-gradient(135deg, #2d5016 0%, #1a3d0a 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            background: linear-gradient(135deg, #3a6b1d 0%, #2d5016 100%);
        }
        
        /* Mejora en las métricas */
        [data-testid="stMetricValue"] {
            color: #1a365d;
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Dataframes con mejor presentación */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        /* Scrollbar personalizado */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #2d5016;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #1a3d0a;
        }
        
        /* Animación de entrada */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .main > div {
            animation: fadeIn 0.5s ease-out;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h2 style='color: #ffffff; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>🤖 ML Datasets</h2>
                <p style='color: #b8e6a8; font-size: 14px;'>Explora y analiza</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Menú de opciones 
        dataset = option_menu(
            menu_title=None,
            options=["Titanic", "Student Performance", "Iris"],
            icons=["ship", "book", "flower2"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "5px",
                    "background-color": "transparent"
                },
                "icon": {
                    "color": "#ffffff",
                    "font-size": "20px"
                },
                "nav-link": {
                    "color": "#ffffff",  # Blanco para contraste
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px 0",
                    "padding": "12px 15px",
                    "border-radius": "8px",
                    "background-color": "#000000",  # Fondo más oscuro
                    "transition": "all 0.3s ease",
                    "font-weight": "500"
                },
                "nav-link-selected": {
                    "background-color": "#4a7c2c",  # Verde más brillante
                    "color": "#ffffff",
                    "font-weight": "700",
                    "box-shadow": "0 2px 8px rgba(0,0,0,0.3)"
                },
            },
        )
        
        # Footer del sidebar
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #b8e6a8; font-size: 12px;'>
                <p style='margin: 8px 0;'>💡 Selecciona un dataset</p>
                <p style='margin: 8px 0;'>🔍 Explora los datos</p>
                <p style='margin: 8px 0;'>📊 Visualiza resultados</p>
            </div>
        """, unsafe_allow_html=True)
    
    # === ENCABEZADO PRINCIPAL CON BADGE ===
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("🤖 Procesamiento de Datasets en Machine Learning")
    with col2:
        st.markdown("""
            <div style='text-align: center; padding-top: 10px;'>
                <span style='background: #4a7c2c; color: white; padding: 8px 16px; 
                      border-radius: 20px; font-weight: 600; font-size: 14px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.2);'> 
                    tatito
                </span>
            </div>
        """, unsafe_allow_html=True)

    # === CARGA DINÁMICA DEL CONTENIDO === 
    st.markdown("<br>", unsafe_allow_html=True)
    
    if dataset == "Titanic":
        show_titanic()
    elif dataset == "Student Performance":
        show_student()
    elif dataset == "Iris":
        show_iris()
    else:
        st.warning("⚠️ Seleccione un dataset del menú lateral.")


# === PUNTO DE ENTRADA ===
if __name__ == "__main__":
    show_layout()