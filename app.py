import streamlit as st

st.set_page_config(
    page_title="RestaurantAI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🍽️ RestaurantAI")
    st.subheader("Plataforma de Inteligencia de Clientes para Restaurantes")
    st.markdown("---")

    st.markdown(
        """
        Bienvenido a **RestaurantAI**. Transforma el feedback de tus comensales en
        **insights accionables** y **oportunidades de negocio** mediante inteligencia artificial.
        """
    )

    # ── Tarjetas de navegación ────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📊 Dashboard de Análisis")
        st.markdown(
            """
            Visualiza el análisis completo del feedback de tus clientes:
            - Scores de sentimiento por categoría (atención, comida, precio, ambiente)
            - Principales elogios y quejas detectados por IA
            - Platos más valorados por los comensales
            - Resumen ejecutivo y recomendaciones
            """
        )
        st.page_link("pages/1_Dashboard.py", label="Ir al Dashboard →", icon="📊")

    with col2:
        st.markdown("### 🎯 Generador de Leads")
        st.markdown(
            """
            Identifica y gestiona tus mejores oportunidades de negocio:
            - Scoring de clientes con potencial de retorno o fidelización
            - Categorización: alto valor, retención, recurrentes, referidores
            - Acciones de CRM y marketing sugeridas por IA
            - Exportación de leads para seguimiento
            """
        )
        st.page_link("pages/2_Leads.py", label="Ver Leads →", icon="🎯")

    # ── Información del dataset ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📂 Fuente de Datos")
    st.info(
        "Los análisis se generan a partir del archivo `Data/data.csv`, "
        "que contiene las respuestas de los comensales a 6 preguntas sobre su experiencia en el restaurante."
    )

    with st.expander("Ver preguntas del formulario de feedback"):
        st.markdown(
            """
            1. ¿Qué mejorarías de la atención?
            2. ¿Qué te pareció la atención?
            3. ¿Qué te gustó más de la comida?
            4. ¿Qué opina sobre la relación entre calidad y precio?
            5. ¿Qué te gustó más del ambiente?
            6. ¿Qué es lo que cambiarías de la experiencia?
            """
        )


if __name__ == "__main__":
    main()
