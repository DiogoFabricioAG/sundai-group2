import streamlit as st
import plotly.graph_objects as go

from Backend.Dashboard.dashboard_agent import run_dashboard_agent
from Frontend.utils.data_loader import df_to_text, get_data_summary, load_data

st.set_page_config(
    page_title="Dashboard – RestaurantAI",
    page_icon="📊",
    layout="wide",
)

SCORE_LABELS = {
    "atencion": "Atención",
    "comida": "Comida",
    "precio_calidad": "Calidad / Precio",
    "ambiente": "Ambiente",
    "experiencia_general": "Experiencia General",
}


@st.cache_data(show_spinner="Analizando feedback con IA…")
def get_analysis(data_text: str) -> dict:
    return run_dashboard_agent(data_text)


def render_score_bar(sentiment_scores: dict) -> go.Figure:
    labels, values, colors = [], [], []
    for key, label in SCORE_LABELS.items():
        if key in sentiment_scores:
            v = sentiment_scores[key]
            labels.append(label)
            values.append(v)
            colors.append(
                "#4CAF50" if v >= 7 else "#FF9800" if v >= 5 else "#F44336"
            )

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Scores de Satisfacción por Categoría (0 – 10)",
        yaxis=dict(range=[0, 10], title="Score"),
        xaxis_title="Categoría",
        height=380,
        margin=dict(t=50, b=30),
    )
    return fig


def render_distribution_donut(sentiment_scores: dict) -> go.Figure:
    labels = ["Positivos", "Neutros", "Negativos"]
    values = [
        sentiment_scores.get("positivos", 0),
        sentiment_scores.get("neutros", 0),
        sentiment_scores.get("negativos", 0),
    ]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker_colors=["#4CAF50", "#FF9800", "#F44336"],
        )
    )
    fig.update_layout(
        title="Distribución de Experiencias",
        height=380,
        margin=dict(t=50, b=30),
    )
    return fig


def main():
    st.title("📊 Dashboard de Análisis")
    st.caption("Análisis inteligente del feedback de tus comensales · Powered by Gemini + LangGraph")

    # ── Carga de datos ─────────────────────────────────────────────────────────
    try:
        df = load_data()
    except Exception as e:
        st.error(f"No se pudo cargar el archivo de datos: {e}")
        return

    stats = get_data_summary(df)

    # ── KPIs ───────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Comensales", stats["total_customers"])
    c2.metric("Consumo Promedio", f"S/. {stats['avg_consumption']:.2f}")
    c3.metric("Consumo Máximo", f"S/. {stats['max_consumption']:.2f}")
    c4.metric("Consumo Mínimo", f"S/. {stats['min_consumption']:.2f}")
    c5.metric("Ingresos Totales", f"S/. {stats['total_revenue']:.2f}")

    st.markdown("---")

    # ── Botón de actualización ─────────────────────────────────────────────────
    if st.button("🔄 Actualizar análisis", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # ── Análisis IA ────────────────────────────────────────────────────────────
    data_text = df_to_text(df)
    result = get_analysis(data_text)

    if result.get("error"):
        st.error(f"Error durante el análisis: {result['error']}")
        return

    # ── Gráficos de sentimiento ────────────────────────────────────────────────
    sentiment_scores = result.get("sentiment_scores", {})
    if sentiment_scores:
        col_bar, col_pie = st.columns([3, 2])
        with col_bar:
            st.plotly_chart(render_score_bar(sentiment_scores), use_container_width=True)
        with col_pie:
            st.plotly_chart(render_distribution_donut(sentiment_scores), use_container_width=True)

    # ── Temas principales ──────────────────────────────────────────────────────
    key_themes = result.get("key_themes", {})
    if key_themes:
        st.markdown("### 🔍 Temas Principales")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 👍 Elogios destacados")
            for item in key_themes.get("top_praises", []):
                st.markdown(f"- {item}")

            st.markdown("#### 🍽️ Platos más valorados")
            for item in key_themes.get("top_dishes", []):
                st.markdown(f"- {item}")

        with col_right:
            st.markdown("#### 👎 Principales quejas")
            for item in key_themes.get("top_complaints", []):
                st.markdown(f"- {item}")

            st.markdown("#### 🔧 Áreas de mejora")
            for item in key_themes.get("improvement_areas", []):
                st.markdown(f"- {item}")

    # ── Resumen ejecutivo ──────────────────────────────────────────────────────
    summary = result.get("summary", {})
    if summary:
        st.markdown("---")
        st.markdown("### 📋 Resumen Ejecutivo")
        rc1, rc2, rc3 = st.columns(3)
        rc1.info(f"**Resumen**\n\n{summary.get('resumen', '')}")
        rc2.success(f"**Fortaleza Principal**\n\n{summary.get('fortaleza_principal', '')}")
        rc3.warning(f"**Recomendación Urgente**\n\n{summary.get('recomendacion_principal', '')}")

    # ── Datos crudos ───────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📂 Ver datos originales"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
