import streamlit as st
import plotly.graph_objects as go
import logging

from Backend.Dashboard.dashboard_agent import run_dashboard_agent_from_df
from Backend.Dashboard.tag_analytics import reset_incremental_tag_storage
from Frontend.utils.data_loader import get_data_summary, load_data

st.set_page_config(
    page_title="Dashboard – RestaurantAI",
    page_icon="📊",
    layout="wide",
)

logger = logging.getLogger(__name__)

SCORE_LABELS = {
    "atencion": "Atención",
    "comida": "Comida",
    "precio_calidad": "Calidad / Precio",
    "ambiente": "Ambiente",
    "experiencia_general": "Experiencia General",
}

DEFAULT_HOVER_COMMENT = "Sin comentarios"


@st.cache_data(show_spinner="Analizando feedback con IA…")
def get_analysis(df) -> dict:
    return run_dashboard_agent_from_df(df)


def render_score_bar(sentiment_scores: dict) -> go.Figure:
    labels, values, colors = [], [], []
    for key, label in SCORE_LABELS.items():
        if key in sentiment_scores:
            v = sentiment_scores[key]
            labels.append(label)
            values.append(v)
            if v >= 7:
                color = "#4CAF50"
            elif v >= 5:
                color = "#FF9800"
            else:
                color = "#F44336"
            colors.append(color)

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
        yaxis={"range": [0, 10], "title": "Score"},
        xaxis_title="Categoría",
        height=380,
        margin={"t": 50, "b": 30},
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
        margin={"t": 50, "b": 30},
    )
    return fig


def render_tags_by_polarity(tag_insights: list[dict]) -> go.Figure:
    sorted_tags = sorted(
        tag_insights,
        key=lambda x: x.get("total_mentions", x.get("bien", 0) + x.get("neutral", 0) + x.get("mal", 0)),
        reverse=True,
    )[:5]

    tags = [item.get("tag", "") for item in sorted_tags]
    bien = [item.get("bien", 0) for item in sorted_tags]
    neutral = [item.get("neutral", 0) for item in sorted_tags]
    mal = [item.get("mal", 0) for item in sorted_tags]
    hover_bien = [item.get("hover_bien", DEFAULT_HOVER_COMMENT) for item in sorted_tags]
    hover_neutral = [item.get("hover_neutral", DEFAULT_HOVER_COMMENT) for item in sorted_tags]
    hover_mal = [item.get("hover_mal", DEFAULT_HOVER_COMMENT) for item in sorted_tags]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=tags,
            x=bien,
            name="Bien",
            orientation="h",
            marker_color="#4CAF50",
            customdata=hover_bien,
            hovertemplate="<b>%{y}</b><br>Clientes (bien): %{x}<br><br>%{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=tags,
            x=neutral,
            name="Neutral",
            orientation="h",
            marker_color="#FF9800",
            customdata=hover_neutral,
            hovertemplate="<b>%{y}</b><br>Clientes (neutral): %{x}<br><br>%{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=tags,
            x=mal,
            name="Mal",
            orientation="h",
            marker_color="#F44336",
            customdata=hover_mal,
            hovertemplate="<b>%{y}</b><br>Clientes (mal): %{x}<br><br>%{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Top 5 Tags por Polaridad (clientes únicos)",
        barmode="group",
        xaxis_title="Clientes únicos",
        yaxis_title="Tag",
        height=460,
        margin={"t": 50, "b": 30},
        legend={"orientation": "h", "y": 1.08},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def render_tag_balance(tag_insights: list[dict], balance_mode: str) -> go.Figure:
    if balance_mode == "Top Buenos":
        filtered = [item for item in tag_insights if item.get("balance", 0) > 0]
        sorted_tags = sorted(filtered, key=lambda x: x.get("balance", 0), reverse=True)[:5]
        title = "Balance por Tag · Top Buenos"
        hover_comment_title = "Comentarios (bien)"
        hover_comment_key = "hover_bien"
    else:
        filtered = [item for item in tag_insights if item.get("balance", 0) < 0]
        sorted_tags = sorted(filtered, key=lambda x: x.get("balance", 0))[:5]
        title = "Balance por Tag · Top Malos"
        hover_comment_title = "Comentarios (mal)"
        hover_comment_key = "hover_mal"

    if not sorted_tags:
        sorted_tags = []
        title = "Balance por Tag"

    tags = [item.get("tag", "") for item in sorted_tags]
    balances = [item.get("balance", 0) for item in sorted_tags]
    colors = ["#F44336" if value < 0 else "#4CAF50" for value in balances]
    hover_data = [
        f"Bien: {item.get('bien', 0)}<br>Neutral: {item.get('neutral', 0)}<br>Mal: {item.get('mal', 0)}"
        f"<br><br>{hover_comment_title}:<br>{item.get(hover_comment_key, DEFAULT_HOVER_COMMENT)}"
        for item in sorted_tags
    ]

    fig = go.Figure(
        go.Bar(
            x=tags,
            y=balances,
            marker_color=colors,
            customdata=hover_data,
            hovertemplate="<b>%{x}</b><br>Balance: %{y}<br><br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tag",
        yaxis_title="Balance de clientes",
        height=460,
        margin={"t": 50, "b": 30},
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#9E9E9E")
    return fig


def render_sentiment_section(sentiment_scores: dict) -> None:
    if not sentiment_scores:
        return

    col_bar, col_pie = st.columns([3, 2])
    with col_bar:
        st.plotly_chart(render_score_bar(sentiment_scores), use_container_width=True)
    with col_pie:
        st.plotly_chart(render_distribution_donut(sentiment_scores), use_container_width=True)


def render_tag_insights_section(tag_insights: list[dict]) -> None:
    if not tag_insights:
        return

    st.markdown("### 🏷️ Tags Accionables")
    balance_mode = st.radio(
        "Vista de balance",
        ["Top Malos", "Top Buenos"],
        horizontal=True,
    )
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(render_tags_by_polarity(tag_insights), use_container_width=True)
    with t2:
        st.plotly_chart(render_tag_balance(tag_insights, balance_mode), use_container_width=True)


def render_metadata(metadata: dict) -> list[dict]:
    if not metadata:
        return []

    st.caption(
        f"Procesamiento incremental → nuevas filas: {metadata.get('new_rows_processed', 0)} · "
        f"nuevos eventos de tags: {metadata.get('new_tag_events', 0)} · "
        f"eventos acumulados: {metadata.get('events', 0)}"
    )

    if metadata.get("executed_at"):
        st.caption(f"Último recálculo (UTC): {metadata.get('executed_at')}")

    logger.info(
        "[DashboardPage] Metadata | nuevas_filas=%s | nuevos_eventos=%s | total_eventos=%s | executed_at=%s",
        metadata.get("new_rows_processed", 0),
        metadata.get("new_tag_events", 0),
        metadata.get("events", 0),
        metadata.get("executed_at", ""),
    )

    return metadata.get("tag_insights", [])


def render_executive_summary(summary: dict) -> None:
    if not summary:
        return

    st.markdown("---")
    st.markdown("### 📋 Resumen Ejecutivo")
    rc1, rc2, rc3 = st.columns(3)

    fortalezas = summary.get("fortalezas", [])
    debilidades = summary.get("debilidades", [])
    plan_mejora = summary.get("plan_mejora", [])

    fortalezas_text = "\n".join(f"- {item}" for item in fortalezas) if fortalezas else "Sin fortalezas destacadas"
    debilidades_text = "\n".join(f"- {item}" for item in debilidades) if debilidades else "Sin debilidades destacadas"
    plan_text = "\n".join(f"- {item}" for item in plan_mejora) if plan_mejora else "Sin plan sugerido"

    rc1.info(
        f"**Resumen**\n\n{summary.get('resumen', '')}\n\n"
        f"**Fortalezas (Top 3)**\n{fortalezas_text}"
    )
    rc2.warning(f"**Debilidades (Top 3)**\n\n{debilidades_text}")
    rc3.success(
        f"**Plan de Mejora (Top 3 acciones)**\n\n{plan_text}\n\n"
        f"**Recomendación Principal**\n{summary.get('recomendacion_principal', '')}"
    )


def main():
    st.title("📊 Dashboard de Análisis")
    st.caption("Análisis inteligente del feedback de tus comensales · Powered by Gemini + LangGraph")

    logger.info("[DashboardPage] Render de página iniciado")

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

    # ── Botones de actualización / reproceso ──────────────────────────────────
    btn_refresh, btn_reprocess = st.columns(2)

    if btn_refresh.button("🔄 Actualizar análisis", type="primary"):
        logger.info("[DashboardPage] Botón Actualizar presionado -> limpiando cache y relanzando")
        st.cache_data.clear()
        st.rerun()

    if btn_reprocess.button("♻️ Reprocesar todo"):
        logger.warning("[DashboardPage] Botón Reprocesar todo presionado")
        reset_result = reset_incremental_tag_storage()
        logger.warning("[DashboardPage] Reprocesar todo -> resultado=%s", reset_result)
        st.cache_data.clear()
        st.rerun()

    # ── Análisis IA ────────────────────────────────────────────────────────────
    result = get_analysis(df)
    logger.info("[DashboardPage] Resultado obtenido | error=%s", bool(result.get("error")))

    if result.get("error"):
        st.error(f"Error durante el análisis: {result['error']}")
        return

    metadata = result.get("metadata", {})
    tag_insights = render_metadata(metadata)

    # ── Gráficos de sentimiento ────────────────────────────────────────────────
    sentiment_scores = result.get("sentiment_scores", {})
    render_sentiment_section(sentiment_scores)
    render_tag_insights_section(tag_insights)

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
    render_executive_summary(summary)

    # ── Datos crudos ───────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📂 Ver datos originales"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
