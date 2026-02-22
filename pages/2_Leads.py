import pandas as pd
import streamlit as st

from Backend.Leads.leads_agent import run_leads_agent
from Frontend.utils.data_loader import (
    get_customer_contact_data,
    df_to_text,
    load_data,
)

st.set_page_config(
    page_title="Leads – RestaurantAI",
    page_icon="🎯",
    layout="wide",
)

CATEGORIA_META = {
    "alto_valor":  {"label": "Alto Valor",  "emoji": "🟡", "color": "#FFC107"},
    "retencion":   {"label": "Retención",   "emoji": "🔴", "color": "#F44336"},
    "recurrente":  {"label": "Recurrente",  "emoji": "🟢", "color": "#4CAF50"},
    "referidor":   {"label": "Referidor",   "emoji": "🔵", "color": "#2196F3"},
}


@st.cache_data(show_spinner="Identificando leads con IA…")
def get_leads(data_text: str, customer_data: str) -> dict:
    return run_leads_agent(data_text, customer_data)


def score_badge(score: int) -> str:
    if score >= 9:
        return f"🔥 {score}/10"
    if score >= 7:
        return f"⭐ {score}/10"
    return f"{score}/10"


def main():
    st.title("🎯 Generador de Leads")
    st.caption("Identifica y gestiona tus mejores oportunidades de negocio · Powered by Gemini + LangGraph")

    # ── Carga de datos ─────────────────────────────────────────────────────────
    try:
        df = load_data()
    except Exception as e:
        st.error(f"No se pudo cargar el archivo de datos: {e}")
        return

    # ── Sidebar – filtros ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filtros")
        min_score = st.slider("Score mínimo", min_value=1, max_value=10, value=6)
        categorias_sel = st.multiselect(
            "Categorías",
            options=list(CATEGORIA_META.keys()),
            default=list(CATEGORIA_META.keys()),
            format_func=lambda k: f"{CATEGORIA_META[k]['emoji']} {CATEGORIA_META[k]['label']}",
        )
        if st.button("🔄 Actualizar leads"):
            st.cache_data.clear()
            st.rerun()

    # ── Análisis IA ────────────────────────────────────────────────────────────
    data_text = df_to_text(df)
    customer_data = get_customer_contact_data(df)
    result = get_leads(data_text, customer_data)

    if result.get("error"):
        st.error(f"Error al generar leads: {result['error']}")
        return

    leads_raw = result.get("scored_leads", [])
    if not leads_raw:
        st.warning("No se encontraron leads. Intenta actualizar el análisis.")
        return

    # ── Enriquecimiento con datos de contacto ──────────────────────────────────
    leads_df = pd.DataFrame(leads_raw)

    contact_df = df[["ID_Cliente", "numero_tel_cliente", "costo_del_consumo"]].rename(
        columns={
            "ID_Cliente": "id_cliente",
            "numero_tel_cliente": "telefono",
            "costo_del_consumo": "consumo",
        }
    )

    if "id_cliente" in leads_df.columns:
        leads_df = leads_df.merge(contact_df, on="id_cliente", how="left")

    # ── Aplicar filtros ────────────────────────────────────────────────────────
    filtered = leads_df[leads_df["score"] >= min_score].copy()
    if categorias_sel:
        filtered = filtered[filtered["categoria"].isin(categorias_sel)]

    # ── KPIs ───────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Leads", len(filtered))
    k2.metric(
        "Score Promedio",
        f"{filtered['score'].mean():.1f}" if len(filtered) else "—",
    )
    if "consumo" in filtered.columns and len(filtered):
        k3.metric("Consumo Prom. Leads", f"S/. {filtered['consumo'].mean():.2f}")
        k4.metric("Valor Total Potencial", f"S/. {filtered['consumo'].sum():.2f}")

    st.markdown("---")

    # ── Distribución por categoría ─────────────────────────────────────────────
    if len(filtered):
        st.markdown("### 📊 Distribución por Categoría")
        cat_counts = filtered["categoria"].value_counts()
        cat_cols = st.columns(max(len(cat_counts), 1))
        for i, (cat, count) in enumerate(cat_counts.items()):
            meta = CATEGORIA_META.get(cat, {"emoji": "⚪", "label": cat})
            cat_cols[i].metric(f"{meta['emoji']} {meta['label']}", count)
        st.markdown("---")

    # ── Tabla / Cards de leads ─────────────────────────────────────────────────
    st.markdown("### 👥 Lista de Leads")

    if len(filtered) == 0:
        st.info("No hay leads con los filtros seleccionados.")
        return

    for _, lead in filtered.sort_values("score", ascending=False).iterrows():
        cat = lead.get("categoria", "")
        meta = CATEGORIA_META.get(cat, {"emoji": "⚪", "label": cat})
        with st.expander(
            f"{meta['emoji']} Cliente #{int(lead.get('id_cliente', 0))}  —  "
            f"Score: {score_badge(int(lead.get('score', 0)))}  —  {meta['label']}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Teléfono:** {lead.get('telefono', '—')}")
            c2.markdown(f"**Consumo:** S/. {float(lead.get('consumo', 0)):.2f}")
            c3.markdown(f"**Categoría:** {meta['emoji']} {meta['label']}")

            st.markdown(f"**Motivo:** {lead.get('motivo', '—')}")
            st.info(f"💡 **Acción Sugerida:** {lead.get('accion_sugerida', '—')}")

    # ── Exportar ───────────────────────────────────────────────────────────────
    st.markdown("---")
    export_cols = [c for c in ["id_cliente", "telefono", "consumo", "score", "categoria", "motivo", "accion_sugerida"] if c in filtered.columns]
    csv_bytes = filtered[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Exportar Leads (CSV)",
        data=csv_bytes,
        file_name="leads_restaurante.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
