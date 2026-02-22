import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from Backend.Leads.leads_agent import create_leads_agent
from Frontend.utils.data_loader import (
    df_to_text,
    get_customer_contact_data,
    get_data_summary,
    load_data,
)

st.set_page_config(
    page_title="Leads – RestaurantAI",
    page_icon="🎯",
    layout="wide",
)

CATEGORIA_META = {
    "alto_valor": {"label": "Alto Valor",  "emoji": "🟡"},
    "retencion":  {"label": "Retención",   "emoji": "🔴"},
    "recurrente": {"label": "Recurrente",  "emoji": "🟢"},
    "referidor":  {"label": "Referidor",   "emoji": "🔵"},
}


# ── Agent (cached globally, checkpointer persists in server memory) ────────────
@st.cache_resource
def get_agent():
    return create_leads_agent()


# ── Session state helpers ──────────────────────────────────────────────────────
def init_session():
    defaults = {
        "leads_phase": "idle",        # idle | awaiting_review | done
        "leads_thread_id": str(uuid.uuid4()),
        "leads_promotions": [],
        "leads_approved": [],
        "leads_error": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_flow():
    """Reinicia el flujo generando un nuevo thread para el agente."""
    st.session_state.leads_phase = "idle"
    st.session_state.leads_thread_id = str(uuid.uuid4())
    st.session_state.leads_promotions = []
    st.session_state.leads_approved = []
    st.session_state.leads_error = ""


# ── Helpers de UI ──────────────────────────────────────────────────────────────
def score_badge(score: int) -> str:
    if score >= 9:
        return f"🔥 {score}/10"
    if score >= 7:
        return f"⭐ {score}/10"
    return f"{score}/10"


def categoria_chip(cat: str) -> str:
    meta = CATEGORIA_META.get(cat, {"emoji": "⚪", "label": cat})
    return f"{meta['emoji']} {meta['label']}"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_session()

    st.title("🎯 Generador de Leads")
    st.caption(
        "Filtra clientes por gasto, genera promociones con IA "
        "y apruébalas antes de enviar · Powered by Gemini + LangGraph"
    )

    # ── Carga de datos ─────────────────────────────────────────────────────────
    try:
        df = load_data()
    except Exception as e:
        st.error(f"No se pudo cargar el archivo de datos: {e}")
        return

    stats = get_data_summary(df)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración")

        st.markdown("**Estadísticas de gasto del dataset:**")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Mín.", f"S/. {stats['min_consumption']:.0f}")
        sc2.metric("Prom.", f"S/. {stats['avg_consumption']:.0f}")
        sc3.metric("Máx.", f"S/. {stats['max_consumption']:.0f}")

        spending_threshold = st.slider(
            "Gasto mínimo para calificar (S/.)",
            min_value=0.0,
            max_value=float(stats["max_consumption"]),
            value=round(float(stats["avg_consumption"]) * 0.6, 0),
            step=5.0,
            help=(
                "Solo se generarán leads para clientes que hayan gastado "
                "por encima de este monto (filtro cuantitativo)"
            ),
        )

        qualifying = int((df["costo_del_consumo"] >= spending_threshold).sum())
        st.info(
            f"Clientes que califican: **{qualifying} de {len(df)}**\n\n"
            f"_(gasto >= S/. {spending_threshold:.2f})_"
        )

        st.markdown("---")

        # Diagrama del flujo del agente
        st.markdown("**Flujo del agente:**")
        st.markdown(
            """
            ```
            filter_and_score
                  ↓
            generate_promotions
                  ↓
            human_review  ← tú estás aquí
                  ↓
                 END
            ```
            """
        )

    phase = st.session_state.leads_phase

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1 — IDLE
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "idle":
        st.info(
            "Ajusta el umbral de gasto en el panel izquierdo "
            "y pulsa **Generar Leads** para comenzar."
        )

        if st.button("🚀 Generar Leads", type="primary", use_container_width=True):
            reset_flow()  # nuevo thread_id cada vez que se genera
            agent = get_agent()
            config = {"configurable": {"thread_id": st.session_state.leads_thread_id}}

            initial_state = {
                "raw_data": df_to_text(df),
                "customer_data": get_customer_contact_data(df),
                "spending_threshold": spending_threshold,
                "scored_leads": [],
                "promotions": [],
                "approved_leads": [],
                "error": "",
            }

            with st.spinner(
                "Filtrando clientes, evaluando leads y generando promociones con IA…"
            ):
                result = agent.invoke(initial_state, config=config)

            if result.get("error"):
                st.error(f"Error: {result['error']}")
                return

            # Verificar si el agente está esperando revisión humana
            snapshot = agent.get_state(config)
            if snapshot.next:  # interrupted en human_review
                st.session_state.leads_promotions = result.get("promotions", [])
                st.session_state.leads_phase = "awaiting_review"
                st.rerun()
            else:
                st.error("El agente finalizó sin solicitar revisión. Intenta de nuevo.")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — HUMAN IN THE LOOP: revisión y edición
    # ══════════════════════════════════════════════════════════════════════════
    elif phase == "awaiting_review":
        promotions = st.session_state.leads_promotions

        if not promotions:
            st.warning("No se generaron leads con el umbral de gasto seleccionado.")
            if st.button("↩️ Volver"):
                reset_flow()
                st.rerun()
            return

        st.success(
            f"**{len(promotions)} leads generados.** "
            "Revisa, edita si necesitas y aprueba las promociones antes de enviar."
        )

        col_back, col_info = st.columns([1, 4])
        with col_back:
            if st.button("↩️ Volver a generar"):
                reset_flow()
                st.rerun()
        with col_info:
            st.caption(
                "Marca el checkbox para aprobar el envío. "
                "Edita el mensaje en el área de texto si quieres personalizar."
            )

        st.markdown("---")

        # ── Formulario de revisión ─────────────────────────────────────────────
        approve_flags: dict[int, bool] = {}
        edited_msgs: dict[int, str] = {}

        for promo in promotions:
            id_c = promo["id_cliente"]
            cat = promo.get("categoria", "")
            meta = CATEGORIA_META.get(cat, {"emoji": "⚪", "label": cat})

            with st.container(border=True):
                # Cabecera
                h_col, chk_col = st.columns([5, 1])
                with h_col:
                    st.markdown(
                        f"#### {meta['emoji']} Cliente #{id_c} &nbsp;|&nbsp; "
                        f"Score: {score_badge(promo['score'])} &nbsp;|&nbsp; "
                        f"{meta['label']}"
                    )
                with chk_col:
                    approve_flags[id_c] = st.checkbox(
                        "Aprobar",
                        value=True,
                        key=f"chk_{id_c}",
                    )

                # Info del cliente
                i1, i2, i3 = st.columns(3)
                i1.markdown(f"📞 **Tel:** `{promo.get('telefono', '—')}`")
                i2.markdown(f"💰 **Consumo:** S/. {promo.get('consumo', 0):.2f}")
                i3.markdown(f"💡 **Motivo:** {promo.get('motivo', '—')}")

                # Mensaje editable
                edited_msgs[id_c] = st.text_area(
                    "✏️ Mensaje promocional (editable)",
                    value=promo.get("mensaje_promo", ""),
                    height=110,
                    key=f"msg_{id_c}",
                )

        # ── Barra de confirmación ──────────────────────────────────────────────
        st.markdown("---")
        n_approved = sum(1 for v in approve_flags.values() if v)

        cola, colb = st.columns([2, 1])
        cola.markdown(f"**{n_approved} de {len(promotions)} leads seleccionados para enviar.**")

        if colb.button(
            "✅ Confirmar y enviar leads aprobados",
            type="primary",
            use_container_width=True,
            disabled=(n_approved == 0),
        ):
            # Construir lista de leads aprobados con mensajes (posiblemente editados)
            approved_leads = [
                {**promo, "mensaje_promo": edited_msgs.get(promo["id_cliente"], promo["mensaje_promo"])}
                for promo in promotions
                if approve_flags.get(promo["id_cliente"], False)
            ]

            agent = get_agent()
            config = {"configurable": {"thread_id": st.session_state.leads_thread_id}}

            with st.spinner("Confirmando leads aprobados…"):
                result = agent.invoke(Command(resume=approved_leads), config=config)

            st.session_state.leads_approved = result.get("approved_leads", approved_leads)
            st.session_state.leads_phase = "done"
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3 — DONE: leads aprobados listos para enviar
    # ══════════════════════════════════════════════════════════════════════════
    elif phase == "done":
        approved = st.session_state.leads_approved

        st.success(f"✅ **{len(approved)} leads aprobados** y listos para enviar.")

        if st.button("🔄 Generar nuevos leads", use_container_width=False):
            reset_flow()
            st.rerun()

        st.markdown("---")

        # KPIs rápidos
        if approved:
            k1, k2, k3 = st.columns(3)
            k1.metric("Leads aprobados", len(approved))
            k2.metric(
                "Score promedio",
                f"{sum(l['score'] for l in approved) / len(approved):.1f}",
            )
            k3.metric(
                "Valor total potencial",
                f"S/. {sum(l.get('consumo', 0) for l in approved):.2f}",
            )
            st.markdown("---")

        for lead in approved:
            cat = lead.get("categoria", "")
            meta = CATEGORIA_META.get(cat, {"emoji": "⚪", "label": cat})

            with st.container(border=True):
                st.markdown(
                    f"#### {meta['emoji']} Cliente #{lead['id_cliente']} &nbsp;|&nbsp; "
                    f"Score: {score_badge(lead['score'])} &nbsp;|&nbsp; {meta['label']}"
                )
                c1, c2 = st.columns(2)
                c1.markdown(f"📞 **Teléfono:** `{lead.get('telefono', '—')}`")
                c2.markdown(f"💰 **Consumo:** S/. {lead.get('consumo', 0):.2f}")

                st.markdown("**Mensaje a enviar:**")
                st.info(lead.get("mensaje_promo", "—"))

        # Exportar
        if approved:
            st.markdown("---")
            export_df = pd.DataFrame(approved)
            cols = [
                c for c in
                ["id_cliente", "telefono", "consumo", "score", "categoria", "motivo", "mensaje_promo"]
                if c in export_df.columns
            ]
            st.download_button(
                label="⬇️ Exportar leads aprobados (CSV)",
                data=export_df[cols].to_csv(index=False).encode("utf-8"),
                file_name="leads_aprobados.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
