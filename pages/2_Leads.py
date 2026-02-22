import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from Backend.Leads.leads_agent import create_leads_agent, regenerate_single_promotion
from Frontend.utils.data_loader import (
    df_to_text,
    get_customer_contact_data,
    get_data_summary,
    get_feedback_blocks,
    load_data,
)

st.set_page_config(
    page_title="Leads – RestaurantAI",
    page_icon="🎯",
    layout="wide",
)

CATEGORIA_META = {
    "alto_valor": {"label": "Alto Valor", "emoji": "🟡"},
    "retencion":  {"label": "Retención",  "emoji": "🔴"},
    "recurrente": {"label": "Recurrente", "emoji": "🟢"},
    "referidor":  {"label": "Referidor",  "emoji": "🔵"},
}


# ── Agent (cached globally, el checkpointer persiste en el servidor) ───────────
@st.cache_resource
def get_agent():
    return create_leads_agent()


# ── Session state ──────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "leads_phase": "idle",        # idle | awaiting_review | done
        "leads_thread_id": str(uuid.uuid4()),
        "leads_promotions": [],
        "leads_approved": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_flow():
    """Reinicia el flujo completo con un nuevo thread."""
    st.session_state.leads_phase = "idle"
    st.session_state.leads_thread_id = str(uuid.uuid4())
    st.session_state.leads_promotions = []
    st.session_state.leads_approved = []
    # Limpiar claves de mensajes e instrucciones previas
    for key in list(st.session_state.keys()):
        if key.startswith(("msg_", "instr_", "chk_", "regen_", "pending_msg_", "clear_instr_")):
            del st.session_state[key]


def categoria_label(cat: str) -> str:
    meta = CATEGORIA_META.get(cat, {"emoji": "⚪", "label": cat})
    return f"{meta['emoji']} {meta['label']}"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_session()

    st.title("🎯 Generador de Leads")
    st.caption(
        "Filtra clientes por gasto, categorízalos con IA y revisa "
        "cada promoción antes de enviar · Powered by Gemini + LangGraph"
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

        spending_threshold = st.slider(
            "Gasto mínimo para calificar (S/.)",
            min_value=0.0,
            max_value=float(stats["max_consumption"]),
            value=round(float(stats["avg_consumption"]) * 0.6, 0),
            step=5.0,
            help="Filtro cuantitativo: solo pasan clientes que gastaron más de este monto",
        )

        qualifying = int((df["costo_del_consumo"] >= spending_threshold).sum())
        st.info(
            f"Clientes que califican: **{qualifying} de {len(df)}**\n\n"
            f"_(gasto >= S/. {spending_threshold:.2f})_"
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
            reset_flow()
            agent = get_agent()
            config = {"configurable": {"thread_id": st.session_state.leads_thread_id}}

            initial_state = {
                "raw_data": df_to_text(df),
                "customer_data": get_customer_contact_data(df),
                "spending_threshold": spending_threshold,
                "categorized_leads": [],
                "promotions": [],
                "approved_leads": [],
                "error": "",
            }

            with st.spinner(
                "Filtrando clientes, categorizando y generando promociones con IA… "
                "(procesando cliente por cliente)"
            ):
                result = agent.invoke(initial_state, config=config)

            if result.get("error"):
                st.error(f"Error: {result['error']}")
                return

            snapshot = agent.get_state(config)
            if snapshot.next:  # interrumpido en human_review
                promotions = result.get("promotions", [])
                st.session_state.leads_promotions = promotions
                # Inicializar mensajes en session_state para edición
                for p in promotions:
                    st.session_state[f"msg_{p['id_cliente']}"] = p.get("mensaje_promo", "")
                st.session_state.leads_phase = "awaiting_review"
                st.rerun()
            else:
                st.error("El agente finalizó sin solicitar revisión. Intenta de nuevo.")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — HUMAN IN THE LOOP
    # ══════════════════════════════════════════════════════════════════════════
    elif phase == "awaiting_review":
        promotions = st.session_state.leads_promotions
        feedback_blocks = get_feedback_blocks(df)

        if not promotions:
            st.warning("No se generaron leads con el umbral seleccionado.")
            if st.button("↩️ Volver"):
                reset_flow()
                st.rerun()
            return

        st.success(
            f"**{len(promotions)} leads generados.** "
            "Revisa cada promoción, edita el mensaje o pídele cambios a la IA."
        )

        col_back, _ = st.columns([1, 4])
        with col_back:
            if st.button("↩️ Volver a generar"):
                reset_flow()
                st.rerun()

        st.markdown("---")

        approve_flags: dict[int, bool] = {}

        for promo in promotions:
            id_c = promo["id_cliente"]
            cat = promo.get("categoria", "")
            meta = CATEGORIA_META.get(cat, {"emoji": "⚪", "label": cat})
            msg_key = f"msg_{id_c}"

            # Aplicar mensaje pendiente (regeneración IA) ANTES de instanciar el widget
            pending_key = f"pending_msg_{id_c}"
            if pending_key in st.session_state:
                st.session_state[msg_key] = st.session_state.pop(pending_key)
            elif msg_key not in st.session_state:
                st.session_state[msg_key] = promo.get("mensaje_promo", "")

            with st.container(border=True):

                # ── Cabecera + checkbox ────────────────────────────────────
                h_col, chk_col = st.columns([5, 1])
                with h_col:
                    st.markdown(
                        f"#### {meta['emoji']} Cliente #{id_c}"
                        f"&nbsp;|&nbsp; S/. {promo['consumo']:.2f}"
                        f"&nbsp;|&nbsp; {meta['label']}"
                    )
                with chk_col:
                    approve_flags[id_c] = st.checkbox(
                        "Aprobar", value=True, key=f"chk_{id_c}"
                    )

                # ── Info del cliente ───────────────────────────────────────
                i1, i2 = st.columns(2)
                i1.markdown(f"📞 **Tel:** `{promo.get('telefono', '—')}`")
                i2.markdown(f"💡 **Motivo:** {promo.get('motivo', '—')}")

                # ── Mensaje editable ───────────────────────────────────────
                st.text_area(
                    "✏️ Mensaje promocional (editable directamente)",
                    key=msg_key,
                    height=110,
                )

                # ── Sección de regeneración con IA ─────────────────────────
                # Aplicar limpieza pendiente de instrucciones ANTES de instanciar el widget
                instr_key = f"instr_{id_c}"
                if st.session_state.pop(f"clear_instr_{id_c}", False):
                    st.session_state[instr_key] = ""

                st.markdown("**¿Quieres que la IA modifique el mensaje?**")
                instr_col, btn_col = st.columns([4, 1])
                with instr_col:
                    st.text_input(
                        "instrucciones",
                        key=instr_key,
                        placeholder=(
                            "Ej: Hazlo más formal, ofrece 20% de descuento, "
                            "menciona el ceviche que pidió..."
                        ),
                        label_visibility="collapsed",
                    )
                with btn_col:
                    if st.button(
                        "🤖 Regenerar",
                        key=f"regen_{id_c}",
                        use_container_width=True,
                    ):
                        instructions = st.session_state.get(instr_key, "").strip()
                        if not instructions:
                            st.warning("Escribe instrucciones antes de regenerar.")
                        else:
                            feedback = feedback_blocks.get(id_c, "Sin feedback disponible.")
                            with st.spinner(f"Regenerando mensaje para cliente #{id_c}…"):
                                new_msg = regenerate_single_promotion(
                                    promo, instructions, feedback
                                )
                            st.session_state[f"pending_msg_{id_c}"] = new_msg
                            st.session_state[f"clear_instr_{id_c}"] = True
                            st.rerun()

        # ── Confirmación ───────────────────────────────────────────────────────
        st.markdown("---")
        n_approved = sum(1 for v in approve_flags.values() if v)
        cola, colb = st.columns([3, 1])
        cola.markdown(f"**{n_approved} de {len(promotions)} leads seleccionados para enviar.**")

        if colb.button(
            "✅ Confirmar envío",
            type="primary",
            use_container_width=True,
            disabled=(n_approved == 0),
        ):
            approved_leads = [
                {
                    **promo,
                    "mensaje_promo": st.session_state.get(
                        f"msg_{promo['id_cliente']}", promo["mensaje_promo"]
                    ),
                }
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
    # FASE 3 — DONE
    # ══════════════════════════════════════════════════════════════════════════
    elif phase == "done":
        approved = st.session_state.leads_approved

        st.success(f"✅ **{len(approved)} leads aprobados** y listos para enviar.")

        if st.button("🔄 Generar nuevos leads"):
            reset_flow()
            st.rerun()

        st.markdown("---")

        if approved:
            k1, k2, k3 = st.columns(3)
            k1.metric("Leads aprobados", len(approved))
            k2.metric(
                "Consumo promedio",
                f"S/. {sum(l.get('consumo', 0) for l in approved) / len(approved):.2f}",
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
                    f"#### {meta['emoji']} Cliente #{lead['id_cliente']}"
                    f"&nbsp;|&nbsp; S/. {lead.get('consumo', 0):.2f}"
                    f"&nbsp;|&nbsp; {meta['label']}"
                )
                c1, c2 = st.columns(2)
                c1.markdown(f"📞 **Teléfono:** `{lead.get('telefono', '—')}`")
                c2.markdown(f"💡 **Motivo:** {lead.get('motivo', '—')}")
                st.markdown("**Mensaje a enviar:**")
                st.info(lead.get("mensaje_promo", "—"))

        if approved:
            st.markdown("---")
            export_df = pd.DataFrame(approved)
            cols = [
                c for c in
                ["id_cliente", "telefono", "consumo", "categoria", "motivo", "mensaje_promo"]
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
