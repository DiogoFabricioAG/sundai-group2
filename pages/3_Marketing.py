import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from Backend.Marketing.marketing_agent import (
    create_marketing_agent,
    extract_top_dishes,
    regenerate_campaign_image,
    regenerate_campaign_text,
)
from Frontend.utils.data_loader import load_data

st.set_page_config(
    page_title="Marketing – RestaurantAI",
    page_icon="🎨",
    layout="wide",
)

TOP_DISHES_CSV = "Data/top_platos.csv"


# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Analizando platos favoritos con IA…")
def get_top_dishes() -> list[dict]:
    df = load_data()
    food_col = "¿Qué te gustó más de la comida?"
    responses = df[food_col].dropna().tolist()
    dishes = extract_top_dishes(responses, top_n=10)
    if dishes:
        pd.DataFrame(dishes).to_csv(TOP_DISHES_CSV, index=False, encoding="utf-8")
    return dishes


@st.cache_resource
def get_agent():
    return create_marketing_agent()


# ── Session state ─────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "mkt_phase": "idle",          # idle | reviewing | done
        "mkt_thread_id": str(uuid.uuid4()),
        "mkt_text": "",
        "mkt_image": b"",
        "mkt_selected": [],
        "mkt_approved_text": "",
        "mkt_approved_image": b"",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_flow():
    st.session_state.mkt_phase = "idle"
    st.session_state.mkt_thread_id = str(uuid.uuid4())
    st.session_state.mkt_text = ""
    st.session_state.mkt_image = b""
    st.session_state.mkt_selected = []
    st.session_state.mkt_approved_text = ""
    st.session_state.mkt_approved_image = b""
    for key in list(st.session_state.keys()):
        if key.startswith(("mkt_instr_", "mkt_pending_", "mkt_clear_", "mkt_error")):
            del st.session_state[key]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_session()

    st.title("🎨 Campañas de Marketing")
    st.caption(
        "Selecciona los platos estrella · genera texto e imagen con IA · "
        "revisa y ajusta antes de aprobar · Powered by Gemini + OpenAI + LangGraph"
    )

    top_dishes = get_top_dishes()
    if not top_dishes:
        st.error("No se pudieron extraer platos del dataset. Revisa la consola.")
        return

    phase = st.session_state.mkt_phase

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("🍽️ Platos para campaña")
        st.caption("Selecciona los platos que quieres destacar.")
        st.markdown("---")

        selected_dishes: list[str] = []
        disabled_checks = phase != "idle"
        for d in top_dishes:
            label = f"**{d['plato']}**  ·  {d['menciones']} menciones"
            if st.checkbox(
                label, value=False, key=f"dish_{d['plato']}",
                disabled=disabled_checks,
            ):
                selected_dishes.append(d["plato"])

        st.markdown("---")

        can_generate = len(selected_dishes) > 0 and phase == "idle"
        if phase == "idle" and not selected_dishes:
            st.caption("☝️ Selecciona al menos un plato.")

        generate_btn = st.button(
            "✨ Generar Campaña",
            type="primary",
            use_container_width=True,
            disabled=not can_generate,
        )

        if phase != "idle":
            if st.button("🗑️ Nueva campaña", use_container_width=True):
                reset_flow()
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1 — IDLE
    # ══════════════════════════════════════════════════════════════════════════
    if phase == "idle":
        _render_placeholder()

        if generate_btn and selected_dishes:
            reset_flow()
            st.session_state.mkt_selected = selected_dishes
            agent = get_agent()
            config = {"configurable": {"thread_id": st.session_state.mkt_thread_id}}

            initial_state = {
                "selected_dishes": selected_dishes,
                "campaign_text": "",
                "image_bytes": b"",
                "approved_text": "",
                "approved_image": b"",
                "error": "",
            }

            with st.status("Generando campaña con IA…", expanded=True) as status:
                st.write("✍️ Generando texto de campaña…")
                st.write("🖼️ Generando imagen (esto puede tardar ~30s)…")
                result = agent.invoke(initial_state, config=config)
                status.update(label="✅ Campaña generada", state="complete")

            if result.get("error"):
                st.error(result["error"])
                return

            snapshot = agent.get_state(config)
            if snapshot.next:
                st.session_state.mkt_text = result.get("campaign_text", "")
                st.session_state.mkt_image = result.get("image_bytes", b"")
                st.session_state.mkt_phase = "reviewing"
                st.rerun()
            else:
                st.error("El agente finalizó sin solicitar revisión.")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — HUMAN IN THE LOOP (reviewing)
    # ══════════════════════════════════════════════════════════════════════════
    elif phase == "reviewing":
        _render_reviewing()

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3 — DONE
    # ══════════════════════════════════════════════════════════════════════════
    elif phase == "done":
        _render_done()

    # ── Tabla de referencia ───────────────────────────────────────────────────
    with st.expander("📊 Ver ranking de platos favoritos"):
        df_dishes = pd.DataFrame(top_dishes)
        df_dishes.index = df_dishes.index + 1
        df_dishes.columns = ["Plato", "Menciones"]
        st.dataframe(df_dishes, use_container_width=True)


# ── Renders ───────────────────────────────────────────────────────────────────

def _render_placeholder():
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 12px; height: 340px;
            display: flex; align-items: center; justify-content: center;
            flex-direction: column; color: #aaa; font-size: 1.1rem;
            border: 1px dashed #444;
        ">
            <div style="font-size: 3rem; margin-bottom: 12px;">🖼️</div>
            <div>La imagen de tu campaña aparecerá aquí</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.info(
        "Selecciona platos en el panel izquierdo y pulsa "
        "**✨ Generar Campaña** para comenzar."
    )


def _render_reviewing():
    dishes = st.session_state.mkt_selected

    st.success(
        "**Campaña generada.** Revisa el texto y la imagen. "
        "Puedes editarlos directamente o pedirle cambios a la IA."
    )

    # ── Aplicar cambios pendientes (regeneración IA) ──────────────────────
    if "mkt_pending_text" in st.session_state:
        st.session_state.mkt_text = st.session_state.pop("mkt_pending_text")
    if "mkt_pending_image" in st.session_state:
        st.session_state.mkt_image = st.session_state.pop("mkt_pending_image")
    if st.session_state.pop("mkt_clear_instr_text", False):
        st.session_state["mkt_instr_text"] = ""
    if st.session_state.pop("mkt_clear_instr_image", False):
        st.session_state["mkt_instr_image"] = ""

    # ── Imagen ────────────────────────────────────────────────────────────
    st.subheader("🖼️ Imagen de campaña")

    if st.session_state.mkt_image:
        st.image(st.session_state.mkt_image, use_container_width=True)
    else:
        st.warning("No se pudo generar la imagen.")

    st.markdown("**¿Quieres que la IA modifique la imagen?**")
    img_instr_col, img_btn_col = st.columns([4, 1])
    with img_instr_col:
        st.text_input(
            "instrucciones_imagen",
            key="mkt_instr_image",
            placeholder="Ej: Hazla más cálida, agrega más colores, cambia a fondo oscuro…",
            label_visibility="collapsed",
        )
    with img_btn_col:
        if st.button("🤖 Regenerar imagen", use_container_width=True):
            instructions = st.session_state.get("mkt_instr_image", "").strip()
            if not instructions:
                st.warning("Escribe instrucciones antes de regenerar.")
            else:
                with st.spinner("Regenerando imagen…"):
                    try:
                        new_img = regenerate_campaign_image(dishes, instructions)
                        st.session_state.mkt_pending_image = new_img
                        st.session_state.mkt_clear_instr_image = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error regenerando imagen: {e}")

    st.markdown("---")

    # ── Texto ─────────────────────────────────────────────────────────────
    st.subheader("📣 Texto de campaña")

    st.text_area(
        "✏️ Texto de campaña (editable directamente)",
        key="mkt_text",
        height=150,
    )

    st.markdown("**¿Quieres que la IA modifique el texto?**")
    txt_instr_col, txt_btn_col = st.columns([4, 1])
    with txt_instr_col:
        st.text_input(
            "instrucciones_texto",
            key="mkt_instr_text",
            placeholder="Ej: Hazlo más formal, menciona el ceviche primero, agrega más emojis…",
            label_visibility="collapsed",
        )
    with txt_btn_col:
        if st.button("🤖 Regenerar texto", use_container_width=True):
            instructions = st.session_state.get("mkt_instr_text", "").strip()
            if not instructions:
                st.warning("Escribe instrucciones antes de regenerar.")
            else:
                current_text = st.session_state.get("mkt_text", "")
                with st.spinner("Regenerando texto…"):
                    new_text = regenerate_campaign_text(
                        current_text, dishes, instructions
                    )
                st.session_state.mkt_pending_text = new_text
                st.session_state.mkt_clear_instr_text = True
                st.rerun()

    # ── Botones de acción ─────────────────────────────────────────────────
    st.markdown("---")
    col_approve, col_back = st.columns([3, 1])

    with col_approve:
        if st.button(
            "✅ Aprobar campaña",
            type="primary",
            use_container_width=True,
        ):
            approved_text = st.session_state.get("mkt_text", "")
            approved_image = st.session_state.get("mkt_image", b"")

            agent = get_agent()
            config = {"configurable": {"thread_id": st.session_state.mkt_thread_id}}

            with st.spinner("Confirmando campaña…"):
                result = agent.invoke(
                    Command(resume={
                        "campaign_text": approved_text,
                        "image_bytes": approved_image,
                    }),
                    config=config,
                )

            st.session_state.mkt_approved_text = result.get(
                "approved_text", approved_text
            )
            st.session_state.mkt_approved_image = result.get(
                "approved_image", approved_image
            )
            st.session_state.mkt_phase = "done"
            st.rerun()

    with col_back:
        if st.button("↩️ Volver a generar", use_container_width=True):
            reset_flow()
            st.rerun()


def _render_done():
    st.success("✅ **Campaña aprobada** y lista para publicar.")

    approved_text = st.session_state.mkt_approved_text
    approved_image = st.session_state.mkt_approved_image

    # ── Imagen final ──────────────────────────────────────────────────────
    if approved_image:
        st.image(approved_image, use_container_width=True, caption="Imagen aprobada")

    st.markdown("---")

    # ── Texto final + acciones ────────────────────────────────────────────
    text_col, action_col = st.columns([4, 1])

    with text_col:
        st.subheader("📣 Texto de campaña")
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                border-left: 4px solid #e63946;
                border-radius: 6px;
                padding: 16px 20px;
                font-size: 1.05rem;
                line-height: 1.7;
                color: #f0f0f0;
                white-space: pre-wrap;
            ">{approved_text}</div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        st.subheader("Acciones")

        if approved_image:
            st.download_button(
                "⬇️ Imagen (PNG)",
                data=approved_image,
                file_name="campaña_marketing.png",
                mime="image/png",
                use_container_width=True,
            )

        if approved_text:
            st.download_button(
                "📋 Texto (TXT)",
                data=approved_text.encode("utf-8"),
                file_name="campaña_texto.txt",
                mime="text/plain",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
