import pandas as pd
import streamlit as st

from Backend.Marketing.marketing_agent import (
    extract_top_dishes,
    generate_campaign_image,
    generate_campaign_text,
)
from Frontend.utils.data_loader import load_data

st.set_page_config(
    page_title="Marketing – RestaurantAI",
    page_icon="🎨",
    layout="wide",
)

TOP_DISHES_CSV = "Data/top_platos.csv"


# ── Extracción de platos (cacheada: solo corre una vez por sesión de servidor) ─
@st.cache_data(show_spinner="Analizando platos favoritos con IA…")
def get_top_dishes() -> list[dict]:
    """Extrae y rankea los platos más mencionados del CSV de feedback."""
    df = load_data()
    food_col = "¿Qué te gustó más de la comida?"
    responses = df[food_col].dropna().tolist()
    dishes = extract_top_dishes(responses, top_n=10)

    if dishes:
        pd.DataFrame(dishes).to_csv(TOP_DISHES_CSV, index=False, encoding="utf-8")

    return dishes


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.title("🎨 Campañas de Marketing")
    st.caption(
        "Selecciona los platos estrella · genera texto con IA · "
        "crea la imagen con Google Imagen 3"
    )

    # ── Obtener platos top ─────────────────────────────────────────────────────
    top_dishes = get_top_dishes()
    if not top_dishes:
        st.error("No se pudieron extraer platos del dataset. Revisa la consola.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Checklist de platos
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("🍽️ Platos para campaña")
        st.caption("Selecciona los platos que quieres destacar.")
        st.markdown("---")

        selected_dishes: list[str] = []
        for d in top_dishes:
            label = f"**{d['plato']}**  ·  {d['menciones']} menciones"
            if st.checkbox(label, value=False, key=f"dish_{d['plato']}"):
                selected_dishes.append(d["plato"])

        st.markdown("---")

        can_generate = len(selected_dishes) > 0
        if not can_generate:
            st.caption("☝️ Selecciona al menos un plato.")

        generate_btn = st.button(
            "✨ Generar Campaña",
            type="primary",
            use_container_width=True,
            disabled=not can_generate,
        )

        # Botón de limpiar
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.pop("mkt_image", None)
            st.session_state.pop("mkt_text", None)
            st.session_state.pop("mkt_error", None)
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # GENERACIÓN — cuando se pulsa el botón
    # ══════════════════════════════════════════════════════════════════════════
    if generate_btn and selected_dishes:
        st.session_state.pop("mkt_image", None)
        st.session_state.pop("mkt_text", None)
        st.session_state.pop("mkt_error", None)

        with st.status(
            f"Creando campaña para: {', '.join(selected_dishes)}…", expanded=True
        ) as status:
            st.write("✍️ Generando texto de campaña…")
            try:
                campaign_text = generate_campaign_text(selected_dishes)
                st.session_state.mkt_text = campaign_text
                st.write("✅ Texto listo.")
            except Exception as e:
                st.session_state.mkt_error = f"Error en texto: {e}"
                st.write(f"❌ {st.session_state.mkt_error}")

            st.write("🖼️ Generando imagen con Gemini…")
            try:
                image_bytes = generate_campaign_image(selected_dishes)
                st.session_state.mkt_image = image_bytes
                st.write("✅ Imagen lista.")
            except Exception as e:
                st.session_state.mkt_error = f"Error en imagen: {e}"
                st.write(f"❌ {st.session_state.mkt_error}")

            status.update(label="✅ Campaña generada.", state="complete", expanded=False)

        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY — imagen (arriba) + texto (abajo)
    # ══════════════════════════════════════════════════════════════════════════
    if "mkt_error" in st.session_state:
        st.error(st.session_state.mkt_error)

    has_image = "mkt_image" in st.session_state
    has_text = "mkt_text" in st.session_state

    # ── Área de imagen ─────────────────────────────────────────────────────────
    if has_image:
        st.image(
            st.session_state.mkt_image,
            use_container_width=True,
            caption="Campaña generada con Google Imagen 3",
        )
    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                border-radius: 12px;
                height: 340px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                color: #aaa;
                font-size: 1.1rem;
                border: 1px dashed #444;
            ">
                <div style="font-size: 3rem; margin-bottom: 12px;">🖼️</div>
                <div>La imagen de tu campaña aparecerá aquí</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Barra horizontal inferior: texto de campaña ────────────────────────────
    st.markdown("---")

    if has_text:
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
                ">{st.session_state.mkt_text}</div>
                """,
                unsafe_allow_html=True,
            )

        with action_col:
            st.subheader("Acciones")

            if has_image:
                st.download_button(
                    "⬇️ Imagen (JPG)",
                    data=st.session_state.mkt_image,
                    file_name="campaña_marketing.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

    else:
        st.info(
            "El texto de la campaña aparecerá aquí. "
            "Selecciona platos en el panel izquierdo y pulsa **✨ Generar Campaña**."
        )

    # ── Tabla de referencia: platos favoritos ──────────────────────────────────
    with st.expander("📊 Ver ranking de platos favoritos"):
        df_dishes = pd.DataFrame(top_dishes)
        df_dishes.index = df_dishes.index + 1
        df_dishes.columns = ["Plato", "Menciones"]
        st.dataframe(df_dishes, use_container_width=True)


if __name__ == "__main__":
    main()
