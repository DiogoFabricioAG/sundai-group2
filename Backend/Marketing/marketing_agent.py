import base64
import json
import logging
import os
import re
from typing import List, TypedDict

import openai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

load_dotenv()

logger = logging.getLogger("marketing_agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

_checkpointer = MemorySaver()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text) -> dict | list:
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        text = "\n".join(parts)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    clean = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
    clean = re.sub(r"\n?```\s*$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No se pudo parsear JSON:\n{text[:300]}")


def _get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=temperature)


def _generate_image_bytes(prompt: str) -> bytes:
    """Genera imagen vía OpenAI gpt-image-1. Devuelve bytes."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no está configurada en el entorno.")

    client = openai.OpenAI(api_key=api_key)
    logger.info("Llamando a OpenAI gpt-image-1…")

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1536x1024",
    )

    b64_data = result.data[0].b64_json
    if not b64_data:
        raise RuntimeError("No se recibió imagen en la respuesta de OpenAI.")

    img_bytes = base64.b64decode(b64_data)
    logger.info("Imagen recibida: %s bytes", f"{len(img_bytes):,}")
    return img_bytes


def _build_image_prompt(dishes: List[str], extra: str = "") -> str:
    dishes_str = " and ".join(dishes[:3])
    base = (
        f"Professional food photography for a Peruvian restaurant marketing campaign. "
        f"Hero dishes: {dishes_str}. "
        f"Style: vibrant and appetizing, warm golden-hour lighting, "
        f"elegant plating on a rustic wooden table with Andean textiles, "
        f"fresh colorful ingredients visible around the dish, "
        f"shallow depth of field, bokeh background, "
        f"high-end food magazine quality, wide 16:9 format. "
        f"No text overlays. No people."
    )
    if extra:
        base += f" Additional art direction: {extra}"
    return base


# ── 1. Extracción de platos (standalone, cacheado en Streamlit) ───────────────

def extract_top_dishes(food_responses: List[str], top_n: int = 10) -> List[dict]:
    llm = _get_llm(temperature=0)
    clean = [str(r) for r in food_responses if str(r).strip() not in ("", "nan", "NaN")]
    combined = "\n".join(f"- {r}" for r in clean)
    logger.info("Analizando %d respuestas de comida…", len(clean))

    system = """Eres un analista de datos para restaurantes peruanos.
Analiza las respuestas de clientes sobre qué les gustó más de la comida.
Identifica los PLATOS o INGREDIENTES específicos más mencionados y estima su frecuencia.

Reglas:
- Solo incluye nombres de platos o ingredientes concretos (ceviche, lomo saltado, etc.)
- Ignora expresiones genéricas: "todo", "la sazón", "la comida en general", "el sabor"
- Si un plato aparece con variaciones, agrúpalos (ceviche + ceviche de conchas = ceviche)
- Estima las menciones de forma razonable

Devuelve ÚNICAMENTE un JSON válido (sin markdown):
{
  "platos": [
    {"plato": "<nombre del plato>", "menciones": <número entero>},
    ...
  ]
}
Máximo 10 platos, ordenados de mayor a menor mención."""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Respuestas de clientes:\n{combined}"),
    ]

    try:
        response = llm.invoke(messages)
        data = _parse_json(response.content)
        dishes = data.get("platos", [])[:top_n]
        logger.info("Platos extraídos: %s", [d["plato"] for d in dishes])
        return dishes
    except Exception as e:
        logger.error("Error extrayendo platos: %s", e)
        return []


# ── 2. LangGraph — Estado y Grafo ────────────────────────────────────────────

class MarketingState(TypedDict):
    selected_dishes: List[str]
    campaign_text: str
    image_bytes: bytes
    approved_text: str
    approved_image: bytes
    error: str


def create_marketing_agent():
    """Crea el grafo LangGraph: generate_text → generate_image → human_review."""

    # ── Nodo 1: Generar texto ──────────────────────────────────────────────
    def generate_text(state: MarketingState) -> dict:
        logger.info("── generate_text START ──")
        llm = _get_llm(temperature=0.9)
        dishes_str = ", ".join(state["selected_dishes"])

        system = """Eres un copywriter creativo especializado en gastronomía peruana.
Crea un texto de campaña de marketing para redes sociales que:
- Sea apasionante, evocador y apetecible
- Mencione los platos de forma poética y sensorial
- Use entre 2 y 4 emojis apropiados
- Incluya UN hashtag creativo al final
- Termine con un call-to-action breve (ej: "¡Reserva tu mesa hoy!")
- Máximo 5 líneas en total
- Tono: cálido, auténtico, peruano

Devuelve ÚNICAMENTE el texto de la campaña, sin JSON, sin comillas extra."""

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Platos para la campaña: {dishes_str}"),
        ]

        try:
            response = llm.invoke(messages)
            text = response.content
            if isinstance(text, list):
                text = " ".join(
                    p if isinstance(p, str) else p.get("text", "") for p in text
                )
            text = text.strip()
            logger.info("── generate_text END ── %d chars", len(text))
            return {"campaign_text": text}
        except Exception as e:
            logger.error("Error generando texto: %s", e)
            return {"error": f"Error generando texto: {e}"}

    # ── Nodo 2: Generar imagen ─────────────────────────────────────────────
    def generate_image(state: MarketingState) -> dict:
        if state.get("error"):
            return {}
        logger.info("── generate_image START ──")
        try:
            prompt = _build_image_prompt(state["selected_dishes"])
            img_bytes = _generate_image_bytes(prompt)
            logger.info("── generate_image END ── %s bytes", f"{len(img_bytes):,}")
            return {"image_bytes": img_bytes}
        except Exception as e:
            logger.error("Error generando imagen: %s", e)
            return {"error": f"Error generando imagen: {e}"}

    # ── Nodo 3: Human-in-the-loop ──────────────────────────────────────────
    def human_review(state: MarketingState) -> dict:
        logger.info("── human_review ── esperando revisión humana")
        approved = interrupt({
            "campaign_text": state.get("campaign_text", ""),
            "image_bytes": state.get("image_bytes", b""),
        })
        logger.info("── human_review RESUMED ── campaña aprobada")
        return {
            "approved_text": approved.get("campaign_text", state.get("campaign_text", "")),
            "approved_image": approved.get("image_bytes", state.get("image_bytes", b"")),
        }

    # ── Construcción del grafo ─────────────────────────────────────────────
    graph = StateGraph(MarketingState)
    graph.add_node("generate_text", generate_text)
    graph.add_node("generate_image", generate_image)
    graph.add_node("human_review", human_review)

    graph.set_entry_point("generate_text")
    graph.add_edge("generate_text", "generate_image")
    graph.add_edge("generate_image", "human_review")
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=_checkpointer)


# ── 3. Funciones standalone para HITL ─────────────────────────────────────────

def regenerate_campaign_text(current_text: str, dishes: List[str], instructions: str) -> str:
    """Regenera el texto de campaña siguiendo instrucciones del humano."""
    logger.info("── regenerate_campaign_text ── instrucciones: %s", instructions[:80])
    llm = _get_llm(temperature=0.9)

    system = """Eres un copywriter creativo especializado en gastronomía peruana.
Tienes un texto de campaña ya generado y debes modificarlo siguiendo
exactamente las instrucciones del revisor humano.

Reglas:
- Mantén el tono cálido, auténtico y peruano
- Sigue las instrucciones de cambio al pie de la letra
- Máximo 5 líneas en total
- Termina con un call-to-action y un hashtag

Devuelve ÚNICAMENTE el texto modificado, sin JSON, sin comillas extra."""

    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Platos de la campaña: {', '.join(dishes)}\n\n"
                f"Texto actual:\n{current_text}\n\n"
                f"Instrucciones del revisor:\n{instructions}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        text = response.content
        if isinstance(text, list):
            text = " ".join(
                p if isinstance(p, str) else p.get("text", "") for p in text
            )
        result = text.strip()
        logger.info("Texto regenerado: %d chars", len(result))
        return result
    except Exception as e:
        logger.error("Error regenerando texto: %s", e)
        return current_text


def regenerate_campaign_image(dishes: List[str], instructions: str) -> bytes:
    """Regenera la imagen de campaña con instrucciones adicionales del humano."""
    logger.info("── regenerate_campaign_image ── instrucciones: %s", instructions[:80])
    prompt = _build_image_prompt(dishes, extra=instructions)
    return _generate_image_bytes(prompt)
