import json
import logging
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class DashboardState(TypedDict):
    raw_data: str
    sentiment_scores: dict
    key_themes: dict
    summary: dict
    error: str


def _parse_llm_json(content, node_name: str) -> dict:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        text = "\n".join(parts).strip()
    else:
        text = str(content).strip()

    if not text:
        raise ValueError("Respuesta vacía del modelo")

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            return json.loads(obj_match.group(0))

        logger.error(
            "[%s] No se pudo parsear JSON. Respuesta recibida (preview): %s",
            node_name,
            text[:300],
        )
        raise


def create_dashboard_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    def analyze_sentiment(state: DashboardState) -> dict:
        """Nodo 1: Calcula scores de sentimiento por categoría."""
        logger.info("[analyze_sentiment] Inicio")
        system = """Eres un analista de experiencia del cliente para restaurantes peruanos.
Analiza los feedbacks y devuelve ÚNICAMENTE un JSON válido (sin markdown, sin texto adicional) con:
{
  "atencion": <score 0-10>,
  "comida": <score 0-10>,
  "precio_calidad": <score 0-10>,
  "ambiente": <score 0-10>,
  "experiencia_general": <score 0-10>,
  "positivos": <cantidad de clientes con experiencia positiva>,
  "negativos": <cantidad de clientes con experiencia negativa>,
  "neutros": <cantidad de clientes con experiencia neutra>
}"""
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=f"Analiza estos feedbacks de comensales:\n\n{state['raw_data']}"
            ),
        ]
        try:
            response = llm.invoke(messages)
            scores = _parse_llm_json(response.content, "analyze_sentiment")
            logger.info("[analyze_sentiment] OK")
            return {"sentiment_scores": scores}
        except Exception as e:
            logger.exception("[analyze_sentiment] Error")
            return {"error": str(e), "sentiment_scores": {}}

    def extract_themes(state: DashboardState) -> dict:
        """Nodo 2: Extrae temas, quejas y elogios principales."""
        if state.get("error"):
            logger.warning("[extract_themes] Saltado por error previo")
            return {}

        logger.info("[extract_themes] Inicio")

        system = """Eres un analista de experiencia del cliente para restaurantes peruanos.
Extrae los temas principales de los feedbacks y devuelve ÚNICAMENTE un JSON válido (sin markdown) con:
{
  "top_praises": ["elogio 1", "elogio 2", "elogio 3", "elogio 4", "elogio 5"],
  "top_complaints": ["queja 1", "queja 2", "queja 3", "queja 4", "queja 5"],
  "top_dishes": ["plato 1", "plato 2", "plato 3"],
  "improvement_areas": ["área 1", "área 2", "área 3"]
}"""
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=f"Extrae los temas de estos feedbacks:\n\n{state['raw_data']}"
            ),
        ]
        try:
            response = llm.invoke(messages)
            themes = _parse_llm_json(response.content, "extract_themes")
            logger.info("[extract_themes] OK")
            return {"key_themes": themes}
        except Exception as e:
            logger.exception("[extract_themes] Error")
            return {"error": str(e), "key_themes": {}}

    def build_summary(state: DashboardState) -> dict:
        """Nodo 3: Genera el resumen ejecutivo."""
        if state.get("error"):
            logger.warning("[build_summary] Saltado por error previo")
            return {}

        logger.info("[build_summary] Inicio")

        system = """Eres un consultor de restaurantes. Genera un resumen ejecutivo del feedback recibido
y devuelve ÚNICAMENTE un JSON válido (sin markdown) con:
{
  "resumen": "<2-3 oraciones resumiendo la experiencia general>",
  "fortaleza_principal": "<principal punto fuerte del restaurante>",
  "recomendacion_principal": "<acción más urgente para mejorar>"
}"""
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=f"Genera un resumen ejecutivo de estos feedbacks:\n\n{state['raw_data']}"
            ),
        ]
        try:
            response = llm.invoke(messages)
            summary = _parse_llm_json(response.content, "build_summary")
            logger.info("[build_summary] OK")
            return {"summary": summary}
        except Exception as e:
            logger.exception("[build_summary] Error")
            return {"error": str(e), "summary": {}}

    # Construcción del grafo
    graph = StateGraph(DashboardState)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("extract_themes", extract_themes)
    graph.add_node("build_summary", build_summary)

    graph.set_entry_point("analyze_sentiment")
    graph.add_edge("analyze_sentiment", "extract_themes")
    graph.add_edge("extract_themes", "build_summary")
    graph.add_edge("build_summary", END)

    return graph.compile()


def run_dashboard_agent(data_text: str) -> DashboardState:
    """Ejecuta el agente de dashboard y retorna los resultados."""
    logger.info("[run_dashboard_agent] Inicio de ejecución")
    logger.info("[run_dashboard_agent] Tamaño de entrada: %s caracteres", len(data_text))

    agent = create_dashboard_agent()
    initial_state: DashboardState = {
        "raw_data": data_text,
        "sentiment_scores": {},
        "key_themes": {},
        "summary": {},
        "error": "",
    }

    try:
        logger.info("[run_dashboard_agent] Invocando grafo de LangGraph")
        result = agent.invoke(initial_state)

        if result.get("error"):
            logger.error(
                "[run_dashboard_agent] Finalizó con error reportado en estado: %s",
                result.get("error"),
            )
        else:
            logger.info("[run_dashboard_agent] Finalizó correctamente")

        logger.info(
            "[run_dashboard_agent] Resultado -> sentiment_scores=%s, key_themes=%s, summary=%s",
            bool(result.get("sentiment_scores")),
            bool(result.get("key_themes")),
            bool(result.get("summary")),
        )
        return result
    except Exception:
        logger.exception("[run_dashboard_agent] Error no controlado durante la ejecución")
        raise
