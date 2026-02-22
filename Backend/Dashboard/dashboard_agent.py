import json
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

load_dotenv()


def parse_json(text: str | list) -> dict | list:
    """
    Parsea JSON de la respuesta del LLM con 3 estrategias de fallback:
      1. Parseo directo.
      2. Eliminar bloque de markdown (```json ... ``` o ``` ... ```).
      3. Extraer el primer objeto/arreglo JSON encontrado en el texto.

    También maneja el caso en que `response.content` es una lista de partes
    (comportamiento ocasional de langchain-google-genai).
    """
    # Normalizar a string si el modelo devuelve lista de partes
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        text = "\n".join(parts)

    text = text.strip()

    # Estrategia 1 — parseo directo
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Estrategia 2 — eliminar bloque de markdown
    clean = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
    clean = re.sub(r"\n?```\s*$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Estrategia 3 — extraer primer objeto o arreglo JSON del texto
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No se pudo parsear JSON. Respuesta recibida:\n{text[:300]}")


class DashboardState(TypedDict):
    raw_data: str
    sentiment_scores: dict
    key_themes: dict
    summary: dict
    error: str


def create_dashboard_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

    def analyze_sentiment(state: DashboardState) -> dict:
        """Nodo 1: Calcula scores de sentimiento por categoría."""
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
            scores = parse_json(response.content)
            return {"sentiment_scores": scores}
        except Exception as e:
            return {"error": str(e), "sentiment_scores": {}}

    def extract_themes(state: DashboardState) -> dict:
        """Nodo 2: Extrae temas, quejas y elogios principales."""
        if state.get("error"):
            return {}

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
            themes = parse_json(response.content)
            return {"key_themes": themes}
        except Exception as e:
            return {"error": str(e), "key_themes": {}}

    def build_summary(state: DashboardState) -> dict:
        """Nodo 3: Genera el resumen ejecutivo."""
        if state.get("error"):
            return {}

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
            summary = parse_json(response.content)
            return {"summary": summary}
        except Exception as e:
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
    agent = create_dashboard_agent()
    initial_state: DashboardState = {
        "raw_data": data_text,
        "sentiment_scores": {},
        "key_themes": {},
        "summary": {},
        "error": "",
    }
    return agent.invoke(initial_state)
