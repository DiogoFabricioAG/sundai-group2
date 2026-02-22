import json
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

load_dotenv()


class Lead(TypedDict):
    id_cliente: int
    score: int
    categoria: str
    motivo: str
    accion_sugerida: str


class LeadsState(TypedDict):
    raw_data: str
    customer_data: str
    scored_leads: List[Lead]
    error: str


def create_leads_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    def score_customers(state: LeadsState) -> dict:
        """Nodo 1: Analiza y puntúa cada cliente como lead potencial."""
        system = """Eres un experto en CRM y marketing para restaurantes peruanos.
Analiza los feedbacks e identifica clientes con potencial de fidelización o retorno.

Categorías de lead:
- "alto_valor": gastó bastante dinero y quedó satisfecho (potencial VIP)
- "retencion": tuvo mala experiencia pero es recuperable con atención personalizada
- "recurrente": su feedback indica que definitivamente volverá
- "referidor": recomienda el lugar o tiene perfil de influencer social

Solo incluye clientes con score >= 6.

Devuelve ÚNICAMENTE un JSON válido (sin markdown) con este formato:
{
  "leads": [
    {
      "id_cliente": <número>,
      "score": <entero 1-10>,
      "categoria": "<alto_valor|retencion|recurrente|referidor>",
      "motivo": "<una oración explicando el score>",
      "accion_sugerida": "<una acción concreta de CRM o marketing>"
    }
  ]
}"""
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Feedbacks de clientes:\n\n{state['raw_data']}\n\n"
                    f"Datos de contacto (referencia):\n{state['customer_data']}"
                )
            ),
        ]
        try:
            response = llm.invoke(messages)
            result = json.loads(response.content)
            return {"scored_leads": result.get("leads", [])}
        except Exception as e:
            return {"error": str(e), "scored_leads": []}

    def enrich_leads(state: LeadsState) -> dict:
        """Nodo 2: Enriquece los leads con insights adicionales si hay errores o datos faltantes."""
        if state.get("error") or not state.get("scored_leads"):
            return {}

        # Aquí se pueden agregar integraciones externas (CRM, email marketing, etc.)
        # Por ahora el nodo valida y ordena los leads por score descendente
        leads = sorted(
            state["scored_leads"],
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        return {"scored_leads": leads}

    # Construcción del grafo
    graph = StateGraph(LeadsState)
    graph.add_node("score_customers", score_customers)
    graph.add_node("enrich_leads", enrich_leads)

    graph.set_entry_point("score_customers")
    graph.add_edge("score_customers", "enrich_leads")
    graph.add_edge("enrich_leads", END)

    return graph.compile()


def run_leads_agent(data_text: str, customer_data: str) -> LeadsState:
    """Ejecuta el agente de leads y retorna los resultados."""
    agent = create_leads_agent()
    initial_state: LeadsState = {
        "raw_data": data_text,
        "customer_data": customer_data,
        "scored_leads": [],
        "error": "",
    }
    return agent.invoke(initial_state)
