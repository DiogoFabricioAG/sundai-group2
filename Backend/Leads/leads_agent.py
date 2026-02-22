import json
import re
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

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


# Checkpointer compartido en memoria (persiste durante la sesión del servidor)
_checkpointer = MemorySaver()


class LeadsState(TypedDict):
    raw_data: str
    customer_data: str
    spending_threshold: float   # filtro cuantitativo: gasto mínimo
    scored_leads: List[dict]    # leads calificados (cuali + cuanti)
    promotions: List[dict]      # leads + mensaje promocional generado
    approved_leads: List[dict]  # leads aprobados por el humano (post-HITL)
    error: str


def create_leads_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

    # ── Nodo 1 ────────────────────────────────────────────────────────────────
    def filter_and_score(state: LeadsState) -> dict:
        """
        Filtro cuantitativo: descarta clientes que gastaron menos del umbral.
        Luego usa el LLM para scoring cualitativo sobre los clientes calificados.
        """
        threshold = state.get("spending_threshold", 0.0)

        # Parsear datos de gasto desde customer_data (formato: ID | Tel | Consumo)
        spending_info: dict[int, float] = {}
        for line in state["customer_data"].split("\n")[1:]:  # skip header
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    spending_info[int(parts[0])] = float(parts[2])
                except ValueError:
                    continue

        # Filtro cuantitativo
        qualified_ids = {k for k, v in spending_info.items() if v >= threshold}
        if not qualified_ids:
            return {
                "error": f"No hay clientes con gasto >= S/. {threshold:.2f}",
                "scored_leads": [],
            }

        # Filtrar bloques de feedback a solo los clientes calificados
        filtered_blocks = [
            block
            for block in state["raw_data"].split("\n\n")
            if (m := re.match(r"Cliente (\d+):", block.strip()))
            and int(m.group(1)) in qualified_ids
        ]
        filtered_data = "\n\n".join(filtered_blocks)

        system = """Eres un experto en CRM y marketing para restaurantes peruanos.
Los clientes ya fueron pre-filtrados por un umbral de gasto mínimo (filtro cuantitativo).
Ahora realiza el scoring CUALITATIVO basándote en el sentimiento del feedback.

Categorías:
- "alto_valor": gasto alto + satisfacción alta → potencial VIP
- "retencion": tuvo mala experiencia pero es recuperable con atención personalizada
- "recurrente": su feedback indica que definitivamente volverá
- "referidor": recomienda el lugar o puede traer nuevos clientes

Solo incluye clientes con score cualitativo >= 6.

Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin bloques de código):
{
  "leads": [
    {
      "id_cliente": <número>,
      "score": <entero 1-10>,
      "categoria": "<alto_valor|retencion|recurrente|referidor>",
      "motivo": "<una oración explicando el score combinado cuali+cuantitativo>"
    }
  ]
}"""
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Feedbacks pre-filtrados (umbral de gasto: S/. {threshold:.2f}):\n\n"
                    f"{filtered_data}\n\n"
                    f"Referencia de gasto por cliente:\n{state['customer_data']}"
                )
            ),
        ]
        try:
            response = llm.invoke(messages)
            result = parse_json(response.content)
            # Enriquecer leads con consumo real
            leads = [
                {**lead, "consumo": spending_info.get(lead["id_cliente"], 0.0)}
                for lead in result.get("leads", [])
            ]
            leads.sort(key=lambda x: x.get("score", 0), reverse=True)
            return {"scored_leads": leads}
        except Exception as e:
            return {"error": str(e), "scored_leads": []}

    # ── Nodo 2 ────────────────────────────────────────────────────────────────
    def generate_promotions(state: LeadsState) -> dict:
        """
        Para cada lead genera un mensaje promocional personalizado
        listo para enviar por WhatsApp.
        """
        if state.get("error") or not state.get("scored_leads"):
            return {"promotions": []}

        # Mapa de contacto: id_cliente → {telefono, consumo}
        contact_map: dict[int, dict] = {}
        for line in state["customer_data"].split("\n")[1:]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    contact_map[int(parts[0])] = {
                        "telefono": parts[1],
                        "consumo": float(parts[2]),
                    }
                except ValueError:
                    continue

        leads_json = json.dumps(state["scored_leads"], ensure_ascii=False, indent=2)

        system = """Eres copywriter experto en marketing de restaurantes peruanos.
Para cada lead genera un mensaje de WhatsApp personalizado y convincente que:
- Sea cálido, breve y natural (máximo 3-4 oraciones)
- Mencione algo específico de su experiencia en el restaurante
- Ofrezca una promoción concreta según su categoría:
  * alto_valor  → invitación VIP / mesa reservada / descuento exclusivo
  * retencion   → disculpa sincera + regalo o descuento especial para recuperar su confianza
  * recurrente  → beneficio por fidelidad / bienvenida anticipada
  * referidor   → invitación para traer un amigo + beneficio doble para ambos
- Termine con un call-to-action claro (ej: "Reserva al XXX-XXXX")

Devuelve ÚNICAMENTE un JSON válido (sin markdown, sin bloques de código):
{
  "promotions": [
    {
      "id_cliente": <número>,
      "mensaje_promo": "<mensaje de WhatsApp personalizado>"
    }
  ]
}"""
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Leads a los que generar promoción:\n{leads_json}\n\n"
                    f"Feedbacks originales (para personalizar el mensaje):\n{state['raw_data']}"
                )
            ),
        ]
        try:
            response = llm.invoke(messages)
            result = parse_json(response.content)
            promo_map = {
                p["id_cliente"]: p["mensaje_promo"]
                for p in result.get("promotions", [])
            }
            promotions = []
            for lead in state["scored_leads"]:
                id_c = lead["id_cliente"]
                contact = contact_map.get(id_c, {})
                promotions.append(
                    {
                        "id_cliente": id_c,
                        "telefono": contact.get("telefono", "—"),
                        "consumo": lead.get("consumo", 0.0),
                        "score": lead["score"],
                        "categoria": lead["categoria"],
                        "motivo": lead["motivo"],
                        "mensaje_promo": promo_map.get(id_c, ""),
                    }
                )
            return {"promotions": promotions}
        except Exception as e:
            return {"error": str(e), "promotions": []}

    # ── Nodo 3 — HITL ─────────────────────────────────────────────────────────
    def human_review(state: LeadsState) -> dict:
        """
        Human-in-the-loop: pausa el grafo y devuelve las promociones al frontend.
        El humano puede aprobar, rechazar o editar cada mensaje.
        La ejecución se reanuda cuando el frontend llama a Command(resume=...).
        """
        approved = interrupt(state["promotions"])
        return {"approved_leads": approved}

    # ── Construcción del grafo ─────────────────────────────────────────────────
    graph = StateGraph(LeadsState)
    graph.add_node("filter_and_score", filter_and_score)
    graph.add_node("generate_promotions", generate_promotions)
    graph.add_node("human_review", human_review)

    graph.set_entry_point("filter_and_score")
    graph.add_edge("filter_and_score", "generate_promotions")
    graph.add_edge("generate_promotions", "human_review")
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=_checkpointer)
