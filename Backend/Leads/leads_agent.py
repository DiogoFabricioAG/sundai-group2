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

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No se pudo parsear JSON. Respuesta recibida:\n{text[:300]}")


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)


# Checkpointer compartido en memoria (persiste durante la sesión del servidor)
_checkpointer = MemorySaver()


class LeadsState(TypedDict):
    raw_data: str
    customer_data: str
    spending_threshold: float       # filtro cuantitativo: gasto mínimo
    categorized_leads: List[dict]   # {id_cliente, consumo, categoria, motivo}
    promotions: List[dict]          # {id_cliente, telefono, consumo, categoria, motivo, mensaje_promo}
    approved_leads: List[dict]      # leads aprobados por el humano (post-HITL)
    error: str


def create_leads_agent():
    llm = _get_llm()

    # ── Nodo 1 ────────────────────────────────────────────────────────────────
    def categorize_clients(state: LeadsState) -> dict:
        """
        1) Filtro cuantitativo: descarta clientes con gasto < umbral.
        2) Para cada cliente calificado, llama al LLM individualmente
           para determinar su categoría de lead (sin score).
        """
        threshold = state.get("spending_threshold", 0.0)

        # Parsear gasto desde customer_data (formato: ID | Tel | Consumo)
        spending_info: dict[int, float] = {}
        for line in state["customer_data"].split("\n")[1:]:
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
                "categorized_leads": [],
            }

        # Parsear bloques de feedback en un dict {id_cliente: texto}
        feedback_blocks: dict[int, str] = {}
        for block in state["raw_data"].split("\n\n"):
            m = re.match(r"Cliente (\d+):", block.strip())
            if m:
                feedback_blocks[int(m.group(1))] = block.strip()

        system = """Eres un experto en CRM para restaurantes peruanos.
Analiza el feedback de UN cliente y determina su categoría de lead.

Categorías (elige exactamente una):
- "alto_valor"  : gasto alto + satisfacción alta → potencial VIP
- "retencion"   : tuvo mala experiencia pero es recuperable con atención personalizada
- "recurrente"  : su feedback indica claramente que volverá
- "referidor"   : recomienda el lugar o tiene perfil para traer nuevos clientes

Devuelve ÚNICAMENTE un JSON válido (sin markdown):
{
  "categoria": "<alto_valor|retencion|recurrente|referidor>",
  "motivo": "<una oración explicando la categoría asignada>"
}"""

        # Procesar cliente por cliente para no saturar el contexto
        categorized: List[dict] = []
        for client_id in sorted(qualified_ids, key=lambda x: spending_info[x], reverse=True):
            feedback = feedback_blocks.get(client_id)
            if not feedback:
                continue

            messages = [
                SystemMessage(content=system),
                HumanMessage(content=f"Feedback del cliente:\n{feedback}"),
            ]
            try:
                response = llm.invoke(messages)
                result = parse_json(response.content)
                categorized.append({
                    "id_cliente": client_id,
                    "consumo": spending_info[client_id],
                    "categoria": result.get("categoria", "recurrente"),
                    "motivo": result.get("motivo", ""),
                })
            except Exception as e:
                categorized.append({
                    "id_cliente": client_id,
                    "consumo": spending_info[client_id],
                    "categoria": "recurrente",
                    "motivo": f"Categorización por defecto (error: {str(e)[:60]})",
                })

        return {"categorized_leads": categorized}

    # ── Nodo 2 ────────────────────────────────────────────────────────────────
    def generate_promotions(state: LeadsState) -> dict:
        """
        Para cada lead categorizado genera un mensaje promocional personalizado
        de WhatsApp, procesando uno por uno para controlar el contexto.
        """
        if state.get("error") or not state.get("categorized_leads"):
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

        # Bloques de feedback: {id_cliente: texto}
        feedback_blocks: dict[int, str] = {}
        for block in state["raw_data"].split("\n\n"):
            m = re.match(r"Cliente (\d+):", block.strip())
            if m:
                feedback_blocks[int(m.group(1))] = block.strip()

        system = """Eres copywriter experto en marketing de restaurantes peruanos.
Genera un mensaje de WhatsApp personalizado para el cliente que:
- Sea cálido, breve y natural (máximo 3-4 oraciones)
- Mencione algo específico de su experiencia
- Ofrezca una promoción concreta según la categoría:
  * alto_valor → invitación VIP / mesa reservada / descuento exclusivo
  * retencion  → disculpa sincera + regalo o descuento para recuperar su confianza
  * recurrente → beneficio por fidelidad / bienvenida anticipada
  * referidor  → invitación para traer un amigo + beneficio doble
- Termine con un call-to-action claro

Devuelve ÚNICAMENTE un JSON válido (sin markdown):
{
  "mensaje_promo": "<mensaje de WhatsApp personalizado>"
}"""

        promotions: List[dict] = []
        for lead in state["categorized_leads"]:
            id_c = lead["id_cliente"]
            feedback = feedback_blocks.get(id_c, "Sin feedback disponible.")
            contact = contact_map.get(id_c, {})

            messages = [
                SystemMessage(content=system),
                HumanMessage(
                    content=(
                        f"Categoría: {lead['categoria']}\n"
                        f"Motivo: {lead['motivo']}\n"
                        f"Feedback original del cliente:\n{feedback}"
                    )
                ),
            ]
            try:
                response = llm.invoke(messages)
                result = parse_json(response.content)
                mensaje = result.get("mensaje_promo", "")
            except Exception as e:
                mensaje = f"[Error al generar mensaje: {str(e)[:60]}]"

            promotions.append({
                "id_cliente": id_c,
                "telefono": contact.get("telefono", "—"),
                "consumo": lead["consumo"],
                "categoria": lead["categoria"],
                "motivo": lead["motivo"],
                "mensaje_promo": mensaje,
            })

        return {"promotions": promotions}

    # ── Nodo 3 — HITL ─────────────────────────────────────────────────────────
    def human_review(state: LeadsState) -> dict:
        """
        Human-in-the-loop: pausa el grafo y devuelve las promociones al frontend.
        Se reanuda con Command(resume=leads_aprobados).
        """
        approved = interrupt(state["promotions"])
        return {"approved_leads": approved}

    # ── Construcción del grafo ─────────────────────────────────────────────────
    graph = StateGraph(LeadsState)
    graph.add_node("categorize_clients", categorize_clients)
    graph.add_node("generate_promotions", generate_promotions)
    graph.add_node("human_review", human_review)

    graph.set_entry_point("categorize_clients")
    graph.add_edge("categorize_clients", "generate_promotions")
    graph.add_edge("generate_promotions", "human_review")
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=_checkpointer)


# ── Función standalone para el HITL ───────────────────────────────────────────
def regenerate_single_promotion(lead: dict, instructions: str, feedback: str) -> str:
    """
    Regenera el mensaje de promoción para un lead específico
    siguiendo las instrucciones del revisor humano.
    Llamado desde el frontend durante la fase de revisión HITL.
    """
    llm = _get_llm()

    system = """Eres copywriter experto en marketing de restaurantes peruanos.
Tienes un mensaje de WhatsApp ya generado para un cliente y debes modificarlo
siguiendo exactamente las instrucciones del revisor humano.

Reglas:
- Mantén el tono cálido y personalizado
- Sigue las instrucciones de cambio al pie de la letra
- Máximo 3-4 oraciones
- Termina con un call-to-action

Devuelve ÚNICAMENTE un JSON válido (sin markdown):
{
  "mensaje_promo": "<mensaje de WhatsApp modificado>"
}"""

    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Categoría del cliente: {lead['categoria']}\n"
                f"Motivo: {lead['motivo']}\n"
                f"Feedback original del cliente:\n{feedback}\n\n"
                f"Mensaje actual:\n{lead.get('mensaje_promo', '')}\n\n"
                f"Instrucciones del revisor:\n{instructions}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        result = parse_json(response.content)
        return result.get("mensaje_promo", lead.get("mensaje_promo", ""))
    except Exception:
        return lead.get("mensaje_promo", "")
