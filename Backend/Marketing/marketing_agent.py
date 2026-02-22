import json
import os
import re
from typing import List

from dotenv import load_dotenv
import openai
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


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


# ── 1. Extracción de platos top ───────────────────────────────────────────────

def extract_top_dishes(food_responses: List[str], top_n: int = 10) -> List[dict]:
    """
    Usa LLM para identificar y rankear los platos más mencionados
    en las respuestas de feedback de comida.
    Devuelve lista de {"plato": str, "menciones": int} ordenada desc.
    """
    llm = _get_llm(temperature=0)

    clean = [str(r) for r in food_responses if str(r).strip() not in ("", "nan", "NaN")]
    combined = "\n".join(f"- {r}" for r in clean)

    print(f"[Marketing] Analizando {len(clean)} respuestas de comida...")

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
        raw = response.content
        print(f"[Marketing] Respuesta LLM (platos): {str(raw)[:300]}")
        data = _parse_json(raw)
        dishes = data.get("platos", [])[:top_n]
        print(f"[Marketing] Platos extraídos: {[d['plato'] for d in dishes]}")
        return dishes
    except Exception as e:
        print(f"[Marketing] Error extrayendo platos: {e}")
        return []


# ── 2. Generación de texto de campaña ─────────────────────────────────────────

def generate_campaign_text(selected_dishes: List[str]) -> str:
    """
    Genera un texto creativo de campaña de marketing para redes sociales
    basado en los platos seleccionados.
    """
    llm = _get_llm(temperature=0.9)
    dishes_str = ", ".join(selected_dishes)

    print(f"[Marketing] Generando texto para: {dishes_str}")

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

    response = llm.invoke(messages)
    text = response.content
    if isinstance(text, list):
        text = " ".join(p if isinstance(p, str) else p.get("text", "") for p in text)
    result = text.strip()
    print(f"[Marketing] Texto generado:\n{result}")
    return result


# ── 3. Generación de imagen con OpenAI (DALL·E) ──────────────────────────────

def generate_campaign_image(selected_dishes: List[str]) -> bytes:
    """
    Genera una imagen de campaña usando la API de OpenAI (gpt-image-1).
    Devuelve los bytes PNG de la imagen.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no está configurada en el entorno.")

    client = openai.OpenAI(api_key=api_key)

    dishes_str = " and ".join(selected_dishes[:3])
    prompt = (
        f"Professional food photography for a Peruvian restaurant marketing campaign. "
        f"Hero dishes: {dishes_str}. "
        f"Style: vibrant and appetizing, warm golden-hour lighting, "
        f"elegant plating on a rustic wooden table with Andean textiles, "
        f"fresh colorful ingredients visible around the dish, "
        f"shallow depth of field, bokeh background, "
        f"high-end food magazine quality, wide 16:9 format. "
        f"No text overlays. No people."
    )

    print(f"[Marketing] Generando imagen con OpenAI...")
    print(f"[Marketing] Prompt: {prompt}")

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1536x1024",
    )

    b64_data = result.data[0].b64_json
    if not b64_data:
        raise RuntimeError("No se recibió imagen en la respuesta de OpenAI.")

    import base64
    img_bytes = base64.b64decode(b64_data)
    print(f"[Marketing] Imagen generada: {len(img_bytes):,} bytes")
    return img_bytes
