from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph


logger = logging.getLogger(__name__)
load_dotenv()


class TagAnalysisResult(TypedDict):
    sentiment_scores: dict
    key_themes: dict
    summary: dict
    error: str
    metadata: dict


class ExecutiveSummaryState(TypedDict):
    context_json: str
    summary: dict
    error: str


def _get_tag_comment_preview(
    events: pd.DataFrame,
    tag: str,
    polarity: str,
    max_items: int = 3,
) -> str:
    subset = events[(events["tag"] == tag) & (events["polarity"] == polarity)]
    if subset.empty:
        return "Sin comentarios de muestra"

    comments = (
        subset["text"]
        .astype(str)
        .drop_duplicates()
        .head(max_items)
        .tolist()
    )
    return "<br>".join(f"• {comment[:180]}" for comment in comments)


def _build_tag_insights(events: pd.DataFrame, customer_tag: pd.DataFrame) -> list[dict]:
    negative_counts = (
        customer_tag[customer_tag["polarity"] == "mal"]
        .groupby(["tag", "category"])
        .size()
        .rename("mal")
    )
    positive_counts = (
        customer_tag[customer_tag["polarity"] == "bien"]
        .groupby(["tag", "category"])
        .size()
        .rename("bien")
    )
    neutral_counts = (
        customer_tag[customer_tag["polarity"] == "neutral"]
        .groupby(["tag", "category"])
        .size()
        .rename("neutral")
    )

    combined = pd.concat([positive_counts, neutral_counts, negative_counts], axis=1).fillna(0).reset_index()
    if combined.empty:
        return []

    insights = []
    for _, row in combined.iterrows():
        tag = str(row["tag"])
        category = str(row["category"])
        bien = int(row.get("bien", 0))
        neutral = int(row.get("neutral", 0))
        mal = int(row.get("mal", 0))

        insights.append(
            {
                "tag": tag,
                "category": category,
                "bien": bien,
                "neutral": neutral,
                "mal": mal,
                "total_mentions": bien + neutral + mal,
                "balance": bien - mal,
                "hover_bien": _get_tag_comment_preview(events, tag, "bien"),
                "hover_neutral": _get_tag_comment_preview(events, tag, "neutral"),
                "hover_mal": _get_tag_comment_preview(events, tag, "mal"),
            }
        )

    insights.sort(key=lambda x: (x["mal"], x["neutral"], x["bien"]), reverse=True)
    return insights


def _format_tag_item(tag: str, count: int, balance: int) -> str:
    return f"{tag} ({count} clientes, balance {balance:+d})"


DATA_DIR = Path(__file__).parent.parent.parent / "Data"
TAG_EVENTS_PATH = DATA_DIR / "tag_events.csv"
TAG_INDEX_PATH = DATA_DIR / "tag_index.csv"
TAG_CATALOG_PATH = DATA_DIR / "tag_catalog.csv"
TAG_CATALOG_PENDING_PATH = DATA_DIR / "tag_catalog_pending.csv"
TAG_LLM_CACHE_PATH = DATA_DIR / "tag_llm_cache.csv"

QUESTION_COLUMNS = [
    "¿Qué mejorarías de la atención?",
    "¿Qué te pareció la atención?",
    "¿Qué te gustó más de la comida?",
    "¿Qué opina sobre la relación entre calidad y precio?",
    "¿Qué te gustó mas del ambiente?",
    "¿Qué es lo que cambiarías de la experiencia?",
]

QUESTION_DEFAULT_POLARITY = {
    "¿Qué mejorarías de la atención?": "mal",
    "¿Qué te pareció la atención?": "bien",
    "¿Qué te gustó más de la comida?": "bien",
    "¿Qué opina sobre la relación entre calidad y precio?": "bien",
    "¿Qué te gustó mas del ambiente?": "bien",
    "¿Qué es lo que cambiarías de la experiencia?": "mal",
}

POSITIVE_HINTS = {
    "excelente",
    "amable",
    "amables",
    "buena",
    "bueno",
    "bien",
    "genial",
    "agradable",
    "recomendado",
    "espectacular",
    "rico",
    "justo",
    "impecable",
}

NEGATIVE_HINTS = {
    "colapsada",
    "colapsado",
    "olvidaban",
    "olvidaron",
    "ignorar",
    "ignoraba",
    "ignoraban",
    "desaparecieron",
    "malo",
    "mala",
    "pesimo",
    "pésimo",
    "lento",
    "lenta",
    "demora",
    "tard",
    "caro",
    "frio",
    "fría",
    "prepotente",
    "desorganizada",
    "descuidada",
    "mal",
    "nunca",
    "rogar",
    "queja",
    "error",
}

NEUTRAL_HINTS = {
    "normal",
    "regular",
    "promedio",
    "aceptable",
    "correcta",
    "correcto",
    "ok",
    "a secas",
}

DEFAULT_CATALOG = [
    {"tag": "mesero", "category": "atencion", "synonyms": "mesero|mozo|mozos|personal|recepcion|garzon|garzón", "enabled": 1},
    {"tag": "tiempo_espera", "category": "atencion", "synonyms": "espera|demora|tard|lento|rapidez|agil|ágil", "enabled": 1},
    {"tag": "cobro", "category": "atencion", "synonyms": "cuenta|cobraron|cobro|vuelto|pago|recargo", "enabled": 1},
    {"tag": "comida", "category": "comida", "synonyms": "comida|sabor|plato|ceviche|lomo|aji|ají|arroz|postre|menu|menú", "enabled": 1},
    {"tag": "ceviche", "category": "comida", "synonyms": "ceviche|tiradito|leche_de_tigre", "enabled": 1},
    {"tag": "lomo_saltado", "category": "comida", "synonyms": "lomo_saltado|lomo", "enabled": 1},
    {"tag": "tacu_tacu", "category": "comida", "synonyms": "tacu_tacu", "enabled": 1},
    {"tag": "anticuchos", "category": "comida", "synonyms": "anticucho|anticuchos", "enabled": 1},
    {"tag": "aji_de_gallina", "category": "comida", "synonyms": "aji_de_gallina|ají_de_gallina", "enabled": 1},
    {"tag": "postres", "category": "comida", "synonyms": "postre|postres|picarones|alfajores|suspiro|mazamorra", "enabled": 1},
    {"tag": "bebidas", "category": "comida", "synonyms": "chicha|pisco|bebida|bebidas|cafe|café", "enabled": 1},
    {"tag": "salado", "category": "comida", "synonyms": "salado|salada|sal", "enabled": 1},
    {"tag": "precio", "category": "precio_calidad", "synonyms": "precio|caro|economico|económico|valor|porciones|costo", "enabled": 1},
    {"tag": "ambiente", "category": "ambiente", "synonyms": "ambiente|musica|música|ruido|iluminacion|iluminación|terraza|decoracion|decoración|olor", "enabled": 1},
    {"tag": "experiencia", "category": "experiencia_general", "synonyms": "experiencia|reserva|aforo|trato|servicio|general", "enabled": 1},
]

LLM_MODEL = "gemini-2.5-flash"
_LLM_CLIENT: ChatGoogleGenerativeAI | None = None


def _ensure_storage_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TAG_CATALOG_PATH.exists():
        pd.DataFrame(DEFAULT_CATALOG).to_csv(TAG_CATALOG_PATH, index=False)

    if not TAG_EVENTS_PATH.exists():
        pd.DataFrame(
            columns=[
                "processed_at",
                "source_row_hash",
                "id_cliente",
                "telefono",
                "question_id",
                "question",
                "text",
                "tag",
                "category",
                "polarity",
                "origin",
            ]
        ).to_csv(TAG_EVENTS_PATH, index=False)

    if not TAG_INDEX_PATH.exists():
        pd.DataFrame(
            columns=["source_row_hash", "processed_at", "id_cliente", "telefono"]
        ).to_csv(TAG_INDEX_PATH, index=False)

    if not TAG_CATALOG_PENDING_PATH.exists():
        pd.DataFrame(columns=["tag", "first_seen_at", "example_text"]).to_csv(
            TAG_CATALOG_PENDING_PATH, index=False
        )

    if not TAG_LLM_CACHE_PATH.exists():
        pd.DataFrame(
            columns=["cache_key", "created_at", "items_json"]
        ).to_csv(TAG_LLM_CACHE_PATH, index=False)


def _get_llm_client() -> ChatGoogleGenerativeAI:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return _LLM_CLIENT


def _parse_llm_json(content) -> dict:
    if isinstance(content, str):
        text = content.strip()
    else:
        text = str(content).strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        object_match = re.search(r"\{[\s\S]*\}", text)
        if object_match:
            return json.loads(object_match.group(0))
        raise


def _normalize_tag_name(tag: str) -> str:
    return re.sub(r"\s+", "_", str(tag).strip().lower())


def _build_catalog_signature(catalog: pd.DataFrame) -> str:
    values = catalog[["tag", "category"]].astype(str).to_dict(orient="records")
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _build_llm_prompt(catalog: pd.DataFrame) -> tuple[str, str]:
    allowed_tags = catalog[["tag", "category", "synonyms"]].fillna("").to_dict(orient="records")
    system = """Eres un analista de experiencia para restaurantes peruanos.
Tu trabajo es etiquetar comentarios de clientes en tags y polaridad.

Devuelve ÚNICAMENTE JSON válido (sin markdown) con este formato:
{
  "items": [
    {
      "tag": "<tag_en_snake_case>",
      "category": "<atencion|comida|precio_calidad|ambiente|experiencia_general>",
      "polarity": "<bien|mal|neutral>"
    }
  ]
}

Reglas:
- Puedes devolver múltiples items si el comentario menciona varios aspectos.
- Si una mención no tiene suficiente señal positiva ni negativa, usa "neutral".
- Usa preferentemente tags del catálogo proporcionado.
- Si detectas un tema relevante no presente en catálogo, crea un tag nuevo corto en snake_case y categoría adecuada.
- No incluyas explicación, solo el JSON."""
    human = (
        f"CATÁLOGO DE TAGS:\n{json.dumps(allowed_tags, ensure_ascii=False)}\n\n"
        "Analiza este bloque completo de feedback del cliente (6 preguntas con 6 respuestas) y clasifícalo."
    )
    return system, human


def _load_llm_cache() -> pd.DataFrame:
    return pd.read_csv(TAG_LLM_CACHE_PATH)


def _save_llm_cache(cache_df: pd.DataFrame) -> None:
    cache_df.to_csv(TAG_LLM_CACHE_PATH, index=False)


def _llm_evaluate_row(
    row_context: str,
    row_hash: str,
    catalog: pd.DataFrame,
    cache_df: pd.DataFrame,
) -> tuple[list[dict], pd.DataFrame]:
    catalog_signature = _build_catalog_signature(catalog)
    cache_key_raw = f"{catalog_signature}||row||{row_hash}||{row_context}"
    cache_key = hashlib.sha256(cache_key_raw.encode("utf-8")).hexdigest()

    if not cache_df.empty:
        existing = cache_df[cache_df["cache_key"] == cache_key]
        if not existing.empty:
            cached_json = existing.iloc[-1]["items_json"]
            return json.loads(str(cached_json)), cache_df

    llm = _get_llm_client()
    system_prompt, human_template = _build_llm_prompt(catalog)
    human_content = f"{human_template}\n\nFEEDBACK CLIENTE:\n{row_context}"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    parsed = _parse_llm_json(response.content)
    items = parsed.get("items", []) if isinstance(parsed, dict) else []

    row = {
        "cache_key": cache_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items_json": json.dumps(items, ensure_ascii=False),
    }
    cache_df = pd.concat([cache_df, pd.DataFrame([row])], ignore_index=True)
    return items, cache_df


def _build_row_qa_context(row: pd.Series) -> str:
    qa_lines = []
    for idx, question in enumerate(QUESTION_COLUMNS, start=1):
        answer = str(row.get(question, "")).strip()
        qa_lines.append(f"P{idx}: {question}\nR{idx}: {answer}")
    return "\n\n".join(qa_lines)


def reset_incremental_tag_storage() -> dict:
    _ensure_storage_files()

    pd.DataFrame(
        columns=[
            "processed_at",
            "source_row_hash",
            "id_cliente",
            "telefono",
            "question_id",
            "question",
            "text",
            "tag",
            "category",
            "polarity",
            "origin",
        ]
    ).to_csv(TAG_EVENTS_PATH, index=False)

    pd.DataFrame(
        columns=["source_row_hash", "processed_at", "id_cliente", "telefono"]
    ).to_csv(TAG_INDEX_PATH, index=False)

    pd.DataFrame(columns=["tag", "first_seen_at", "example_text"]).to_csv(
        TAG_CATALOG_PENDING_PATH, index=False
    )

    pd.DataFrame(columns=["cache_key", "created_at", "items_json"]).to_csv(
        TAG_LLM_CACHE_PATH, index=False
    )

    logger.warning("[reset_incremental_tag_storage] Reset completo ejecutado (events/index/pending/llm_cache vaciados)")
    return {
        "events_cleared": True,
        "index_cleared": True,
        "pending_cleared": True,
        "llm_cache_cleared": True,
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _compute_row_hash(row: pd.Series) -> str:
    raw = "||".join(str(row.get(col, "")) for col in ["ID_Cliente", "numero_tel_cliente", *QUESTION_COLUMNS])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(TAG_CATALOG_PATH)
    if "enabled" not in catalog.columns:
        catalog["enabled"] = 1
    return catalog[catalog["enabled"] == 1].copy()


def _extract_bracket_tags(text: str) -> list[str]:
    found = re.findall(r"\[([^\]]+)\]", text)
    return [re.sub(r"\s+", "_", t.strip().lower()) for t in found if t.strip()]


def _detect_tags(text: str, catalog: pd.DataFrame) -> list[tuple[str, str, str]]:
    normalized = _normalize_text(text)
    tags: list[tuple[str, str, str]] = []

    for _, row in catalog.iterrows():
        tag = str(row["tag"]).strip().lower()
        category = str(row["category"]).strip().lower()
        synonyms = str(row.get("synonyms", ""))
        candidates = [tag] + [s.strip().lower() for s in synonyms.split("|") if s.strip()]

        for candidate in candidates:
            if candidate and candidate in normalized:
                tags.append((tag, category, "catalog"))
                break

    unique_tags: list[tuple[str, str, str]] = []
    seen = set()
    for item in tags:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            unique_tags.append(item)

    return unique_tags


def _infer_polarity(text: str, question: str) -> str:
    normalized = _normalize_text(text)

    if re.search(r"\bcolapsad[oa]s?\b", normalized):
        return "mal"
    if re.search(r"\bolvid\w+\b", normalized):
        return "mal"
    if re.search(r"\bignor\w+\b", normalized):
        return "mal"
    if re.search(r"\bdesapareci\w+\b", normalized):
        return "mal"

    has_pos = any(h in normalized for h in POSITIVE_HINTS)
    has_neg = any(h in normalized for h in NEGATIVE_HINTS)
    has_neutral = any(h in normalized for h in NEUTRAL_HINTS)

    if has_pos and has_neg:
        return "neutral"

    if has_neg:
        return "mal"
    if has_pos:
        return "bien"
    if has_neutral:
        return "neutral"

    return QUESTION_DEFAULT_POLARITY.get(question, "bien")


def _heuristic_detect_with_polarity(text: str, question: str, catalog: pd.DataFrame) -> list[dict]:
    detected = _detect_tags(text, catalog)
    if not detected:
        return []

    polarity = _infer_polarity(text, question)
    return [
        {
            "tag": tag,
            "category": category,
            "polarity": polarity,
            "origin": "fallback",
        }
        for tag, category, _origin in detected
    ]


def _normalize_llm_items(items: list[dict], catalog: pd.DataFrame) -> list[dict]:
    if not items:
        return []

    catalog_map: dict[str, tuple[str, str]] = {}
    for _, row in catalog.iterrows():
        canonical_tag = _normalize_tag_name(str(row["tag"]))
        category = str(row["category"]).strip().lower()
        catalog_map[canonical_tag] = (canonical_tag, category)

        synonyms = str(row.get("synonyms", ""))
        for synonym in [item.strip().lower() for item in synonyms.split("|") if item.strip()]:
            normalized_synonym = _normalize_tag_name(synonym)
            catalog_map[normalized_synonym] = (canonical_tag, category)

    normalized = []
    seen = set()
    for item in items:
        raw_tag = _normalize_tag_name(item.get("tag", ""))
        if not raw_tag or raw_tag not in catalog_map:
            continue

        canonical_tag, category = catalog_map[raw_tag]

        polarity = str(item.get("polarity", "neutral")).strip().lower()
        if polarity not in {"bien", "mal", "neutral"}:
            polarity = "neutral"

        unique_key = (canonical_tag, category, polarity)
        if unique_key in seen:
            continue
        seen.add(unique_key)

        normalized.append(
            {
                "tag": canonical_tag,
                "category": category,
                "polarity": polarity,
                "origin": "catalog",
            }
        )

    return normalized


def _fallback_items_from_questions(row: pd.Series, catalog: pd.DataFrame) -> list[dict]:
    items: list[dict] = []
    for question in QUESTION_COLUMNS:
        text = str(row.get(question, "")).strip()
        if not text:
            continue
        items.extend(_heuristic_detect_with_polarity(text, question, catalog))
    return items


def _dedup_tag_items(items: list[dict]) -> list[dict]:
    unique_items: list[dict] = []
    seen = set()
    for item in items:
        key = (item["tag"], item["category"], item["polarity"])
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def _append_pending_tags(new_tags: list[tuple[str, str]]) -> None:
    if not new_tags:
        return

    pending = pd.read_csv(TAG_CATALOG_PENDING_PATH)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    existing = set(pending["tag"].astype(str).str.lower().tolist()) if not pending.empty else set()

    for tag, example_text in new_tags:
        if tag.lower() not in existing:
            rows.append({"tag": tag, "first_seen_at": now, "example_text": example_text[:220]})
            existing.add(tag.lower())

    if rows:
        pd.concat([pending, pd.DataFrame(rows)], ignore_index=True).to_csv(
            TAG_CATALOG_PENDING_PATH, index=False
        )


def _build_events_for_row(
    row: pd.Series,
    row_hash: str,
    catalog: pd.DataFrame,
    processed_at: str,
    cache_df: pd.DataFrame,
) -> tuple[list[dict], list[tuple[str, str]]]:
    id_cliente = row.get("ID_Cliente", "")
    telefono = str(row.get("numero_tel_cliente", "")).strip()

    row_events: list[dict] = []
    row_pending: list[tuple[str, str]] = []

    row_context = _build_row_qa_context(row)

    try:
        llm_items, cache_df = _llm_evaluate_row(
            row_context=row_context,
            row_hash=row_hash,
            catalog=catalog,
            cache_df=cache_df,
        )
        evaluated_items = _normalize_llm_items(llm_items, catalog)
    except Exception as exc:
        logger.exception("[_build_events_for_row] Fallback heurístico por error LLM (fila completa): %s", exc)
        evaluated_items = _fallback_items_from_questions(row, catalog)

    if not evaluated_items:
        evaluated_items = _fallback_items_from_questions(row, catalog)

    for item in _dedup_tag_items(evaluated_items):

        tag = item["tag"]
        category = item["category"]
        polarity = item["polarity"]
        origin = item["origin"]
        row_events.append(
            {
                "processed_at": processed_at,
                "source_row_hash": row_hash,
                "id_cliente": id_cliente,
                "telefono": telefono,
                "question_id": 0,
                "question": "FULL_ROW",
                "text": row_context,
                "tag": tag,
                "category": category,
                "polarity": polarity,
                "origin": origin,
            }
        )
        if origin in {"free", "llm_new"}:
            row_pending.append((tag, row_context))

    return row_events, row_pending, cache_df


def process_incremental_tags(df: pd.DataFrame) -> dict:
    _ensure_storage_files()
    catalog = _load_catalog()

    index_df = pd.read_csv(TAG_INDEX_PATH)
    llm_cache_df = _load_llm_cache()
    processed_hashes = set(index_df["source_row_hash"].astype(str).tolist()) if not index_df.empty else set()

    logger.info(
        "[process_incremental_tags] Inicio | filas_csv=%s | hashes_ya_procesados=%s | tags_catalogo=%s | llm_cache=%s",
        len(df),
        len(processed_hashes),
        len(catalog),
        len(llm_cache_df),
    )

    events_to_append = []
    index_to_append = []
    pending_to_append: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        row_hash = _compute_row_hash(row)
        if row_hash in processed_hashes:
            continue

        now = datetime.now(timezone.utc).isoformat()
        row_events, row_pending, llm_cache_df = _build_events_for_row(
            row=row,
            row_hash=row_hash,
            catalog=catalog,
            processed_at=now,
            cache_df=llm_cache_df,
        )
        events_to_append.extend(row_events)
        pending_to_append.extend(row_pending)

        index_to_append.append(
            {
                "source_row_hash": row_hash,
                "processed_at": now,
                "id_cliente": row.get("ID_Cliente", ""),
                "telefono": str(row.get("numero_tel_cliente", "")).strip(),
            }
        )

    if events_to_append:
        events_df = pd.read_csv(TAG_EVENTS_PATH)
        pd.concat([events_df, pd.DataFrame(events_to_append)], ignore_index=True).to_csv(
            TAG_EVENTS_PATH, index=False
        )

    if index_to_append:
        pd.concat([index_df, pd.DataFrame(index_to_append)], ignore_index=True).to_csv(
            TAG_INDEX_PATH, index=False
        )

    _save_llm_cache(llm_cache_df)

    _append_pending_tags(pending_to_append)

    logger.info(
        "[process_incremental_tags] Fin | nuevas_filas=%s | nuevos_eventos=%s | nuevos_tags_pending=%s",
        len(index_to_append),
        len(events_to_append),
        len(pending_to_append),
    )

    return {
        "new_rows_processed": len(index_to_append),
        "new_tag_events": len(events_to_append),
    }


def _resolve_customer_tag_polarity(polarities: pd.Series) -> str:
    vals = {str(p).strip().lower() for p in polarities}
    if "mal" in vals:
        return "mal"
    if "bien" in vals:
        return "bien"
    if "neutral" in vals:
        return "neutral"
    return "neutral"


def _category_score(df: pd.DataFrame, category: str) -> float:
    subset = df[df["category"] == category]
    if subset.empty:
        return 5.0
    positive = int((subset["polarity"] == "bien").sum())
    neutral = int((subset["polarity"] == "neutral").sum())
    total = len(subset)
    return round(((positive + 0.5 * neutral) / total) * 10, 1)


def _collect_tag_comments(events: pd.DataFrame, tag: str, polarity: str, max_items: int = 3) -> list[str]:
    subset = events[(events["tag"] == tag) & (events["polarity"] == polarity)]
    if subset.empty:
        subset = events[events["tag"] == tag]
    return subset["text"].astype(str).drop_duplicates().head(max_items).tolist()


def _select_top_signals(tag_insights: list[dict]) -> tuple[list[dict], list[dict]]:
    strengths = sorted(
        [item for item in tag_insights if item.get("balance", 0) > 0],
        key=lambda item: (item.get("balance", 0), item.get("bien", 0)),
        reverse=True,
    )[:3]

    weaknesses = sorted(
        [item for item in tag_insights if item.get("balance", 0) < 0],
        key=lambda item: (item.get("balance", 0), -item.get("mal", 0)),
    )[:3]

    if not weaknesses:
        weaknesses = sorted(
            [item for item in tag_insights if item.get("mal", 0) > 0],
            key=lambda item: (item.get("mal", 0), -item.get("balance", 0)),
            reverse=True,
        )[:3]

    return strengths, weaknesses


def _build_executive_context(events: pd.DataFrame, tag_insights: list[dict], total_customers: int) -> dict:
    strengths, weaknesses = _select_top_signals(tag_insights)

    strength_data = []
    for item in strengths:
        strength_data.append(
            {
                "tag": item.get("tag", ""),
                "balance": item.get("balance", 0),
                "bien": item.get("bien", 0),
                "neutral": item.get("neutral", 0),
                "mal": item.get("mal", 0),
                "comentarios": _collect_tag_comments(events, item.get("tag", ""), "bien"),
            }
        )

    weakness_data = []
    for item in weaknesses:
        weakness_data.append(
            {
                "tag": item.get("tag", ""),
                "balance": item.get("balance", 0),
                "bien": item.get("bien", 0),
                "neutral": item.get("neutral", 0),
                "mal": item.get("mal", 0),
                "comentarios": _collect_tag_comments(events, item.get("tag", ""), "mal"),
            }
        )

    return {
        "total_customers": total_customers,
        "fortalezas_top3": strength_data,
        "debilidades_top3": weakness_data,
    }


def _to_text_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _fallback_executive_summary(context: dict) -> dict:
    strengths = context.get("fortalezas_top3", [])
    weaknesses = context.get("debilidades_top3", [])

    strength_tags = [item.get("tag", "") for item in strengths]
    weakness_tags = [item.get("tag", "") for item in weaknesses]

    fortalezas = [
        f"{item.get('tag', '')}: {item.get('bien', 0)} clientes positivos (balance {item.get('balance', 0):+d})."
        for item in strengths
    ]
    debilidades = [
        f"{item.get('tag', '')}: {item.get('mal', 0)} clientes negativos (balance {item.get('balance', 0):+d})."
        for item in weaknesses
    ]

    plan_mejora = [
        f"Atender de inmediato el tag '{weakness_tags[0]}' con plan operativo semanal." if weakness_tags else "Definir foco principal de mejora semanal.",
        "Capacitar al equipo usando comentarios reales como casos de entrenamiento.",
        "Medir semanalmente la variación del balance por tag para validar impacto.",
    ]

    resumen = (
        f"Se analizaron {context.get('total_customers', 0)} clientes. "
        f"Fortalezas: {', '.join(strength_tags) if strength_tags else 'sin datos'}. "
        f"Debilidades: {', '.join(weakness_tags) if weakness_tags else 'sin datos'}."
    )

    return {
        "resumen": resumen,
        "fortalezas": fortalezas,
        "debilidades": debilidades,
        "plan_mejora": plan_mejora,
        "fortaleza_principal": strength_tags[0] if strength_tags else "sin hallazgos",
        "recomendacion_principal": plan_mejora[0],
    }


def _run_executive_summary_langgraph(context: dict) -> dict:
    llm = _get_llm_client()

    def prepare_context(state: ExecutiveSummaryState) -> dict:
        return {"context_json": state["context_json"]}

    def generate_summary(state: ExecutiveSummaryState) -> dict:
        system = """Eres consultor senior de restaurantes.
Recibirás top 3 fortalezas y top 3 debilidades con comentarios reales de clientes.

Devuelve ÚNICAMENTE JSON válido con este formato:
{
  "resumen": "<2-4 oraciones ejecutivas>",
  "fortalezas": ["<fortaleza 1>", "<fortaleza 2>", "<fortaleza 3>"],
  "debilidades": ["<debilidad 1>", "<debilidad 2>", "<debilidad 3>"],
  "plan_mejora": ["<acción 1>", "<acción 2>", "<acción 3>"],
  "fortaleza_principal": "<principal fortaleza>",
  "recomendacion_principal": "<acción de impacto inmediato>"
}

Reglas:
- Usa solo la evidencia del contexto.
- No inventes tags, cifras ni comentarios.
- Escribe en español claro y accionable para el dueño."""
        human = f"CONTEXTO:\n{state['context_json']}"
        try:
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            parsed = _parse_llm_json(response.content)
            return {"summary": parsed}
        except Exception as exc:
            return {"error": str(exc), "summary": {}}

    graph = StateGraph(ExecutiveSummaryState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("generate_summary", generate_summary)
    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "generate_summary")
    graph.add_edge("generate_summary", END)

    app = graph.compile()
    initial_state: ExecutiveSummaryState = {
        "context_json": json.dumps(context, ensure_ascii=False),
        "summary": {},
        "error": "",
    }
    result = app.invoke(initial_state)

    if result.get("error"):
        logger.error("[_run_executive_summary_langgraph] Error: %s", result.get("error"))
        return _fallback_executive_summary(context)

    parsed = result.get("summary", {}) if isinstance(result, dict) else {}
    if not parsed:
        return _fallback_executive_summary(context)

    normalized = {
        "resumen": str(parsed.get("resumen", "")).strip(),
        "fortalezas": _to_text_list(parsed.get("fortalezas")),
        "debilidades": _to_text_list(parsed.get("debilidades")),
        "plan_mejora": _to_text_list(parsed.get("plan_mejora")),
        "fortaleza_principal": str(parsed.get("fortaleza_principal", "")).strip(),
        "recomendacion_principal": str(parsed.get("recomendacion_principal", "")).strip(),
    }

    if not normalized["resumen"]:
        return _fallback_executive_summary(context)
    if not normalized["fortalezas"]:
        normalized["fortalezas"] = _fallback_executive_summary(context).get("fortalezas", [])
    if not normalized["debilidades"]:
        normalized["debilidades"] = _fallback_executive_summary(context).get("debilidades", [])
    if not normalized["plan_mejora"]:
        normalized["plan_mejora"] = _fallback_executive_summary(context).get("plan_mejora", [])
    if not normalized["fortaleza_principal"]:
        normalized["fortaleza_principal"] = normalized["fortalezas"][0] if normalized["fortalezas"] else "sin hallazgos"
    if not normalized["recomendacion_principal"]:
        normalized["recomendacion_principal"] = normalized["plan_mejora"][0] if normalized["plan_mejora"] else "sin recomendación"

    return normalized


def build_dashboard_payload_from_tags() -> TagAnalysisResult:
    _ensure_storage_files()

    events = pd.read_csv(TAG_EVENTS_PATH)
    logger.info("[build_dashboard_payload_from_tags] Construyendo payload | eventos=%s", len(events))

    if events.empty:
        return {
            "sentiment_scores": {},
            "key_themes": {},
            "summary": {},
            "error": "No hay eventos de tags aún. Agrega respuestas en Data/data.csv y actualiza el análisis.",
            "metadata": {"events": 0, "customers": 0, "tag_insights": []},
        }

    events["telefono"] = events["telefono"].astype(str)
    events["tag"] = events["tag"].astype(str)
    events["polarity"] = events["polarity"].astype(str)
    events["category"] = events["category"].astype(str)

    customer_tag = (
        events.groupby(["telefono", "tag", "category"], as_index=False)["polarity"]
        .agg(_resolve_customer_tag_polarity)
    )

    atencion_score = _category_score(customer_tag, "atencion")
    comida_score = _category_score(customer_tag, "comida")
    precio_score = _category_score(customer_tag, "precio_calidad")
    ambiente_score = _category_score(customer_tag, "ambiente")
    experiencia_score = _category_score(customer_tag, "experiencia_general")

    customer_sentiment = (
        customer_tag.groupby("telefono", as_index=False)["polarity"]
        .agg(_resolve_customer_tag_polarity)
    )

    positives = int((customer_tag["polarity"] == "bien").sum())
    negatives = int((customer_tag["polarity"] == "mal").sum())
    neutrals = int((customer_tag["polarity"] == "neutral").sum())

    tag_insights = _build_tag_insights(events, customer_tag)

    negative_counts = (
        customer_tag[customer_tag["polarity"] == "mal"]
        .groupby("tag")
        .size()
        .sort_values(ascending=False)
    )
    positive_counts = (
        customer_tag[customer_tag["polarity"] == "bien"]
        .groupby("tag")
        .size()
        .sort_values(ascending=False)
    )

    praise_candidates = sorted(
        [item for item in tag_insights if item["balance"] > 0],
        key=lambda x: (x["balance"], x["bien"]),
        reverse=True,
    )
    complaint_candidates = sorted(
        [item for item in tag_insights if item["mal"] > 0],
        key=lambda x: (x["mal"], -x["balance"]),
        reverse=True,
    )

    top_praises = [
        _format_tag_item(item["tag"], item["bien"], item["balance"])
        for item in praise_candidates[:5]
    ]
    top_complaints = [
        _format_tag_item(item["tag"], item["mal"], item["balance"])
        for item in complaint_candidates[:5]
    ]

    worst_tag = negative_counts.index[0] if not negative_counts.empty else "sin hallazgos negativos"
    best_tag = positive_counts.index[0] if not positive_counts.empty else "sin hallazgos positivos"

    executive_context = _build_executive_context(
        events=events,
        tag_insights=tag_insights,
        total_customers=len(customer_sentiment),
    )
    executive_summary = _run_executive_summary_langgraph(executive_context)
    resumen = executive_summary.get("resumen", "")

    return {
        "sentiment_scores": {
            "atencion": atencion_score,
            "comida": comida_score,
            "precio_calidad": precio_score,
            "ambiente": ambiente_score,
            "experiencia_general": experiencia_score,
            "positivos": positives,
            "negativos": negatives,
            "neutros": neutrals,
        },
        "key_themes": {
            "top_praises": top_praises,
            "top_complaints": top_complaints,
            "top_dishes": [
                _format_tag_item(item["tag"], item["bien"], item["balance"])
                for item in praise_candidates
                if item.get("category") == "comida" and item["balance"] > 0
            ][:3],
            "improvement_areas": top_complaints[:3],
        },
        "summary": {
            "resumen": resumen,
            "fortaleza_principal": executive_summary.get("fortaleza_principal", best_tag),
            "recomendacion_principal": executive_summary.get(
                "recomendacion_principal",
                f"Atacar el tag '{worst_tag}' con un plan operativo semanal.",
            ),
            "fortalezas": executive_summary.get("fortalezas", []),
            "debilidades": executive_summary.get("debilidades", []),
            "plan_mejora": executive_summary.get("plan_mejora", []),
        },
        "error": "",
        "metadata": {
            "events": int(len(events)),
            "customers": int(customer_sentiment["telefono"].nunique()),
            "tag_insights": tag_insights,
        },
    }


def run_incremental_tag_pipeline(df: pd.DataFrame) -> TagAnalysisResult:
    try:
        executed_at = datetime.now(timezone.utc).isoformat()
        logger.info("[run_incremental_tag_pipeline] Trigger recalculo | executed_at=%s", executed_at)

        process_stats = process_incremental_tags(df)
        payload = build_dashboard_payload_from_tags()
        payload["metadata"].update(process_stats)
        payload["metadata"]["executed_at"] = executed_at

        logger.info(
            "[run_incremental_tag_pipeline] OK | nuevas_filas=%s | nuevos_eventos=%s | total_eventos=%s",
            process_stats.get("new_rows_processed", 0),
            process_stats.get("new_tag_events", 0),
            payload.get("metadata", {}).get("events", 0),
        )
        return payload
    except Exception as exc:
        logger.exception("[run_incremental_tag_pipeline] Error")
        return {
            "sentiment_scores": {},
            "key_themes": {},
            "summary": {},
            "error": str(exc),
            "metadata": {},
        }
