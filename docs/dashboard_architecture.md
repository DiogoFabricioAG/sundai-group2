# Dashboard Architecture (RestaurantAI)

## 1) Objetivo

Este documento describe el flujo técnico actual del módulo de Dashboard: cómo se procesa `Data/data.csv`, cómo intervienen LangChain/LangGraph, qué modelo se usa, cómo funciona el sistema de tags y qué muestra finalmente la UI.

---

## 2) Vista General

El Dashboard tiene dos caminos:

1. **Camino legacy (texto completo + LLM)**
   - Implementado en `Backend/Dashboard/dashboard_agent.py`.
   - Usa un grafo de 3 nodos (`analyze_sentiment -> extract_themes -> build_summary`).
   - Se mantiene para compatibilidad, pero no es el flujo principal de la pantalla actual.

2. **Camino actual (incremental por tags + LLM + persistencia CSV)**
   - Implementado en `Backend/Dashboard/tag_analytics.py`.
   - Es el flujo usado por la página `pages/1_Dashboard.py` mediante `run_dashboard_agent_from_df(...)`.
   - Procesa incrementalmente, persiste eventos y alimenta gráficos/KPIs con datos estructurados.

---

## 3) Arquitectura de LangChain

### Modelo

- **Proveedor/modelo:** `gemini-2.5-flash`
- **Cliente:** `ChatGoogleGenerativeAI`
- **Temperatura:** `0`

### Uso de LangChain en el flujo actual

Se usa LangChain para dos tareas:

1. **Clasificación por tags/polaridad del feedback por fila completa**
   - Entrada: bloque con 6 preguntas + 6 respuestas de una fila.
   - Salida esperada: JSON con lista de items (`tag`, `category`, `polarity`).
   - Se parsea robustamente con fallback de limpieza de markdown/JSON.

2. **Generación de Resumen Ejecutivo**
   - Entrada: top 3 fortalezas y top 3 debilidades con comentarios reales.
   - Salida esperada: JSON con `resumen`, `fortalezas`, `debilidades`, `plan_mejora`, `fortaleza_principal`, `recomendacion_principal`.

---

## 4) Arquitectura de LangGraph

## 4.1 Grafo legacy (`create_dashboard_agent`)

Archivo: `Backend/Dashboard/dashboard_agent.py`

Grafo secuencial:

- `analyze_sentiment`
- `extract_themes`
- `build_summary`
- `END`

Se usa sobre texto crudo completo y devuelve campos de dashboard en JSON.

## 4.2 Grafo actual de Resumen Ejecutivo (`_run_executive_summary_langgraph`)

Archivo: `Backend/Dashboard/tag_analytics.py`

Grafo secuencial:

- `prepare_context`
- `generate_summary`
- `END`

Este grafo no clasifica tags; consume evidencia ya estructurada (top señales + comentarios) y genera narrativa/plan ejecutivo.

---

## 5) Sistema de Tags (actual)

## 5.1 Catálogo fijo

El sistema usa **tags fijos** (`DEFAULT_CATALOG`) y no debería crear tags libres en producción de métricas.

Categorías:

- `atencion`
- `comida`
- `precio_calidad`
- `ambiente`
- `experiencia_general`

Tags destacados de comida:

- `comida`
- `ceviche`
- `lomo_saltado`
- `tacu_tacu`
- `anticuchos`
- `aji_de_gallina`
- `postres`
- `bebidas`
- `salado`

## 5.2 Polaridad

Valores permitidos:

- `bien`
- `neutral`
- `mal`

Regla de agregación por cliente/tag:

- Prioridad: `mal > bien > neutral`

---

## 6) Persistencia incremental (CSV)

Todos los artefactos viven en `Data/`:

- `tag_events.csv`: eventos de tags clasificados.
- `tag_index.csv`: hashes de filas ya procesadas.
- `tag_catalog.csv`: catálogo activo de tags/sinónimos.
- `tag_catalog_pending.csv`: pendiente/revisión (si aplica).
- `tag_llm_cache.csv`: caché de respuestas LLM por fila.

Botón de reproceso total (`♻️ Reprocesar todo`) limpia:

- `tag_events`
- `tag_index`
- `tag_catalog_pending`
- `tag_llm_cache`

---

## 7) Pipeline final (end-to-end)

1. Streamlit carga `Data/data.csv`.
2. `run_dashboard_agent_from_df(df)` dispara `run_incremental_tag_pipeline(df)`.
3. Para cada fila nueva (hash no procesado):
   - Se construye un bloque con `P1..P6 / R1..R6`.
   - Gemini clasifica tags/polaridad (1 llamada por fila).
   - Se normaliza contra catálogo fijo.
   - Se persiste evento en `tag_events.csv`.
4. Se agregan métricas para:
   - scores por categoría,
   - distribución dona (`positivos/neutros/negativos`),
   - top tags por polaridad,
   - balance por tag,
   - temas principales.
5. Se construye contexto ejecutivo (top 3 fortalezas/debilidades + comentarios).
6. LangGraph genera el resumen ejecutivo con plan de mejora.
7. `pages/1_Dashboard.py` renderiza:
   - KPIs,
   - barra de scores,
   - dona,
   - sección tags accionables (con toggle Top Malos/Top Buenos),
   - temas principales,
   - resumen ejecutivo en 3 recuadros.

---

## 8) Visualización actual (UI)

Archivo: `pages/1_Dashboard.py`

- **Top 5 tags por polaridad**: ordenados por `total_mentions = bien + neutral + mal`.
- **Balance por tag**: toggle entre
  - `Top Malos` (comentarios malos en hover),
  - `Top Buenos` (comentarios buenos en hover).
- **Resumen Ejecutivo**:
  - Resumen + fortalezas (top 3)
  - debilidades (top 3)
  - plan de mejora (top 3) + recomendación principal

---

## 9) Consideraciones de integración

- Si el análisis parece “igual”, revisar:
  - `nuevas_filas` y `nuevos_eventos` en metadata/logs.
- Si se necesita forzar recomputo:
  - usar `♻️ Reprocesar todo`.
- Si cambia catálogo de tags:
  - conviene reprocesar para mantener coherencia histórica.

---

## 10) Archivos clave

- `Backend/Dashboard/dashboard_agent.py`
- `Backend/Dashboard/tag_analytics.py`
- `pages/1_Dashboard.py`
- `Data/tag_events.csv`
- `Data/tag_index.csv`
- `Data/tag_catalog.csv`
- `Data/tag_llm_cache.csv`
