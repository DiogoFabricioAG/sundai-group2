# 🍽️ RestaurantAI

Plataforma de inteligencia artificial para analizar el feedback de comensales de un restaurante y convertirlo en **insights accionables** y **oportunidades de negocio**.

---

## ¿Qué hace?

A partir de un CSV con preguntas y respuestas de comensales, la plataforma ofrece tres módulos principales:

| Módulo | Descripción |
|---|---|
| **📊 Dashboard de Análisis** | Visualiza scores de sentimiento, temas principales, platos destacados y un resumen ejecutivo generado por IA |
| **🎯 Generador de Leads** | Identifica y puntúa clientes con potencial de retorno o fidelización, con acciones de CRM sugeridas |
| **🤖 Chatbot Restaurante** | Módulo conversacional para encuestas de satisfacción y generación de códigos de descuento |

---

## Tecnologías

- **[Streamlit](https://streamlit.io/)** — Interfaz web interactiva (multi-página)
- **[LangChain](https://python.langchain.com/)** — Orquestación de llamadas al LLM
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Agentes con grafos de estado
- **[Gemini / Google](https://ai.google.dev/)** — Modelo de lenguaje para análisis de texto
- **[Plotly](https://plotly.com/python/)** — Visualizaciones interactivas
- **[Pandas](https://pandas.pydata.org/)** — Procesamiento del dataset

---

## Estructura del Proyecto

```
hackSundAI/
├── app.py                            # Página Home (punto de entrada de Streamlit)
├── pages/                            # Vistas Streamlit (debe estar en la raíz)
│   ├── 1_Dashboard.py                # Vista de Dashboard con gráficos
│   ├── 2_Leads.py                    # Vista de Leads generados por IA
│   ├── 3_Marketing.py                # Vista de Marketing
│   └── 4_Chatbot.py                  # Vista Chatbot Restaurante
├── Backend/
│   ├── Dashboard/
│   │   └── dashboard_agent.py        # Agente LangGraph para análisis de feedback
│   ├── Leads/
│   │   └── leads_agent.py            # Agente para generación de leads
│   ├── Marketing/
│   │   └── marketing_agent.py        # Agente de marketing
│   └── Chatbot/
│       └── chatbot.py                # Lógica del chatbot restaurante
├── Frontend/
│   └── utils/
│       └── data_loader.py            # Carga y preprocesamiento del CSV
├── Data/
│   └── data.csv                      # Dataset de feedback de comensales
├── .env.example                      # Plantilla de variables de entorno
├── requirements.txt                  # Dependencias del proyecto
└── README.md
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd hackSundAI
```

### 2. Crear entorno virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita el archivo `.env` y agrega tu API Key de Google:

```env
GOOGLE_API_KEY=AIza...
```

> Obtén tu API Key en [aistudio.google.com](https://aistudio.google.com/app/apikey)

---

## Uso

```bash
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`. Desde la página **Home** puedes navegar a:

- **Dashboard** — para ver el análisis de sentimiento y temas
- **Leads** — para ver los leads identificados y exportarlos
- **Marketing** — para ver las promociones generadas
- **Chatbot** — para interactuar con el chatbot

---

## Dataset

El archivo `Data/data.csv` contiene respuestas de comensales a 6 preguntas:

| Columna | Descripción |
|---|---|
| `ID_Cliente` | Identificador del cliente |
| `numero_tel_cliente` | Teléfono de contacto |
| `costo_del_consumo` | Monto consumido en S/. |
| `¿Qué mejorarías de la atención?` | Feedback sobre atención |
| `¿Qué te pareció la atención?` | Percepción general de la atención |
| `¿Qué te gustó más de la comida?` | Platos y sabores destacados |
| `¿Qué opina sobre la relación entre calidad y precio?` | Percepción de valor |
| `¿Qué te gustó mas del ambiente?` | Aspectos positivos del ambiente |
| `¿Qué es lo que cambiarías de la experiencia?` | Oportunidades de mejora |

---

## Agentes LangGraph

### `dashboard_agent.py`

Grafo de 3 nodos ejecutados en secuencia (sin checkpointer):

```
analyze_sentiment → extract_themes → build_summary → END
```

**Estado:** `DashboardState` — `raw_data`, `sentiment_scores`, `key_themes`, `summary`, `error`

| Nodo | Salida | Descripción |
|---|---|---|
| `analyze_sentiment` | `sentiment_scores` | Scores 0–10 para `atencion`, `comida`, `precio_calidad`, `ambiente`, `experiencia_general` + conteo de clientes `positivos / negativos / neutros` |
| `extract_themes` | `key_themes` | Listas de `top_praises`, `top_complaints`, `top_dishes` y `improvement_areas` (5 / 5 / 3 / 3 ítems) |
| `build_summary` | `summary` | Resumen ejecutivo con `resumen`, `fortaleza_principal` y `recomendacion_principal` |

**Función pública:** `run_dashboard_agent(data_text: str) -> DashboardState`

**Modelo:** `gemini-3-flash-preview` · temperature=0

---

### `leads_agent.py`

Grafo de 3 nodos con **Human-in-the-Loop (HITL)** y `MemorySaver` checkpointer:

```
categorize_clients → generate_promotions → human_review → END
```

**Estado:** `LeadsState` — `raw_data`, `customer_data`, `spending_threshold`, `categorized_leads`, `promotions`, `approved_leads`, `error`

| Nodo | Salida | Descripción |
|---|---|---|
| `categorize_clients` | `categorized_leads` | Filtra clientes por `spending_threshold`, luego llama al LLM en **batch** (`max_concurrency=10`) para asignar categoría: `alto_valor`, `retencion`, `recurrente` o `referidor` |
| `generate_promotions` | `promotions` | Genera mensajes de WhatsApp personalizados en **batch** para cada lead categorizado |
| `human_review` | `approved_leads` | Pausa el grafo con `interrupt()` y devuelve las promociones al frontend; se reanuda con `Command(resume=leads_aprobados)` |

**Función auxiliar:** `regenerate_single_promotion(lead, instructions, feedback) -> str`  
Regenera el mensaje de un lead individual siguiendo las instrucciones del revisor (llamada desde el frontend durante la fase HITL).

**Modelo:** `gemini-3-flash-preview` · temperature=0  
**Checkpointer:** `MemorySaver` compartido en memoria (persiste durante la sesión del servidor)

---

## Contribuir

1. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
2. Realiza tus cambios y haz commit
3. Abre un Pull Request

---

## Licencia

MIT
