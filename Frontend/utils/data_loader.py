import pandas as pd
import streamlit as st
from pathlib import Path

# Frontend/utils/data_loader.py → sube 3 niveles para llegar a la raíz del proyecto
DATA_PATH = Path(__file__).parent.parent.parent / "Data" / "data.csv"

# Mapeo de columnas del CSV
COL_ID = "ID_Cliente"
COL_TEL = "numero_tel_cliente"
COL_CONSUMO = "costo_del_consumo"
COL_MEJORA_ATENCION = "¿Qué mejorarías de la atención?"
COL_ATENCION = "¿Qué te pareció la atención?"
COL_COMIDA = "¿Qué te gustó más de la comida?"
COL_PRECIO_CALIDAD = "¿Qué opina sobre la relación entre calidad y precio?"
COL_AMBIENTE = "¿Qué te gustó mas del ambiente?"
COL_CAMBIO = "¿Qué es lo que cambiarías de la experiencia?"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Carga el dataset de feedback de comensales desde el CSV."""
    return pd.read_csv(DATA_PATH)


def get_data_summary(df: pd.DataFrame) -> dict:
    """Retorna estadísticas básicas del dataset."""
    return {
        "total_customers": len(df),
        "avg_consumption": df[COL_CONSUMO].mean(),
        "max_consumption": df[COL_CONSUMO].max(),
        "min_consumption": df[COL_CONSUMO].min(),
        "total_revenue": df[COL_CONSUMO].sum(),
    }


def df_to_text(df: pd.DataFrame) -> str:
    """Convierte el DataFrame a texto estructurado para procesamiento por LLM."""
    records = []
    for _, row in df.iterrows():
        record = (
            f"Cliente {row[COL_ID]}:\n"
            f"- Consumo: S/. {row[COL_CONSUMO]}\n"
            f"- Qué mejoraría de la atención: {row[COL_MEJORA_ATENCION]}\n"
            f"- Cómo le pareció la atención: {row[COL_ATENCION]}\n"
            f"- Qué le gustó más de la comida: {row[COL_COMIDA]}\n"
            f"- Opinión calidad/precio: {row[COL_PRECIO_CALIDAD]}\n"
            f"- Qué le gustó del ambiente: {row[COL_AMBIENTE]}\n"
            f"- Qué cambiaría: {row[COL_CAMBIO]}"
        )
        records.append(record)
    return "\n\n".join(records)


def get_feedback_blocks(df: pd.DataFrame) -> dict:
    """Retorna un dict {id_cliente: texto_feedback} para uso en regeneración HITL."""
    blocks = {}
    for _, row in df.iterrows():
        blocks[int(row[COL_ID])] = (
            f"Cliente {row[COL_ID]}:\n"
            f"- Consumo: S/. {row[COL_CONSUMO]}\n"
            f"- Qué mejoraría de la atención: {row[COL_MEJORA_ATENCION]}\n"
            f"- Cómo le pareció la atención: {row[COL_ATENCION]}\n"
            f"- Qué le gustó más de la comida: {row[COL_COMIDA]}\n"
            f"- Opinión calidad/precio: {row[COL_PRECIO_CALIDAD]}\n"
            f"- Qué le gustó del ambiente: {row[COL_AMBIENTE]}\n"
            f"- Qué cambiaría: {row[COL_CAMBIO]}"
        )
    return blocks


def get_customer_contact_data(df: pd.DataFrame) -> str:
    """Retorna datos de contacto de clientes en formato texto para el LLM."""
    lines = ["ID_Cliente | Telefono | Consumo (S/.)"]
    for _, row in df.iterrows():
        lines.append(
            f"{row[COL_ID]} | {row[COL_TEL]} | {row[COL_CONSUMO]}"
        )
    return "\n".join(lines)
