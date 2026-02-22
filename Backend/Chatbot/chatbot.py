import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
mi_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=mi_api_key,
    temperature=0.8,
)

prompt = """
Eres un asistente conversacional que tiene como objetivo principal realizar las siguientes preguntas:
1. ¿Qué mejorarias de la atención?
2. ¿Que te pareció la atención del 1 al 10?
3. ¿Qué mejorarias de la comida?
4. ¿Qué te gustó más del ambiente?
5. ¿Qué es lo que cambiarias?\n
Al finalizar, ofrece un codigo de decuento por responder con el formato ABC-123
"""

st.set_page_config(page_title="Agente Restaurante", layout="centered")
st.title("🍳 Asistente Restaurante")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=f"Eres un asistente de cocina bastante experimentado en cocina peruana. Tu tarea es hacer las preguntas indicadas en el siguiente prompt: {prompt}")
    ]

if "finished" not in st.session_state:
    st.session_state.finished = False

if "current_question" not in st.session_state:
    # Obtener la primera pregunta del agente
    response = llm.invoke(st.session_state.messages + [HumanMessage(content="Comenzamos. Haz tu primera pregunta sobre cocina peruana.")])
    st.session_state.current_question = response.content
    st.session_state.messages.append(AIMessage(content=st.session_state.current_question))

# Mostrar la pregunta actual
# Mostrar historial de chat (se actualiza cada vez que se agrega un mensaje)
st.subheader("Historial")
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    else:
        role = "system"
    with st.chat_message(role):
        st.write(msg.content)

# Mostrar la pregunta actual
st.subheader("Pregunta")
st.write(f"**{st.session_state.current_question}**")

# Formulario para responder (se oculta cuando la conversación termina)
if not st.session_state.finished:
    with st.form("answer_form", clear_on_submit=True):
        user_input = st.text_area("Tu respuesta:", placeholder="Escribe aquí tu respuesta...")
        submitted = st.form_submit_button("Enviar")
        
        if submitted and user_input.strip():
            # Agregar respuesta del usuario al historial
            st.session_state.messages.append(HumanMessage(content=user_input))
            
            # Obtener siguiente pregunta del agente
            response = llm.invoke(st.session_state.messages)
            next_question = response.content
            
            # Agregar respuesta del agente al historial
            st.session_state.messages.append(AIMessage(content=next_question))
            st.session_state.current_question = next_question

            # Si el agente agradece o genera un ticket, marcar como finalizado
            nq_lower = next_question.lower() if isinstance(next_question, str) else ""
            if not next_question.strip() or "gracias" in nq_lower or "agradec" in nq_lower or "ticket" in nq_lower:
                st.session_state.finished = True

            # Recargar la página
            st.rerun()
else:
    st.info("La encuesta ha finalizado. Gracias por tu colaboración.")
    st.text_area("Encuesta finalizada", value="", disabled=True)

# Botón para reiniciar conversación
if st.button("🔄 Reiniciar"):
    st.session_state.messages = [
        SystemMessage(content=prompt)
    ]
    if "current_question" in st.session_state:
        del st.session_state.current_question
    st.rerun()
