# my_ss_chatbot/app.py
# Aplicación principal de Streamlit
# Contiene la UI del chat, la orquestación de agentes y la carga de recursos con caché

import streamlit as st
import time # Para simular tiempo de procesamiento (opcional, quitar después)
import torch # Para pasar el dispositivo a las funciones de predicción

# Importar tus módulos de modelos
from models import relevance_model, categorization_model
# Importar tus módulos de agentes (aún no implementados, solo para estructura conceptual)
# from agents import relevance_agent, categorization_agent, rag_agent, response_agent
# Importar tu lógica de RAG (aún no implementada)
# from rag import retriever

# --- Configuración de la Página ---
st.set_page_config(page_title="Chatbot Seguridad Social", layout="centered")
st.title("Asistente de la Seguridad Social")

st.markdown("""
Soy tu asistente virtual de la Seguridad Social y puedo ayudarte a encontrar el servicio o trámite que necesitas, así como resolver tus dudas.
Informarte que, con el fin de mejorar mi entrenamiento y ofrecer un servicio de calidad, las conversaciones se guardan. Por otro lado, no trato con ningún tipo de dato personal (nombre, DNI, teléfono, dirección...), con lo cual no se requiere que los indiques. Puedes leer la política de protección de datos de la Seguridad Social [aquí](https://www.seg-social.es/wps/portal/wss/internet/Politica+de+privacidad).
¿Con qué te puedo ayudar?
""")

# --- Cargar Modelos y Recursos (con caché) ---
# Usa st.cache_resource para cargar modelos y recursos costosos una sola vez
@st.cache_resource
def load_all_resources():
    """Carga todos los modelos y recursos necesarios."""
    print("--- Cargando todos los recursos (modelos, retriever, etc.) con st.cache_resource ---") # Log para depuración

    # Cargar Modelo 1 (Relevancia)
    # Las funciones load_*_model ahora devuelven (tokenizer, model, device)
    relevance_tokenizer, relevance_model_loaded, relevance_device = relevance_model.load_relevance_model()

    # Cargar Modelo 2 (Categorización)
    categorization_tokenizer, categorization_model_loaded, categorization_device = categorization_model.load_categorization_model()

    # Cargar Modelo LLM para respuesta (si usas uno) - Implementar más tarde
    llm_model = None # response_agent.load_llm() # Descomentar e implementar después

    # Inicializar el retriever del RAG - Implementar más tarde
    rag_retriever = None # retriever.initialize_retriever() # Descomentar e implementar después

    print("--- Recursos cargados exitosamente ---")
    # Devolver todos los recursos cargados
    return (relevance_tokenizer, relevance_model_loaded, relevance_device,
            categorization_tokenizer, categorization_model_loaded, categorization_device,
            llm_model, rag_retriever)

# Cargar los recursos al inicio de la aplicación.
# Esta función solo se ejecutará la primera vez que se corre el script
# o si el código dentro de ella cambia.
(relevance_tokenizer, relevance_model_loaded, relevance_device,
 categorization_tokenizer, categorization_model_loaded, categorization_device,
 llm_model, rag_retriever) = load_all_resources()


# --- Inicializar el Historial de Conversación ---
# Usamos st.session_state para mantener el estado a través de las re-ejecuciones del script
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Mostrar Mensajes Anteriores ---
# Itera sobre el historial y muestra cada mensaje
for message in st.session_state.messages:
    # st.chat_message es un componente nativo para interfaces de chat
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Área de Entrada del Usuario ---
# st.chat_input es un componente nativo para la entrada de chat
prompt = st.chat_input("Escribe tu pregunta aquí...")

# --- Lógica al Recibir un Mensaje del Usuario (Orquestación de Agentes Inicial) ---
if prompt:
    # Añadir el mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Iniciar la secuencia de Agentes (simulada por ahora) ---
    with st.chat_message("assistant"):
        with st.spinner("Procesando tu pregunta..."): # Muestra un indicador de carga

            final_response = "Lo siento, ocurrió un error al procesar tu solicitud." # Respuesta por defecto en caso de error

            try:
                # --- Agente 1: Determinar Relevancia ---
                # Llamamos a la función de predicción del módulo models/relevance_model.py
                # Pasamos el prompt y los recursos cargados (tokenizer, model, device)
                print(f"\n--- Ejecutando Agente 1 (Relevancia) para: '{prompt}' ---")
                is_relevant = relevance_model.predict_relevance(
                    prompt,
                    relevance_tokenizer,
                    relevance_model_loaded,
                    relevance_device
                )
                print(f"Resultado Agente 1 (Relevancia): {is_relevant}")

                if not is_relevant:
                    # Si no es relevante, generamos la respuesta final aquí mismo
                    final_response = "Tu pregunta no parece estar relacionada con la Seguridad Social española. Por favor, consulta temas sobre prestaciones, trámites o normativas de la Seguridad Social."
                    print("Pregunta no relevante. Flujo terminado.")
                else:
                    # Si es relevante, pasamos al siguiente agente (simulado)

                    # --- Agente 2: Categorizar la pregunta ---
                    # Llamamos a la función de predicción del módulo models/categorization_model.py
                    # Pasamos el prompt y los recursos cargados (tokenizer, model, device)
                    print(f"\n--- Ejecutando Agente 2 (Categorización) para: '{prompt}' ---")
                    category, subcategory = categorization_model.predict_category(
                        prompt,
                        categorization_tokenizer,
                        categorization_model_loaded,
                        categorization_device
                    )
                    print(f"Resultado Agente 2 (Categorización): {category} / {subcategory}")

                    # Opcional: mostrar categoría/subcategoría detectada para depuración
                    st.info(f"Categoría detectada: **{category}** / **{subcategory}**")

                    # --- Agente 3: Recuperar información del RAG ---
                    # Esta parte aún no está implementada. La simulamos.
                    print("\n--- Simulando Agente 3 (RAG) ---")
                    retrieved_info = None # rag_agent.retrieve_info(...) # Descomentar e implementar después
                    print(f"Resultado Agente 3 (RAG): {retrieved_info}")


                    # --- Agente 4: Generar la respuesta final ---
                    # Esta parte aún no está implementada. La simulamos.
                    print("\n--- Simulando Agente 4 (Respuesta) ---")
                    if not retrieved_info:
                        # Si no hay info del RAG (simulado), damos una respuesta basada en la categorización
                        final_response = (
                            f"Tu pregunta sobre '{prompt}' ha sido categorizada como **'{category} / {subcategory}'**. "
                            "Actualmente, la búsqueda de información específica (RAG) y la generación de la respuesta final están pendientes de implementación."
                        )
                    else:
                        # Si hubiera info del RAG, usaríamos el LLM para generar la respuesta
                        # final_response = response_agent.generate_response(prompt, retrieved_info, llm_model) # Descomentar e implementar después
                        final_response = "Respuesta generada basada en RAG (simulada)." # Respuesta temporal si RAG estuviera implementado

                    print(f"Respuesta final generada (simulada): {final_response}")


            except Exception as e:
                # Capturamos cualquier error que ocurra durante la ejecución de los agentes/modelos
                final_response = f"Lo siento, ocurrió un error interno al procesar tu pregunta: {e}"
                print(f"Error crítico durante la orquestación de agentes: {e}") # Log del error en la terminal
                # Opcional: Mostrar el traceback completo en la terminal
                # import traceback
                # traceback.print_exc()


            # --- Mostrar la respuesta final del bot ---
            # st.markdown(final_response) # Ya se hace al final del bloque 'if prompt'

    # --- Mostrar la respuesta final del bot (fuera del spinner) ---
    # Esto asegura que la respuesta aparece después de que el spinner desaparece
    st.markdown(final_response)


    # Añadir la respuesta del bot al historial
    st.session_state.messages.append({"role": "assistant", "content": final_response})