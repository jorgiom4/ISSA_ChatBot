# my_ss_chatbot/hello_streamlit.py
import streamlit as st

st.title("¡Hola, Streamlit!")
st.write("Esta es una prueba básica para verificar que Streamlit funciona.")
st.success("¡Streamlit está funcionando correctamente!")

# Opcional: Añadir un widget interactivo
name = st.text_input("¿Cuál es tu nombre?")
if name:
    st.write(f"¡Hola, {name}!")