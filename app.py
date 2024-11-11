import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from src.text_mining import preprocess_text, extract_significant_concepts

# Cargar las variables de entorno
load_dotenv()

# Configuración de la clave de API de OpenAI
client = OpenAI(api_key=os.getenv('api_key'))

# Configuración de la carpeta para guardar las imágenes
output_folder = os.getenv('output_folder')
os.makedirs(output_folder, exist_ok=True)

# Configuración de la imagen de fondo
background_image_url = os.getenv('url_imagen_fondo')

# Cargar el archivo CSS externo
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Título de la aplicación
st.title('Generador de música a partir de texto')

# Descripción de la aplicación
st.write("""
    Esta aplicación utiliza modelos avanzados de procesamiento de lenguaje natural para analizar el texto que introduces y 
    generar los conceptos más relevantes. ¡Introduce una descripción y descubre los temas claves!
""")

# Campo de texto para que el usuario ingrese su descripción
user_input = st.text_input("Introduce el texto (mínimo 10 caracteres):")

# Botón para analizar el texto
if st.button('Analizar texto'):
    if len(user_input) < 10:
        st.warning("Por favor, introduce una descripción de al menos 10 caracteres.")
    else:
        # Mostrar spinner mientras se realiza el análisis
        with st.spinner('Analizando el texto...'):
            try:
                # Procesamiento del texto
                text = user_input
                preprocessed_text = preprocess_text(text)

                # Extraer conceptos más significativos
                significant_concepts = extract_significant_concepts(preprocessed_text)

                # Mostrar los conceptos más significativos en un contenedor
                st.subheader("Conceptos más significativos:")
                st.text_area("", "\n".join(significant_concepts), height=200)

            except Exception as e:
                st.error(f"Error al procesar el texto: {e}")
