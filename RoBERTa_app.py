import streamlit as st
import requests
import json

# Configura la página de Streamlit
st.set_page_config(page_title="Inferencia de Estrés con RoBERTa", layout="wide")

# URL de la API
API_URL = "http://127.0.0.1:8000/predict/"  # Cambia la IP y el puerto si es necesario

def get_prediction(text):
    # Prepara los datos para la solicitud POST
    data = {"text": text}
    response = requests.post(API_URL, json=data)
    return response.json()

def main():
    st.title("Detector de Estrés con RoBERTa")
    text = st.text_area("Ingrese el texto para analizar:", height=150)
    
    if st.button("Analizar"):
        if text:
            result = get_prediction(text)
            st.write("Resultado de la Predicción:")
            st.json(result)
        else:
            st.error("Por favor, ingrese algún texto para analizar.")

if __name__ == "__main__":
    main()
