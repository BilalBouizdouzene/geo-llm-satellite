from transformers import T5Tokenizer, T5ForConditionalGeneration
from config.settings import DEVICE
from utils.analysis import create_geo_prompt, generate_geo_analysis

def load_llm():
    """Charge le modèle de langage"""
    import streamlit as st
    
    try:
        try:
            model_name = "google/flan-t5-base"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            llm = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        except Exception as e:
            model_name = "google/flan-t5-small"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            llm = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        return tokenizer, llm, model_name
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        raise