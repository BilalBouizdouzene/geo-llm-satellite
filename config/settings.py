import os
import torch

# Configuration de base
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemins des modèles
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PATHS = {
    "CLASS_MODEL": os.path.join(BASE_DIR, "models1", "models", "classifier1.pth"),
    "SEG_MODEL": os.path.join(BASE_DIR, "models1", "models", "segmentation_model.pth"),
    "CLASS_DICT": os.path.join(BASE_DIR, "data", "class_dict.csv")
}

# Paramètres des modèles
NUM_CLASSES_CLS = 10
NUM_CLASSES_SEG = 7
IMAGE_SIZE = 256

def configure_page():
    """Configure la page Streamlit"""
    import streamlit as st
    st.set_page_config(
        page_title="GeoLLM - Analyse Géographique par IA",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )