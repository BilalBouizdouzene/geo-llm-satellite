import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
RAW_DATA_DIR = Path("/content/drive/MyDrive/STL_TEST/geo-llm-satellite/data/raw/classification")
PROCESSED_DATA_DIR = Path("/content/drive/MyDrive/STL_TEST/geo-llm-satellite/data/processed/clss2")
TARGET_SIZE = (224, 224)  # Taille standard pour les CNN

def preprocess_and_save_images():
    if not RAW_DATA_DIR.exists():
        print(f"Erreur : {RAW_DATA_DIR} n'existe pas.")
        return

    print(f"Prétraitement des images depuis {RAW_DATA_DIR}...")

    # Supprimer les anciennes données si elles existent
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    # Parcours des classes
    for class_dir in RAW_DATA_DIR.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            output_class_dir = PROCESSED_DATA_DIR / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            # Parcours des images
            for img_path in tqdm(list(class_dir.glob("*.jpg")), desc=f"Traitement de {class_name}"):
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")  # Assure-toi que l'image est en RGB
                        img = img.resize(TARGET_SIZE)
                        img.save(output_class_dir / img_path.name)
                except Exception as e:
                    print(f"Erreur sur {img_path.name} : {e}")

    print(f"Images traitées enregistrées dans {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    preprocess_and_save_images()


import numpy as np

def colorize_mask(mask, class_colors):
    """Colorise le masque de segmentation"""
    try:
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_idx, color in class_colors.items():
            if cls_idx < len(class_colors):
                color_mask[mask == cls_idx] = color
        return color_mask
    except Exception as e:
        import streamlit as st
        st.error(f"Error in colorize_mask: {e}")
        raise