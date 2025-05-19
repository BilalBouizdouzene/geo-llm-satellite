import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
RAW_DATA_DIR = Path("data/raw/classefication")
PROCESSED_DATA_DIR = Path("data/processed/classification")
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
