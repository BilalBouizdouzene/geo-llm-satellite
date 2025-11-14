# 🌍 GeoLLM - Analyse Géographique par Intelligence Artificielle

## 👥 Auteurs & Universités
**Bilal Bouizdouzene**  
**Salma Lakehal**  

🎓 **Double Diplôme :**  
- Université Sidi Mohamed Ben Abdellah (USMBA – Maroc)  
- Université Sorbonne Paris Nord (USPN – France)  

---

## 📸 Exemples de Résultats (Screenshots)
Voici quelques exemples de sorties que GeoLLM peut produire :

### 🖼️ Exemple 1 : Classification + Segmentation
![a](/assets/cap1.png)
![a](/assets/cap2.png)
![a](/assets/cap3.png)


### 🧠 Exemple 2 : Analyse LLM
![a](/assets/cap4.png)

---

## 📖 Table des Matières
- [Description du Projet](#description-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du Projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure des Modèles](#structure-des-modèles)
- [Déploiement](#déploiement)
- [Développement](#développement)
- [Contributions](#contributions)
- [Support](#support)

## 🚀 Description du Projet

GeoLLM est une application web avancée d'analyse d'images satellites qui combine la puissance du deep learning et des modèles de langage (LLM) pour fournir des analyses géographiques détaillées.

### 🎯 Cas d'Usage
- **Urbanisme et aménagement du territoire**
- **Monitoring environnemental**
- **Agriculture de précision**
- **Gestion des ressources naturelles**
- **Recherche géographique et cartographie**

## ✨ Fonctionnalités

### 🔍 Analyse Multi-Niveaux
- **Classification** : 10 classes de paysages différentes
- **Segmentation** : 7 types de zones géographiques
- **Analyse contextuelle** : Explications détaillées générées par IA

### 🎨 Interface Utilisateur
- **Design moderne** avec interface Streamlit responsive
- **Visualisations interactives** des résultats
- **Téléchargement** des rapports d'analyse
- **Support multi-formats** d'images (PNG, JPG, JPEG)

## 🏗️ Architecture du Projet
![a](/assets/arch.png)
```
geollm-app/
├── app.py
├── config/
│   ├── settings.py
│   └── __init__.py
├── models/
│   ├── classifier.py
│   ├── segmentation.py
│   ├── llm_model.py
│   └── __init__.py
├── utils/
│   ├── image_processing.py
│   ├── visualization.py
│   ├── analysis.py
│   ├── analysis_helpers.py
│   └── __init__.py
├── ui/
│   ├── components.py
│   ├── styles.py
│   └── __init__.py
├── weights/
│   ├── classifier1.pth
│   └── segmentation_model.pth
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### Prérequis
- **Python** 3.8 ou supérieur
- **pip** (gestionnaire de paquets Python)

### 📥 Installation Pas à Pas

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/geollm-app.git
cd geollm-app
```

2. **Créer un environnement virtuel**
```bash
# Linux / Mac
python -m venv geollm-env
source geollm-env/bin/activate

# Windows
python -m venv geollm-env
geollm-env\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Vérifier la structure des fichiers**
Assurez-vous que les fichiers suivants sont dans `weights/` :
- `classifier1.pth`
- `segmentation_model.pth`

## 🚀 Utilisation

### Lancement de l'Application
```bash
streamlit run app.py
```
Application disponible sur : http://localhost:8501

### 📝 Guide d'Utilisation
- Upload d'une image
- Choisir :
  - Classification + Segmentation + LLM
  - Classification seule
  - Segmentation seule
- Cliquer **"🚀 Analyser l'Image"**
- Résultats :
  - Classe prédite + confiance
  - Masque de segmentation
  - Analyse LLM détaillée
- Télécharger le rapport

## 🎯 Types d'Analyse Disponibles

### Classification (10 classes)
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

### Segmentation (7 classes)
urban_land, agriculture_land, rangeland, forest_land, water, barren_land, unknown

## 🤖 Structure des Modèles

### Modèle de Classification
- ResNet-18 modifié  
- Entrée : 224×224  
- Sortie : 10 classes  
- Poids : `weights/classifier1.pth`

### Modèle de Segmentation
- U-Net + MobileNetV2  
- Entrée : 256×256  
- Sortie : masque 7 classes  
- Poids : `weights/segmentation_model.pth`

### Modèle de Langage (LLM)
- FLAN-T5 base ou small  
- Téléchargement automatique  

## 🌐 Déploiement

### Déploiement Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Déploiement Streamlit Cloud
- Push GitHub  
- Connecter share.streamlit.io  
- Déployer  

## 🛠️ Développement

### Config (config/)
settings.py : config centrale

### Models (models/)
classifier.py  
segmentation.py  
llm_model.py  

### Utils (utils/)
image_processing.py  
visualization.py  
analysis.py  
analysis_helpers.py  

### UI (ui/)
components.py  
styles.py  

## 🤝 Contributions
```bash
git checkout -b feature/new-feature
```
Créez une PR !

## 🎯 Roadmap
- Multi-bandes  
- Analyse temporelle  
- Export GeoJSON  
- API cartographiques  

## 🆘 Support et Dépannage

### Erreurs de Modèles
Vérifier `weights/`

### Problèmes de mémoire
- Réduire la taille des images  
- Utiliser FLAN-T5-small  

### Module introuvable
```bash
pip install -r requirements.txt
```

## 📋 Requirements
```
streamlit>=1.28.0
torch>=1.9.0
torchvision>=0.10.0
segmentation-models-pytorch>=0.2.0
transformers>=4.20.0
Pillow>=9.0.0
numpy>=1.21.0
pandas>=1.3.0
```

## 🔄 Mise à Jour
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---
 
