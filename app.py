import streamlit as st
from config.settings import configure_page, DEVICE, PATHS
from models.classifier import load_classification_model, predict_class
from models.segmentation import load_segmentation_model, predict_mask, get_class_dict
from models.llm_model import load_llm, generate_geo_analysis
from utils.image_processing import colorize_mask
from utils.visualization import display_detailed_results
from ui.components import render_header, render_sidebar, render_upload_section
from ui.styles import load_css

def main():
    # Configuration de la page et du CSS
    configure_page()
    load_css()
    
    # Interface utilisateur
    render_header()
    uploaded_file = render_sidebar()
    
    if uploaded_file is None:
        render_upload_section()
        return

    try:
        # Traitement de l'image uploadée
        from PIL import Image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Affichage de l'image
        st.markdown("###  **Image Uploadée**")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=f"Image: {uploaded_file.name}", use_container_width=True)
        
        # Informations sur l'image
        st.markdown(f"""
        <div class="status-card">
            <strong> Informations de l'image:</strong><br>
            • <strong>Nom:</strong> {uploaded_file.name}<br>
            • <strong>Taille:</strong> {image.size[0]} × {image.size[1]} pixels<br>
            • <strong>Format:</strong> {image.format}<br>
            • <strong>Mode:</strong> {image.mode}
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton d'analyse
        if st.button(" **Analyser l'Image**", type="primary", use_container_width=True):
            analyze_image(image, uploaded_file.name)
            
    except Exception as e:
        st.error(f" **Erreur lors du chargement de l'image:** {str(e)}")

def analyze_image(image, filename):
    """Fonction principale pour analyser l'image"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Étape 1: Chargement des modèles
        status_text.text(" Chargement des modèles...")
        progress_bar.progress(10)
        
        clf_model = load_classification_model()
        seg_model = load_segmentation_model()
        class_colors, class_names = get_class_dict()
        
        progress_bar.progress(30)
        
        # Étape 2: Classification
        status_text.text(" Classification en cours...")
        pred_class, confidence = predict_class(image, clf_model)
        progress_bar.progress(50)
        
        # Étape 3: Segmentation
        status_text.text(" Segmentation en cours...")
        seg_mask = predict_mask(image, seg_model)
        progress_bar.progress(70)
        
        # Étape 4: Analyse LLM
        status_text.text(" Génération de l'analyse LLM...")
        tokenizer, llm, model_name = load_llm()
        explanation, spatial_analysis = generate_geo_analysis(
            pred_class, seg_mask, tokenizer, llm, model_name, class_names
        )
        progress_bar.progress(90)
        
        # Affichage du masque de segmentation coloré
        st.markdown("###  **Masque de Segmentation Coloré**")
        color_mask = colorize_mask(seg_mask, class_colors)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Image Originale", use_container_width=True)
        with col2:
            st.image(color_mask, caption="Segmentation Colorée", use_container_width=True)
        
        # Étape 5: Affichage des résultats
        status_text.text(" Génération des résultats...")
        progress_bar.progress(100)
        
        # Nettoyage
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats détaillés
        st.markdown("---")
        st.markdown("##  **Résultats de l'Analyse**")
        display_detailed_results(
            pred_class, confidence, seg_mask, explanation, 
            spatial_analysis, model_name, class_names, class_colors
        )
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f" **Erreur lors de l'analyse:** {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()