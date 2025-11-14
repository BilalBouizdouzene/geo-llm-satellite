import streamlit as st

def render_header():
    """Affiche l'en-tête de l'application"""
    st.markdown("""
    <div class="header-container">
        <h1> GeoLLM - Analyse Géographique par Intelligence Artificielle</h1>
        <p>Analysez vos images satellites avec des modèles de deep learning avancés</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Affiche la barre latérale et retourne le fichier uploadé"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3> Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("###  **Upload d'Image**")
        uploaded_file = st.file_uploader(
            "Sélectionnez une image satellite",
            type=['png', 'jpg', 'jpeg'],
            help="Formats supportés: PNG, JPG, JPEG"
        )
        
        st.markdown("###  **Paramètres d'Analyse**")
        
        # Options de modèles
        analysis_type = st.selectbox(
            "Type d'analyse",
            ["Classification + Segmentation + LLM", "Classification seule", "Segmentation seule"],
            help="Choisissez le type d'analyse à effectuer"
        )
        
        # Affichage des informations système
        from config.settings import DEVICE, NUM_CLASSES_CLS, NUM_CLASSES_SEG
        st.markdown("###  **Informations Système**")
        st.info(f"**Processeur:** {DEVICE}")
        st.info(f"**Classes de classification:** {NUM_CLASSES_CLS}")
        st.info(f"**Classes de segmentation:** {NUM_CLASSES_SEG}")
        
        # Instructions
        st.markdown("###  **Instructions**")
        st.markdown("""
        1. **Uploadez** une image satellite
        2. **Sélectionnez** le type d'analyse
        3. **Cliquez** sur 'Analyser l'image'
        4. **Consultez** les résultats détaillés
        """)
    
    return uploaded_file

def render_upload_section():
    """Affiche la section d'upload quand aucune image n'est sélectionnée"""
    st.markdown("""
    <div class="upload-container">
        <div class="upload-text">
            <h3> Aucune image sélectionnée</h3>
            <p>Veuillez uploader une image satellite dans la barre latérale pour commencer l'analyse</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Exemple d'images (optionnel)
    st.markdown("###  **Exemples d'Images Supportées**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Images urbaines**\nZones résidentielles, commerciales, infrastructures")
    with col2:
        st.info("**Images rurales**\nTerres agricoles, forêts, pâturages")
    with col3:
        st.info("**Images mixtes**\nCombinations de différents types de terrain")