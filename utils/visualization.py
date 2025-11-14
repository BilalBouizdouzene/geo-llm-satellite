import streamlit as st
import numpy as np

def display_detailed_results(pred_class, confidence, seg_mask, explanation, spatial_analysis, model_name, class_names, class_colors):
    """Affiche les résultats détaillés de l'analyse"""
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # Métriques principales
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(spatial_analysis)}</div>
            <div class="metric-label">Zones Détectées</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{model_name.split('/')[-1]}</div>
            <div class="metric-label">Modèle LLM</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Zones détectées
    st.markdown("###  **Zones Détectées et Analysées**")
    sorted_zones = sorted(spatial_analysis.items(), key=lambda x: x[1]["percentage"], reverse=True)
    
    for i, (cls_idx, info) in enumerate(sorted_zones, 1):
        dominance_icon = "" if info["dominance"] == "majeure" else "📍" if info["dominance"] == "significative" else "📌"
        color = class_colors.get(cls_idx, (128, 128, 128))
        
        st.markdown(f"""
        <div class="zone-card">
            <h4>{dominance_icon} {i}. {info['name']} ({info['original_name']})</h4>
            <div style="display: flex; align-items: center; margin: 10px 0;">
                <div style="width: 20px; height: 20px; background-color: rgb({color[0]}, {color[1]}, {color[2]}); border-radius: 3px; margin-right: 10px;"></div>
                <strong>{info['percentage']}%</strong> ({info['pixel_count']:,} pixels) - <em>{info['dominance']}</em>
            </div>
            <p style="margin: 10px 0; color: #666;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analyse géographique
    st.markdown("### **Analyse Géographique Détaillée**")
    st.markdown(f"""
    <div class="analysis-text">
        {explanation}
    </div>
    """, unsafe_allow_html=True)