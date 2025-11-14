import torch
import numpy as np
from utils.analysis_helpers import (
    get_detailed_class_mapping, 
    analyze_spatial_distribution,
    create_detailed_zones_description,
    clean_generated_text,
    is_valid_detailed_analysis,
    generate_detailed_fallback_analysis
)

def create_geo_prompt(class_label, mask, model_name, class_names):
    """Crée le prompt pour l'analyse géographique"""
    class_mapping = get_detailed_class_mapping(class_names)
    spatial_analysis = analyze_spatial_distribution(mask, class_mapping)
    zone_descriptions = create_detailed_zones_description(spatial_analysis)
    
    if spatial_analysis:
        dominant_zone = max(spatial_analysis.items(), key=lambda x: x[1]["percentage"])
        dominant_name = dominant_zone[1]["name"]
        dominant_percentage = dominant_zone[1]["percentage"]
    else:
        dominant_name = "Zone non identifiée"
        dominant_percentage = 0
        
    significant_zones = len([z for z in spatial_analysis.values() if z["percentage"] > 5])
    zones_list = "\n".join([f"- {desc}" for desc in zone_descriptions[:4]])
    available_classes = [info["name"] for info in spatial_analysis.values()] if spatial_analysis else []
    
    if "base" in model_name or "large" in model_name:
        prompt = f"""ANALYSE GÉOGRAPHIQUE DÉTAILLÉE D'IMAGE SATELLITE

DONNÉES DE CLASSIFICATION:
- Classe principale détectée: {class_label}
- Zone dominante: {dominant_name} ({dominant_percentage}%)
- Nombre de zones significatives: {significant_zones}

TYPES DE TERRAIN IDENTIFIÉS:
{zones_list}

Classes disponibles: {', '.join(available_classes) if available_classes else 'Aucune'}

MISSION: Produisez une analyse géographique experte de 180-220 mots incluant:

1. IDENTIFICATION DU PAYSAGE:
   - Type de terrain et géomorphologie
   - Caractérisation de l'occupation du sol

2. ANALYSE DES ZONES DÉTECTÉES:
   - Description détaillée de chaque zone nommée
   - Interactions spatiales entre les différentes zones
   - Patterns de distribution géographique

3. CONTEXTE ENVIRONNEMENTAL:
   - Végétation et biodiversité observée
   - Ressources hydriques identifiées
   - Conditions topographiques

4. ACTIVITÉS HUMAINES:
   - Infrastructures et aménagements
   - Usage des sols et activités économiques
   - Pression anthropique sur l'environnement

5. IMPLICATIONS TERRITORIALES:
   - Enjeux d'aménagement du territoire
   - Risques environnementaux potentiels
   - Recommandations de gestion

Utilisez les noms exacts des zones détectées et leurs pourcentages."""
    else:
        prompt = f"""ANALYSE SATELLITE:

ZONES DÉTECTÉES:
{zones_list}

Zone dominante: {dominant_name} ({dominant_percentage}%)
Classes: {', '.join(available_classes) if available_classes else 'Aucune'}

ANALYSE REQUISE (120 mots):
1. Décrivez chaque zone nommée précisément
2. Type de paysage et terrain
3. Végétation et activités humaines observées
4. Caractéristiques environnementales
5. Interactions entre les différentes zones

Utilisez les noms exacts des zones et leurs pourcentages."""
    
    return prompt, spatial_analysis

def generate_geo_analysis(class_label, mask, tokenizer, llm, model_name, class_names, max_retries=0):
    """Génère l'analyse géographique détaillée"""
    import streamlit as st
    
    try:
        class_mapping = get_detailed_class_mapping(class_names)
        spatial_analysis = analyze_spatial_distribution(mask, class_mapping)
        
        prompt, spatial_data = create_geo_prompt(class_label, mask, model_name, class_names)
        
        for attempt in range(max_retries):
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=400,
                    padding=True
                ).to(DEVICE)
                
                gen_params = {
                    "max_new_tokens": 250 if "base" in model_name or "large" in model_name else 150,
                    "min_length": 120 if "base" in model_name or "large" in model_name else 80,
                    "temperature": 0.7 if "base" in model_name or "large" in model_name else 0.8,
                    "top_p": 0.9 if "base" in model_name or "large" in model_name else 0.95,
                    "do_sample": True,
                    "num_beams": 3 if "base" in model_name or "large" in model_name else 2,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 2,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
                
                with torch.no_grad():
                    outputs = llm.generate(**inputs, **gen_params)
                    
                generated_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                analysis = clean_generated_text(generated_text, prompt)
                min_words = 80 if "base" in model_name else 50
                
                if is_valid_detailed_analysis(analysis, spatial_analysis, min_words):
                    return analysis, spatial_analysis
                    
            except Exception as e:
                continue
                
        fallback_analysis = generate_detailed_fallback_analysis(class_label, mask, spatial_analysis)
        return fallback_analysis, spatial_analysis
        
    except Exception as e:
        st.error(f"Erreur critique dans la génération: {e}")
        return f"Erreur lors de l'analyse géographique détaillée: {str(e)}", {}