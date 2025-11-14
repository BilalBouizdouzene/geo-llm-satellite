import numpy as np

def get_detailed_class_mapping(class_names):
    """Crée un mapping détaillé des classes"""
    detailed_descriptions = {
        "urban_land": {
            "description": "Zones urbaines avec bâtiments résidentiels, commerciaux, routes pavées et infrastructures urbaines",
            "keywords": ["urbain", "ville", "bâtiments", "résidentiel", "commercial", "infrastructure", "route"]
        },
        "agriculture_land": {
            "description": "Terres agricoles cultivées avec champs, parcelles agricoles et cultures saisonnières",
            "keywords": ["agriculture", "champs", "cultures", "parcelles", "rural", "cultivé", "ferme"]
        },
        "rangeland": {
            "description": "Pâturages et prairies naturelles, terres de parcours pour le bétail",
            "keywords": ["pâturage", "prairie", "herbeux", "parcours", "bétail", "pastoral", "savane"]
        },
        "forest_land": {
            "description": "Couverture forestière avec arbres matures, canopée dense et espaces boisés",
            "keywords": ["forêt", "arbres", "végétation", "canopée", "boisé", "sylviculture"]
        },
        "water": {
            "description": "Plans d'eau incluant rivières, lacs, étangs, zones humides et cours d'eau",
            "keywords": ["eau", "rivière", "lac", "aquatique", "humide", "hydrographie"]
        },
        "barren_land": {
            "description": "Terres arides et stériles, zones désertiques, affleurements rocheux",
            "keywords": ["aride", "désert", "stérile", "rocheux", "sec", "dénudé", "minéral"]
        },
        "unknown": {
            "description": "Zones non identifiées ou de classification incertaine",
            "keywords": ["inconnu", "indéterminé", "classification", "incertain"]
        }
    }
    
    class_mapping = {
        cls_idx: {
            "name": class_name.replace('_', ' ').title(),
            "original_name": class_name,
            "description": detailed_descriptions.get(class_name, {"description": f"Zone de type {class_name}", "keywords": [class_name.replace('_', ' ')]})["description"],
            "keywords": detailed_descriptions.get(class_name, {"description": f"Zone de type {class_name}", "keywords": [class_name.replace('_', ' ')]})["keywords"]
        } for cls_idx, class_name in class_names.items()
    }
    return class_mapping

def analyze_spatial_distribution(mask, class_mapping):
    """Analyse la distribution spatiale des classes"""
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    detailed_analysis = {
        cls_idx: {
            "name": class_mapping[cls_idx]["name"],
            "original_name": class_mapping[cls_idx]["original_name"],
            "description": class_mapping[cls_idx]["description"],
            "keywords": class_mapping[cls_idx]["keywords"],
            "pixel_count": int(pixel_count),
            "percentage": round((pixel_count / total_pixels) * 100, 2),
            "dominance": "majeure" if (pixel_count / total_pixels) * 100 > 25 else "significative" if (pixel_count / total_pixels) * 100 > 10 else "mineure"
        } for cls_idx, pixel_count in zip(unique, counts)
    }
    return detailed_analysis

def create_detailed_zones_description(spatial_analysis):
    """Crée une description détaillée des zones"""
    descriptions = []
    sorted_zones = sorted(spatial_analysis.items(), key=lambda x: x[1]["percentage"], reverse=True)
    
    for cls_idx, info in sorted_zones:
        if info["percentage"] > 5:
            desc = f"**{info['name']}** ({info['percentage']}%, {info['dominance']}): {info['description']}"
            descriptions.append(desc)
            
    return descriptions

def clean_generated_text(text, original_prompt):
    """Nettoie le texte généré"""
    if original_prompt in text:
        text = text.replace(original_prompt, "").strip()
        
    unwanted_tokens = ['</s>', '<pad>', '<unk>', 'Analysez cette image', 'Fournissez une analyse', 'Réponse en']
    for token in unwanted_tokens:
        text = text.replace(token, '')
        
    text = ' '.join(text.split())
    sentences = text.split('. ')
    
    if len(sentences) > 1 and len(sentences[0].split()) < 5:
        text = '. '.join(sentences[1:])
        
    return text.strip()

def is_valid_detailed_analysis(text, spatial_analysis, min_words=40):
    """Vérifie si l'analyse générée est valide"""
    if not text or len(text.strip()) < 30:
        return False
        
    words = text.split()
    if len(words) < min_words:
        return False
        
    invalid_phrases = [
        "ANALYSE GÉOGRAPHIQUE DÉTAILLÉE", "DONNÉES DE CLASSIFICATION",
        "ZONES IDENTIFIÉES", "MISSION:", "Utilisez les noms exacts"
    ]
    
    for phrase in invalid_phrases:
        if phrase in text:
            return False
            
    zone_names_found = 0
    for cls_idx, info in spatial_analysis.items():
        if info["percentage"] > 5:
            if any(keyword in text.lower() for keyword in info["keywords"]):
                zone_names_found += 1
                
    significant_zones = len([z for z in spatial_analysis.values() if z["percentage"] > 5])
    required_zones = max(1, significant_zones // 2)
    
    if zone_names_found < required_zones:
        return False
        
    detailed_geo_keywords = [
        "terrain", "paysage", "végétation", "zone", "région", "environnement",
        "activité", "infrastructure", "occupation", "territoire", "spatial",
        "géographique", "aménagement", "urbain", "rural", "forestier", "agricole"
    ]
    
    text_lower = text.lower()
    detailed_geo_count = sum(1 for keyword in detailed_geo_keywords if keyword in text_lower)
    has_quantitative_data = any(char in text for char in ["%", "pourcentage", "dominante", "majoritaire"])
    
    return detailed_geo_count >= 2 and (has_quantitative_data or zone_names_found >= 1)

def generate_detailed_fallback_analysis(class_label, mask, spatial_analysis):
    """Génère une analyse de secours détaillée"""
    if not spatial_analysis:
        return "Analyse de segmentation non disponible pour cette image."
        
    sorted_zones = sorted(spatial_analysis.items(), key=lambda x: x[1]["percentage"], reverse=True)
    dominant_zone = sorted_zones[0][1]
    dominant_name = dominant_zone["name"]
    dominant_percentage = dominant_zone["percentage"]
    secondary_zones = [z[1] for z in sorted_zones[1:3] if z[1]["percentage"] > 5]
    
    analysis_parts = [
        f"Cette image satellite révèle un paysage dominé par les **{dominant_name}** ({dominant_percentage}%), caractérisé par {dominant_zone['description'].lower()}."
    ]
    
    if secondary_zones:
        secondary_desc = [f"**{zone['name']}** ({zone['percentage']}%)" for zone in secondary_zones]
        if len(secondary_desc) == 1:
            analysis_parts.append(f"La zone secondaire comprend {secondary_desc[0]}, créant une mosaïque paysagère diversifiée.")
        else:
            analysis_parts.append(f"Les zones secondaires incluent {', '.join(secondary_desc[:-1])} et {secondary_desc[-1]}, formant un ensemble territorial complexe.")
            
    total_zones = len(spatial_analysis)
    if total_zones > 3:
        analysis_parts.append(f"L'analyse de segmentation révèle {total_zones} types de zones différentes, témoignant d'une grande diversité d'occupation du sol.")
        
    original_name = dominant_zone["original_name"]
    if original_name == "urban_land":
        analysis_parts.append("Cette configuration spatiale suggère une forte pression anthropique avec des enjeux importants d'aménagement urbain et de gestion environnementale.")
    elif original_name == "forest_land":
        analysis_parts.append("Ce paysage présente une forte valeur écologique avec des services écosystémiques significatifs pour la biodiversité et la régulation climatique.")
    elif original_name == "agriculture_land":
        analysis_parts.append("Cette zone agricole constitue un espace productif essentiel avec des enjeux de durabilité et de préservation des sols.")
    elif original_name == "water":
        analysis_parts.append("La présence marquée de ressources hydriques confère à ce territoire une importance stratégique pour la gestion de l'eau et les écosystèmes aquatiques.")
    elif original_name == "rangeland":
        analysis_parts.append("Ces pâturages naturels représentent un écosystème pastoral important pour l'élevage extensif et la biodiversité des prairies.")
    elif original_name == "barren_land":
        analysis_parts.append("Ces terres arides présentent des défis particuliers de conservation et peuvent nécessiter des stratégies de restauration écologique.")
        
    analysis_parts.append(f"L'ensemble correspond à la classe principale {class_label}, reflétant les dynamiques territoriales et les interactions complexes entre les différentes composantes du paysage.")
    
    return " ".join(analysis_parts)