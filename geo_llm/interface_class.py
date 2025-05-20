import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys
import os

# Modifier selon ton modèle sauvegardé
MODEL_PATH = "models/classifier.pth"

# Les classes à prédire
CLASSES = ['cloudy', 'desert', 'green_area', 'water']

# Prétraitement de l'image comme pendant l'entraînement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modèle non trouvé à {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger l'image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Charger le modèle
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    # Prédiction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted.item()]
        print(f"✅ Classe prédite : {predicted_class}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilisation : python scripts/infer_classifier.py chemin/vers/image.jpg")
    else:
        predict(sys.argv[1])
