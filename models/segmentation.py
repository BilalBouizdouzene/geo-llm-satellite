import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from albumentations import Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

#  CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10

DATA_DIR = "/content/segmentation"
CLASS_DICT_PATH = os.path.join(DATA_DIR, "class_dict.csv")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

#  MAPPING DES COULEURS RGB VERS CLASSES
def get_class_mapping(class_dict_path):
    df = pd.read_csv(class_dict_path)
    return {(row['r'], row['g'], row['b']): i for i, row in df.iterrows()}

CLASS_MAP = get_class_mapping(CLASS_DICT_PATH)

#  VERSION VECTORIZÉE DU CONVERTISSEUR DE MASQUES
def rgb_to_mask(mask, class_map):
    h, w, _ = mask.shape
    mask_encoded = np.zeros((h * w), dtype=np.uint8)
    flat_mask = mask.reshape(-1, 3)
    color_array = np.array(list(class_map.keys()))
    class_array = np.array(list(class_map.values()))
    for i, color in enumerate(color_array):
        matches = np.all(flat_mask == color, axis=1)
        mask_encoded[matches] = class_array[i]
    return mask_encoded.reshape(h, w)

#  DATASET PERSONNALISÉ
class SegmentationDataset(Dataset):
    def __init__(self, df, class_map, root_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.class_map = class_map
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.root_dir, row['sat_image_path'])
        mask_path = os.path.join(self.root_dir, row['mask_path'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = rgb_to_mask(mask, self.class_map)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.long()

#  TRANSFORMATIONS
transform = Compose([
    Resize(IMAGE_SIZE, IMAGE_SIZE),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

#  CHARGEMENT DES DONNÉES
df = pd.read_csv(METADATA_PATH)
df.dropna(subset=['sat_image_path', 'mask_path'], inplace=True)

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'valid']

train_dataset = SegmentationDataset(train_df, CLASS_MAP, DATA_DIR, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = SegmentationDataset(val_df, CLASS_MAP, DATA_DIR, transforms=transform) if not val_df.empty else None
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True) if val_dataset else None

#  MODÈLE (U-Net avec backbone léger)
model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet",
                 in_channels=3, classes=NUM_CLASSES).to(DEVICE)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

#  OPTIMISATION
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

#  ENTRAÎNEMENT
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {total_loss:.4f}")

    #  ÉVALUATION TOUS LES 2 ÉPOCHES
    if val_loader and (epoch + 1) % 2 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == masks).sum().item()
                total += masks.numel()
        acc = 100 * correct / total
        print(f" Validation Accuracy: {acc:.2f}%")


os.makedirs("/content/segmentation_models", exist_ok=True)
local_path = "/content/segmentation_models/segmentation_model.pth"
torch.save(model.state_dict(), local_path)

print(" Modèle sauvegardé avec succès !")