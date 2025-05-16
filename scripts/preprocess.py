from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations appliquées à chaque image
transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Redimensionne les images
    transforms.ToTensor(),                # Convertit en tenseur
    transforms.Normalize(                 # Normalisation (ImageNet)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Chargement du dataset
dataset = datasets.ImageFolder('emplacement', transform=transform)

# Division en train / val / test
from torch.utils.data import random_split

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
