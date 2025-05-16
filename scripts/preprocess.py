from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root="data/raw/classification", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
