import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from medmnist import PathMNIST

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True).expand(3, -1, -1)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = PathMNIST(split="train", download=True, transform=transform)
    test_dataset = PathMNIST(split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(train_dataset.info["label"]) 