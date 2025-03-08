import torch
from models.vgg import VGG16
from models.resnet import ResNet18
from models.vit import ViT
from data.dataset import get_data_loaders
from utils.trainer import train_and_test

def main():
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 獲取數據加載器
    train_loader, test_loader, num_classes = get_data_loaders()
    
    # 訓練模型
    models = {
        "VGG16": VGG16(num_classes),
        "ResNet18": ResNet18(num_classes),
        "ViT": ViT(num_classes)
    }
    
    for model_name, model in models.items():
        train_and_test(model, train_loader, device, model_name, epochs=10)

if __name__ == "__main__":
    main() 