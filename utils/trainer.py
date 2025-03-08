import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_and_test(model, train_loader, device, model_name, epochs=10):
    os.makedirs("checkpoints", exist_ok=True)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, train_accuracies = [], []

    print(f"Training {model_name} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%")

    torch.save(model.state_dict(), f"checkpoints/{model_name}.pth")
    print(f"{model_name} saved!\n")
    
    return train_losses, train_accuracies 