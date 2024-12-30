import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from dataLoader import VisualAcuityDataset, VisualContrastDataset

# Define the model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=200):  # TinyImageNet has 200 classes
        super(CustomResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training function
def train_network(model, dataloader, criterion, optimizer, num_epochs, label):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{label}] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Main script
if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    # Configuration
    data_directory = './tiny-imagenet-200'  # TinyImageNet root directory
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, criterion, and optimizer
    criterion = nn.CrossEntropyLoss()

    # 1. No Curriculum Training (Baseline)
    print("Training TinyImageNet without curriculum...")
    model = CustomResNet18(num_classes=200).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # TinyImageNet images are 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_directory, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_network(model, train_loader, criterion, optimizer, num_epochs, label="TinyImageNet")

    torch.save(model.state_dict(), './networks/model_tinyimagenet.pth')
    print("TinyImageNet training complete. Model saved!")

    # 2. Curriculum Training
    def train_with_curriculum(model_name, dataset_class, stages, transforms):
        model = CustomResNet18(num_classes=200).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for stage_label, age_months in stages:
            print(f"Training {model_name} - Stage {stage_label} (Months {age_months})...")
            dataset = dataset_class(image_dir=data_directory, age_months=age_months)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_network(model, dataloader, criterion, optimizer, num_epochs, label=f"{model_name} {stage_label}")
        torch.save(model.state_dict(), f'./networks/{model_name}.pth')
        print(f"{model_name} training complete. Model saved!")

    # Define stages for curriculum
    stages = [("1-3", 3), ("6-9", 9), ("12-15", 15)]

    # Visual Acuity Curriculum
    train_with_curriculum("VisualAcuity", VisualAcuityDataset, stages, transforms)

    # Visual Contrast Curriculum
    train_with_curriculum("VisualContrast", VisualContrastDataset, stages, transforms)

    # Both Transformations Curriculum
    print("Training Both Transforms Curriculum...")
    model = CustomResNet18(num_classes=200).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for stage_label, age_months in stages:
        print(f"Training Both Transforms - Stage {stage_label} (Months {age_months})...")
        acuity_dataset = VisualAcuityDataset(image_dir=data_directory, age_months=age_months)
        contrast_dataset = VisualContrastDataset(image_dir=data_directory, age_months=age_months)
        acuity_loader = DataLoader(acuity_dataset, batch_size=batch_size, shuffle=True)
        contrast_loader = DataLoader(contrast_dataset, batch_size=batch_size, shuffle=True)
        for loader in [acuity_loader, contrast_loader]:
            train_network(model, loader, criterion, optimizer, num_epochs, label=f"Both Transforms {stage_label}")
    torch.save(model.state_dict(), './networks/BothTransforms.pth')
    print("Both Transforms training complete. Model saved!")
