import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from dataLoader import VisualAcuityDataset, VisualContrastDataset
from torchvision import datasets, transforms

# Define the model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training function
def train_network(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0.0

        # Training loop
        for images, labels in train_dataloader:
            images, labels = images.permute(0, 3, 1, 2).to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Track training loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.permute(0, 3, 1, 2).to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        # Track validation loss
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

# Main script
if __name__ == "__main__":
    # Configuration
    image_directory = './tiny-imagenet-200'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, criterion, and optimizer
    model = CustomResNet18(num_classes=200).to(device)  # TinyImageNet has 200 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data augmentation and transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load TinyImageNet dataset
    train_dataset = datasets.ImageFolder(os.path.join(image_directory, 'train'), transform=transform)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the network
    train_losses, val_losses = train_network(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)

    # Save the trained model
    os.makedirs('./networks', exist_ok=True)
    torch.save(model.state_dict(), './networks/model_tinyimagenet.pth')

    # Save loss data for later analysis
    with open("training_validation_loss.json", "w") as f:
        import json
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    print("Training complete. Model and loss data saved!")
