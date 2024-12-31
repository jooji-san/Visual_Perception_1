import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataLoader import DatasetWithTransformationByAge, TransformationType, SplitType

# Define the model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
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
        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

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
                images, labels = images.to(device), labels.to(device)
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
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    num_classes = 10

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, criterion, and optimizer
    model = CustomResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # my pc
    #dataset_path = os.path.join(os.path.expanduser("~"), "Downloads", "tiny-imagenet-200")
    # cip-pool
    dataset_path = os.path.join("./tiny-imagenet-200")

    transformationTypes = [None, TransformationType.CONTRAST, TransformationType.ACUITY, TransformationType.CONTRAST | TransformationType.ACUITY]
    for transformIndex, transformType  in enumerate(transformationTypes):
        train_dataset_list = []
        val_dataset_list = []
        for age_group in range(3)   :
            train_dataset_age_group_list = []
            val_dataset_age_group_list = []
            for i in range(1, 4):
                train_dataset_age_group_list.append(DatasetWithTransformationByAge(dataset_path, SplitType.TRAIN, age_group * 5 + i, transformType))
                val_dataset_age_group_list.append(DatasetWithTransformationByAge(dataset_path, SplitType.VAL, age_group * 5  + i, transformType))

            # randomize in the group
            train_dataset_age_group = torch.utils.data.ConcatDataset(train_dataset_age_group_list)
            indices = torch.randperm(len(train_dataset_age_group)).tolist()
            shuffled_train_dataset_dataset = torch.utils.data.Subset(train_dataset_age_group, indices)

            train_dataset_list.append(shuffled_train_dataset_dataset)

            val_dataset_age_group = torch.utils.data.ConcatDataset(val_dataset_age_group_list)
            indices = torch.randperm(len(val_dataset_age_group)).tolist()
            shuffled_val_dataset_dataset = torch.utils.data.Subset(val_dataset_age_group, indices)

            val_dataset_list.append(shuffled_val_dataset_dataset)

        train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
        val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # Train the network
        train_losses, val_losses = train_network(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)

        # Save the trained model
        os.makedirs('./networks', exist_ok=True)
        torch.save(model.state_dict(), f'./networks/model_tinyimagenet_{str(transformType)}.pth')
        torch.save(model.state_dict(), f'./networks/model_tinyimagenet_{transformIndex}.pth')

        # Save loss data for later analysis
        with open(f'training_validation_loss_{str(transformType)}.json', "w") as f:
            import json
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

        with open(f'training_validation_loss_{transformIndex}.json', "w") as f:
            import json
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

        print("Training complete. Model and loss data saved!")
