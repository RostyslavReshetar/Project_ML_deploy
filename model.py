import os
import click
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)


class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += ((outputs > 0.5) == labels).sum().item()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += ((outputs > 0.5) == labels).sum().item()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)


def split_dataset(class_1_dir, class_0_dir, split_ratios=(0.8, 0.1, 0.1)):
    class_1_images = [os.path.join(class_1_dir, img) for img in os.listdir(class_1_dir)]
    class_0_images = [os.path.join(class_0_dir, img) for img in os.listdir(class_0_dir)]

    train_1, temp_1 = train_test_split(class_1_images, test_size=split_ratios[1] + split_ratios[2], random_state=42)
    val_1, test_1 = train_test_split(temp_1, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42)

    train_0, temp_0 = train_test_split(class_0_images, test_size=split_ratios[1] + split_ratios[2], random_state=42)
    val_0, test_0 = train_test_split(temp_0, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42)

    train_split = [(img, 1) for img in train_1] + [(img, 0) for img in train_0]
    val_split = [(img, 1) for img in val_1] + [(img, 0) for img in val_0]
    test_split = [(img, 1) for img in test_1] + [(img, 0) for img in test_0]

    return train_split, val_split, test_split

def predict_images(model, image_paths, transform, device):
    model.eval()
    predictions = {}

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image).item()
            predictions[image_path] = (1 if output > 0.5 else 0, output)

    return predictions


@click.command()
@click.option('--dataset_folder', type=str, default='dataset', help='dataset folder')
def main(dataset_folder):
    # Paths
    class_1_dir = os.path.join(dataset_folder, "1_dolphin")
    class_0_dir = os.path.join(dataset_folder, "other_dolphins")

    # Splitting the dataset
    print("Splitting dataset into training, validation, and test sets...")
    train_split, val_split, test_split = split_dataset(class_1_dir, class_0_dir)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and Dataloaders
    train_dataset = CustomDataset(train_split, transform=transform)
    val_dataset = CustomDataset(val_split, transform=transform)
    test_dataset = CustomDataset(test_split, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Check the first sample in the train dataset
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}, Label: {label}")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassificationModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "dolphin_binary_classification.pth")

    # Test a few images
    print("Testing individual images...")
    test_images = ["dataset/1_dolphin/SB15-1593-279.JPG", "dataset/other_dolphins/0J4A1878.JPG"]
    predictions = predict_images(model, test_images, transform, device)
    for image_path, (label, confidence) in predictions.items():
        class_name = "Dolphin #1" if label == 1 else "Not Dolphin #1"
        print(f"Image: {image_path}, Prediction: {class_name}, Confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()