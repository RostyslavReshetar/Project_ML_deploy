import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from model import CustomDataset, split_dataset, BinaryClassificationModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_model(model_path, device):
    model = BinaryClassificationModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def compute_metrics(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            # Get probabilities
            outputs = model(images).squeeze()
            # Classify as 1 if probability > 0.5
            predictions = (outputs > 0.5).long()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)

    precision_class_1 = precision_score(y_true, y_pred)
    recall_class_1 = recall_score(y_true, y_pred)
    f1_class_1 = f1_score(y_true, y_pred)

    precision_class_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_class_0 = recall_score(y_true, y_pred, pos_label=0)
    f1_class_0 = f1_score(y_true, y_pred, pos_label=0)

    conf_matrix = confusion_matrix(y_true, y_pred)

    print("Metrics:")
    print(f"Accuracy: {acc:.4f}")

    print(f"Dolphin #1 (class 1):")
    print(f"Precision: {precision_class_1:.4f}")
    print(f"Recall: {recall_class_1:.4f}")
    print(f"F1 Score: {f1_class_1:.4f}")

    print(f"Other dolphins (class 0):")
    print(f"Precision: {precision_class_0:.4f}")
    print(f"Recall: {recall_class_0:.4f}")
    print(f"F1 Score: {f1_class_0:.4f}")

    print("Confusion Matrix:")
    print(conf_matrix)

def evaluate_on_test_set(model, test_loader, device):
    compute_metrics(model, test_loader, device)

def predict_image(model, image_path, transform, device):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    # Prepare for prediction
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get the probability
        output = model(image).item()

    # Model's confidence
    confidence = output
    # Classify
    prediction = 1 if output > 0.5 else 0
    class_name = "Dolphin #1" if prediction == 1 else "Not Dolphin #1"
    return class_name, confidence

def predict_multiple_images(model, image_paths, transform, device):
    predictions = {}
    for image_path in image_paths:
        class_name, confidence = predict_image(model, image_path, transform, device)
        predictions[image_path] = (class_name, confidence)
    return predictions

def main():
    # Path to the saved model
    model_path = "dolphin_binary_classification.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Image transformations
    transform = get_transform()

    # Paths
    dataset_folder = "dataset"
    class_1_dir = os.path.join(dataset_folder, "1_dolphin")
    class_0_dir = os.path.join(dataset_folder, "other_dolphins")

    # Splitting the dataset
    print("Splitting dataset into training, validation, and test sets...")
    train_split, val_split, test_split = split_dataset(class_1_dir, class_0_dir)

    # Test dataset
    test_dataset = CustomDataset(test_split, transform=transform)

    # DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    print("Loading the model...")
    model = load_model(model_path, device)

    # Evaluate on the test set
    print("Evaluating model on test set...")
    evaluate_on_test_set(model, test_loader, device)

    # Some test images
    image_paths = [
        "dataset/1_dolphin/SB15-1595-027.JPG",  # Dolphin #1 image
        "dataset/1_dolphin/SB16-1603-808.JPG",  # Dolphin #1 image
        "dataset/1_dolphin/SB21-1873-304.JPG",  # Dolphin #1 image
        "dataset/other_dolphins/SB24-2056-016.JPG",  # Other dolphin image
        "dataset/other_dolphins/0J4A2913.JPG", # Other dolphin image
    ]

    # Predict for several images
    print("Making predictions for images:")
    predictions = predict_multiple_images(model, image_paths, transform, device)
    for image_path, (class_name, confidence) in predictions.items():
        print(f"Image: {image_path}, Prediction: {class_name}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()