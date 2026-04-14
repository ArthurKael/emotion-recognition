import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds


def build_resnet18_grayscale(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    DATA_DIR = "data/fer2013"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    MODEL_PATH = "models/emotion_resnet18_gray.pth"
    CM_PATH = "results/confusion_matrix_resnet18_gray.png"
    CURVE_PATH = "results/learning_curves_resnet18_gray.png"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_SIZE = 96
    BATCH_SIZE = 128
    EPOCHS = 25
    LR = 1e-3
    PATIENCE = 5
    MIN_DELTA = 0.001

    NUM_WORKERS = 0 if os.name == "nt" else 2
    PIN_MEMORY = True if torch.cuda.is_available() else False

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    class_names = train_dataset.classes

    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("Classes:", class_names)
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Using device:", DEVICE)
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("Image size:", IMG_SIZE)
    print("Class counts:", class_counts)
    print("Class weights:", class_weights.cpu().numpy())

    model = build_resnet18_grayscale(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    early_stopper = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    best_acc = 0.0
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {val_loss:.4f} | Test  Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if early_stopper.step(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names
    }, MODEL_PATH)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best test accuracy at saved checkpoint: {best_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - ResNet18 Gray")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(CM_PATH)
    plt.show()

    epochs_ran = len(history["train_loss"])
    x = range(1, epochs_ran + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, history["train_loss"], label="Train Loss")
    plt.plot(x, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve - ResNet18 Gray")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, history["train_acc"], label="Train Acc")
    plt.plot(x, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve - ResNet18 Gray")
    plt.legend()

    plt.tight_layout()
    plt.savefig(CURVE_PATH)
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()