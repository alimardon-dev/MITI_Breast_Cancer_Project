import os
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# =========================
# SEED
# =========================

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================
# CONFIG
# =========================

DATA_DIR = "Data/CNN_Training/CNN_CC_View"
RESULTS_DIR = "RESULTS/CC_ResNet18_without_Augmentation"

IMAGE_SIZE = 224
BATCH_SIZE = 32   # larger batch (lighter model)
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 7

CLASS_NAMES = ["Benign", "Malignant"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASETS
# =========================

class SingleImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.exists(class_path):
                print(f"Missing folder: {class_path}")
                continue

            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)

                if os.path.isfile(file_path) and file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((file_path, label_idx))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class CCFromPatientFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_path = os.path.join(root_dir, class_name)

            for patient_id in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_id)

                if not os.path.isdir(patient_path):
                    continue

                cc_path = os.path.join(patient_path, "CC.png")

                if os.path.exists(cc_path):
                    self.samples.append((cc_path, label_idx))

        print(f"Loaded {len(self.samples)} CC test images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# TRANSFORMS (NO AUG)
# =========================

data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# MODEL (ResNet18)
# =========================

class CCResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(CCResNet18, self).__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# =========================
# TRAINING
# =========================

def train_model(model, dataloaders, criterion, optimizer):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for images, labels in dataloaders[phase]:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0

                    torch.save(
                        model.state_dict(),
                        os.path.join(RESULTS_DIR, "best_cc_resnet18_no_aug.pth")
                    )

                    print("✅ Best model saved")
                else:
                    patience_counter += 1
                    print(f"Early stopping patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\n⏹ Early stopping triggered")
            break

    model.load_state_dict(best_model_weights)
    return model, history


# =========================
# EVALUATION
# =========================

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# =========================
# MAIN
# =========================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Seed: {SEED}")

    datasets = {
        "train": SingleImageFolderDataset(os.path.join(DATA_DIR, "train"), data_transform),
        "val": SingleImageFolderDataset(os.path.join(DATA_DIR, "val"), data_transform),
        "test": CCFromPatientFolderDataset(os.path.join(DATA_DIR, "test"), data_transform)
    }

    dataloaders = {
        "train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=BATCH_SIZE),
        "test": DataLoader(datasets["test"], batch_size=BATCH_SIZE)
    }

    model = CCResNet18().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    model, history = train_model(model, dataloaders, criterion, optimizer)

    training_time = time.time() - start_time

    test_labels, test_preds = evaluate_model(model, dataloaders["test"])

    report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES)

    print("\nTest Report:\n", report)

    print(f"\nTraining time: {training_time:.2f} sec")


if __name__ == "__main__":
    main()