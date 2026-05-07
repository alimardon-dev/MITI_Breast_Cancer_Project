import os
import copy
import time
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
# CONFIG
# =========================

DATA_DIR = "Data/CNN_Training/CNN_Both"
RESULTS_DIR = "RESULTS/Dual_without_Augmentation"

BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 7

IMAGE_SIZE = 224

CLASS_NAMES = ["Benign", "Malignant"]

# From your CNN_Both train folder:
# Benign = 178, Malignant = 200
CLASS_WEIGHTS = [1.06, 0.94]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASET
# =========================

class DualMammogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.exists(class_path):
                continue

            for patient_id in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_id)

                if not os.path.isdir(patient_path):
                    continue

                cc_path = os.path.join(patient_path, "CC.png")
                mlo_path = os.path.join(patient_path, "MLO.png")

                if os.path.exists(cc_path) and os.path.exists(mlo_path):
                    self.samples.append((cc_path, mlo_path, label_idx))

        print(f"Loaded {len(self.samples)} patients from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label = self.samples[idx]

        cc_img = Image.open(cc_path).convert("RGB")
        mlo_img = Image.open(mlo_path).convert("RGB")

        if self.transform:
            cc_img = self.transform(cc_img)
            mlo_img = self.transform(mlo_img)

        return cc_img, mlo_img, label


# =========================
# TRANSFORMS
# =========================

# NO AUGMENTATION here
data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# MODEL
# =========================

class DualResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(DualResNet18, self).__init__()

        self.cc_branch = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.mlo_branch = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        cc_features = self.cc_branch.fc.in_features
        mlo_features = self.mlo_branch.fc.in_features

        self.cc_branch.fc = nn.Identity()
        self.mlo_branch.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(cc_features + mlo_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, cc_img, mlo_img):
        cc_features = self.cc_branch(cc_img)
        mlo_features = self.mlo_branch(mlo_img)

        fused_features = torch.cat((cc_features, mlo_features), dim=1)

        output = self.classifier(fused_features)
        return output


# =========================
# TRAIN FUNCTION
# =========================

def train_model(model, dataloaders, criterion, optimizer):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for cc_imgs, mlo_imgs, labels in dataloaders[phase]:
                cc_imgs = cc_imgs.to(DEVICE)
                mlo_imgs = mlo_imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(cc_imgs, mlo_imgs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
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
                        os.path.join(RESULTS_DIR, "best_dual_resnet18_no_aug.pth")
                    )

                    print("✅ Best model saved")
                else:
                    patience_counter += 1

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

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for cc_imgs, mlo_imgs, labels in dataloader:
            cc_imgs = cc_imgs.to(DEVICE)
            mlo_imgs = mlo_imgs.to(DEVICE)

            outputs = model(cc_imgs, mlo_imgs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_training_curves(history):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"))
    plt.close()


def save_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Dual ResNet18 No Augmentation")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()


# =========================
# MAIN
# =========================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    datasets = {
        "train": DualMammogramDataset(
            os.path.join(DATA_DIR, "train"),
            transform=data_transform
        ),
        "val": DualMammogramDataset(
            os.path.join(DATA_DIR, "val"),
            transform=data_transform
        ),
        "test": DualMammogramDataset(
            os.path.join(DATA_DIR, "test"),
            transform=data_transform
        )
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=True if split == "train" else False,
            num_workers=0
        )
        for split in ["train", "val", "test"]
    }

    model = DualResNet18(num_classes=2).to(DEVICE)

    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    model, history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer
    )

    training_time = time.time() - start_time

    plot_training_curves(history)

    test_labels, test_preds = evaluate_model(model, dataloaders["test"])

    report = classification_report(
        test_labels,
        test_preds,
        target_names=CLASS_NAMES
    )

    print("\nTest Classification Report")
    print(report)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write("Dual ResNet18 Without Augmentation\n")
        f.write("==================================\n\n")
        f.write(f"Training time: {training_time:.2f} seconds\n\n")
        f.write(report)

    save_confusion_matrix(test_labels, test_preds)

    print("\n✅ Training finished!")
    print(f"📁 Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()