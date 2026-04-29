import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DATA_DIR = r"C:\Users\USER\Desktop\CNN_TRAINING_2.0\cnn_both"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = r"C:\Users\USER\Desktop\CNN_TRAINING_2.0\RESULTS\Dual_without_Augmentation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_TO_IDX = {"Benign": 0, "Malignant": 1}
CLASS_NAMES = ["Benign", "Malignant"]


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


class PairedDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform

        for label in CLASS_NAMES:
            label_dir = os.path.join(root_dir, label)

            if not os.path.exists(label_dir):
                print(f"Missing folder: {label_dir}")
                continue

            for pid in os.listdir(label_dir):
                patient_dir = os.path.join(label_dir, pid)

                if not os.path.isdir(patient_dir):
                    continue

                cc_path = os.path.join(patient_dir, "CC.png")
                mlo_path = os.path.join(patient_dir, "MLO.png")

                if os.path.exists(cc_path) and os.path.exists(mlo_path):
                    self.samples.append((cc_path, mlo_path, CLASS_TO_IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label = self.samples[idx]

        cc = Image.open(cc_path).convert("L")
        mlo = Image.open(mlo_path).convert("L")

        cc = self.transform(cc)
        mlo = self.transform(mlo)

        return cc, mlo, label


train_dataset = PairedDataset(TRAIN_DIR, transform)
val_dataset   = PairedDataset(VAL_DIR, transform)
test_dataset  = PairedDataset(TEST_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("Device:", DEVICE)
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


class DualModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cc = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.mlo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in self.cc.parameters():
            param.requires_grad = False

        for param in self.mlo.parameters():
            param.requires_grad = False

        for param in self.cc.layer4.parameters():
            param.requires_grad = True

        for param in self.mlo.layer4.parameters():
            param.requires_grad = True

        feat_dim = self.cc.fc.in_features

        self.cc.fc = nn.Identity()
        self.mlo.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, cc, mlo):
        cc_features = self.cc(cc)
        mlo_features = self.mlo(mlo)

        combined = torch.cat((cc_features, mlo_features), dim=1)

        return self.classifier(combined)


model = DualModel().to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


def train_one_epoch():
    model.train()

    correct = 0
    total = 0
    loss_sum = 0

    for cc, mlo, y in train_loader:
        cc, mlo, y = cc.to(DEVICE), mlo.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        out = model(cc, mlo)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        preds = out.argmax(1)

        loss_sum += loss.item() * y.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total


def evaluate(loader):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0

    y_true = []
    y_pred = []
    y_prob_malignant = []

    with torch.no_grad():
        for cc, mlo, y in loader:
            cc, mlo, y = cc.to(DEVICE), mlo.to(DEVICE), y.to(DEVICE)

            out = model(cc, mlo)
            loss = criterion(out, y)

            probs = torch.softmax(out, dim=1)
            preds = out.argmax(1)

            loss_sum += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_malignant.extend(probs[:, 1].cpu().numpy())

    return loss_sum / total, correct / total, y_true, y_pred, y_prob_malignant


best_val_acc = 0
best_wts = copy.deepcopy(model.state_dict())

history = {
    "train_acc": [],
    "val_acc": [],
    "train_loss": [],
    "val_loss": []
}

best_model_path = os.path.join(OUTPUT_DIR, "best_dual_no_aug_model.pth")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc, _, _, _ = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_wts = copy.deepcopy(model.state_dict())
        torch.save(best_wts, best_model_path)
        print("Best model updated.")


print("\nBest Val Acc:", best_val_acc)


model.load_state_dict(best_wts)

test_loss, test_acc, y_true, y_pred, y_prob_malignant = evaluate(test_loader)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print("\nTest Loss:", test_loss)
print("Test Acc:", test_acc)
print("\nClassification Report:\n")
print(report)


with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

with open(os.path.join(OUTPUT_DIR, "final_metrics.txt"), "w") as f:
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write("Model: Dual-input ResNet18 CC + MLO\n")
    f.write("Augmentation: NO\n")
    f.write("Trainable layers: layer4 + classifier\n")

with open(os.path.join(OUTPUT_DIR, "test_predictions.csv"), "w") as f:
    f.write("true_label,predicted_label,malignant_probability\n")
    for t, p, prob in zip(y_true, y_pred, y_prob_malignant):
        f.write(f"{CLASS_NAMES[t]},{CLASS_NAMES[p]},{prob:.6f}\n")

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Dual Input CNN No Augmentation")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve - Dual Input CNN No Augmentation")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve - Dual Input CNN No Augmentation")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved results to:")
print(OUTPUT_DIR)