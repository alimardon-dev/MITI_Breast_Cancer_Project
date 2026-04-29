import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =========================
# SEED
# =========================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================
# PATHS
# =========================
DATA_DIR = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\cnn_data_mlo"

TRAIN_DIR = os.path.join(DATA_DIR, "train", "MLO")
VAL_DIR   = os.path.join(DATA_DIR, "val", "MLO")
TEST_DIR  = os.path.join(DATA_DIR, "test", "MLO")

OUTPUT_DIR = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\results_mlo_with_augmentation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# SETTINGS
# =========================
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# =========================
# DATA
# =========================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class_names = train_dataset.classes

print("Device:", DEVICE)
print("Classes:", class_names)
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# freeze all layers first
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)

model = model.to(DEVICE)


# =========================
# LOSS / OPTIMIZER
# =========================
# class order should be ['Benign', 'Malignant']
class_weights = torch.tensor([1.0, 1.5]).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================
# FUNCTIONS
# =========================
def train_one_epoch():
    model.train()

    correct = 0
    total = 0
    loss_sum = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        out = model(x)
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
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
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


# =========================
# TRAINING
# =========================
history = {
    "train_acc": [],
    "val_acc": [],
    "train_loss": [],
    "val_loss": []
}

best_val_acc = 0
best_model_path = os.path.join(OUTPUT_DIR, "best_mlo_model.pth")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc, _, _, _ = evaluate(val_loader)

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)


print("\nBest Val Acc:", best_val_acc)


# =========================
# TEST
# =========================
model.load_state_dict(torch.load(best_model_path, weights_only=True))

test_loss, test_acc, y_true, y_pred, y_prob_malignant = evaluate(test_loader)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nTest Loss:", test_loss)
print("Test Acc:", test_acc)
print("\nClassification Report:\n")
print(report)


# =========================
# SAVE RESULTS
# =========================
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

with open(os.path.join(OUTPUT_DIR, "final_metrics.txt"), "w") as f:
    f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write("Class weights: Benign=1.0, Malignant=2.0\n")
    f.write("Trainable layers: layer3, layer4, fc\n")

with open(os.path.join(OUTPUT_DIR, "test_predictions.csv"), "w") as f:
    f.write("true_label,predicted_label,malignant_probability\n")
    for t, p, prob in zip(y_true, y_pred, y_prob_malignant):
        f.write(f"{class_names[t]},{class_names[p]},{prob:.6f}\n")

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - MLO CNN")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve - MLO CNN")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve - MLO CNN")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved results to:")
print(OUTPUT_DIR)