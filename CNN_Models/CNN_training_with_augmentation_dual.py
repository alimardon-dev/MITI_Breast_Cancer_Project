"""
CNN_training_dual_improved.py
Dual-input ResNet18 for breast cancer classification (CBIS-DDSM)
Uses both CC and MLO mammogram views per patient.
Includes: dropout, weight decay, early stopping, ReduceLROnPlateau,
          layer freezing, safe augmentation, and full test evaluation.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------------------------------------------
# 0. REPRODUCIBILITY
# ---------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------
DATA_ROOT   = r"C:\Users\USER\Desktop\CNN_TRAINING_2.0\cnn_both"
OUTPUT_DIR  = r"C:\Users\USER\Desktop\CNN_TRAINING_2.0\RESULTS\Dual_with_Augmentation_Improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = 256
BATCH_SIZE  = 32
MAX_EPOCHS  = 30
LR          = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT     = 0.6
PATIENCE    = 3          # early stopping patience (based on val loss)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Class mapping: Benign=0, Malignant=1  (folder names must match exactly)
CLASS_TO_IDX = {"Benign": 0, "Malignant": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ---------------------------------------------
# 2. TRANSFORMS
# Safe augmentation for training; no augmentation for val/test.
# ---------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ---------------------------------------------
# 3. DATASET
# Walks split/Class/PatientFolder and loads CC.png + MLO.png
# ---------------------------------------------
class PairedDataset(Dataset):
    """
    Custom Dataset for paired CC + MLO mammogram images.
    Directory layout:
        root/
          Benign/   P_xxxxx/{CC.png, MLO.png}
          Malignant/P_xxxxx/{CC.png, MLO.png}
    Returns: (cc_tensor, mlo_tensor, label)
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []  # list of (cc_path, mlo_path, label_idx)

        root_dir = Path(root_dir)
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found, skipping.")
                continue
            for patient_dir in sorted(class_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                cc_path  = patient_dir / "CC.png"
                mlo_path = patient_dir / "MLO.png"
                if cc_path.exists() and mlo_path.exists():
                    self.samples.append((cc_path, mlo_path, label))
                else:
                    print(f"Warning: Missing CC or MLO in {patient_dir}, skipping.")

        print(f"  Loaded {len(self.samples)} patients from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label = self.samples[idx]

        cc_img  = Image.open(cc_path).convert("RGB")
        mlo_img = Image.open(mlo_path).convert("RGB")

        if self.transform:
            # Apply the same random seed so both views get identical geometric aug
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            cc_tensor = self.transform(cc_img)

            random.seed(seed)
            torch.manual_seed(seed)
            mlo_tensor = self.transform(mlo_img)
        else:
            cc_tensor  = eval_transform(cc_img)
            mlo_tensor = eval_transform(mlo_img)

        return cc_tensor, mlo_tensor, label


# ---------------------------------------------
# 4. DUAL-INPUT MODEL
# Two ResNet18 branches (shared architecture, separate weights).
# Features from both are concatenated then classified.
# ---------------------------------------------
class DualResNet18(nn.Module):
    """
    Dual-branch ResNet18:
      - branch_cc  processes the CC view
      - branch_mlo processes the MLO view
      - Features are concatenated -> dropout -> FC -> 2 classes
    Only layer4 and the custom head are trainable.
    """
    def __init__(self, dropout=0.6, num_classes=2):
        super().__init__()

        # ---- CC branch ----
        cc_base  = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # ---- MLO branch ----
        mlo_base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze everything first
        for param in cc_base.parameters():
            param.requires_grad = False
        for param in mlo_base.parameters():
            param.requires_grad = False

        # Unfreeze layer4 in both branches
        for param in cc_base.layer4.parameters():
            param.requires_grad = True
        for param in mlo_base.layer4.parameters():
            param.requires_grad = True

        # Remove the original FC classifiers; keep feature extractor
        feature_dim = cc_base.fc.in_features  # 512

        self.branch_cc  = nn.Sequential(*list(cc_base.children())[:-1])   # -> (B, 512, 1, 1)
        self.branch_mlo = nn.Sequential(*list(mlo_base.children())[:-1])  # -> (B, 512, 1, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, cc, mlo):
        f_cc  = self.branch_cc(cc)   # (B, 512, 1, 1)
        f_mlo = self.branch_mlo(mlo) # (B, 512, 1, 1)
        combined = torch.cat([f_cc, f_mlo], dim=1)  # (B, 1024, 1, 1)
        return self.classifier(combined)


# ---------------------------------------------
# 5. TRAINING / EVALUATION HELPERS
# ---------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    """One pass over a DataLoader. If optimizer is None -> eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for cc, mlo, labels in loader:
            cc, mlo, labels = cc.to(device), mlo.to(device), labels.to(device)

            logits = model(cc, mlo)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def collect_predictions(model, loader, device=DEVICE):
    """Return arrays of true labels, predicted labels, and malignant probabilities."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for cc, mlo, labels in loader:
            cc, mlo = cc.to(device), mlo.to(device)
            logits   = model(cc, mlo)
            probs    = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds    = logits.argmax(dim=1).cpu().numpy()
            all_true.extend(labels.numpy())
            all_pred.extend(preds)
            all_prob.extend(probs)

    return np.array(all_true), np.array(all_pred), np.array(all_prob)


# ---------------------------------------------
# 6. MAIN TRAINING LOOP
# ---------------------------------------------
def main():
    # --- Datasets & Loaders ---
    print("\nLoading datasets...")
    train_ds = PairedDataset(os.path.join(DATA_ROOT, "train"), transform=train_transform)
    val_ds   = PairedDataset(os.path.join(DATA_ROOT, "val"),   transform=None)
    test_ds  = PairedDataset(os.path.join(DATA_ROOT, "test"),  transform=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # --- Model ---
    print("\nBuilding model...")
    model = DualResNet18(dropout=DROPOUT, num_classes=2).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}")

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # --- Training loop with early stopping ---
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(OUTPUT_DIR, "best_dual_model.pth")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (early stop patience={PATIENCE})...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion)

        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        # Save best model (based on val loss)
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (val_loss={vl_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

    # -----------------------------------------
    # 7. PLOTS
    # -----------------------------------------
    epochs_ran = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs_ran, history["train_loss"], label="Train Loss")
    plt.plot(epochs_ran, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curve - Dual Input CNN (Improved)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs_ran, history["train_acc"], label="Train Acc")
    plt.plot(epochs_ran, history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Accuracy Curve - Dual Input CNN (Improved)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()

    # -----------------------------------------
    # 8. TEST EVALUATION
    # -----------------------------------------
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    true_labels, pred_labels, mal_probs = collect_predictions(model, test_loader)

    class_names = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]

    print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")
    report_str = classification_report(
        true_labels, pred_labels, target_names=class_names, digits=4
    )
    print("\nClassification Report:\n", report_str)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.title("Confusion Matrix - Dual Input CNN (Improved)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # -----------------------------------------
    # 9. SAVE TEXT OUTPUTS
    # -----------------------------------------
    # Classification report
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_str)

    # Final metrics
    best_val_acc = max(history["val_acc"])
    with open(os.path.join(OUTPUT_DIR, "final_metrics.txt"), "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Model: Dual-input ResNet18 CC + MLO (Improved)\n")
        f.write(f"Augmentation: YES\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"Trainable layers: layer4 + classifier\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
        f.write(f"Early Stopping Patience: {PATIENCE}\n")

    # Predictions CSV
    pred_df = pd.DataFrame({
        "true_label":           [IDX_TO_CLASS[i] for i in true_labels],
        "predicted_label":      [IDX_TO_CLASS[i] for i in pred_labels],
        "malignant_probability": mal_probs,
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()