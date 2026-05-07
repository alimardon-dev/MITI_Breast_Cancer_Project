import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data" / "CNN_Training" / "CNN_Both"

CC_MODEL_PATH = PROJECT_ROOT / "RESULTS" / "CC_ResNet18_with_Augmentation" / "best_cc_resnet18_with_aug.pth"
MLO_MODEL_PATH = PROJECT_ROOT / "RESULTS" / "MLO_ResNet50_with_Augmentation" / "best_mlo_resnet50_with_aug.pth"

RESULTS_DIR = PROJECT_ROOT / "RESULTS" / "Late_Fusion_CC_MLO"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = ["Benign", "Malignant"]

# Equal fusion first. Later you can try 0.6 / 0.4.
CC_WEIGHT = 0.5
MLO_WEIGHT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASET
# =========================

class PairedCCMLODataset(Dataset):
    """
    Uses CNN_Both structure:

    CNN_Both/test/Benign/P_00032/CC.png
    CNN_Both/test/Benign/P_00032/MLO.png

    Returns:
    CC image, MLO image, label, patient_id
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_path = self.root_dir / class_name

            if not class_path.exists():
                print(f"Missing folder: {class_path}")
                continue

            for patient_folder in sorted(class_path.iterdir()):
                if not patient_folder.is_dir():
                    continue

                cc_path = patient_folder / "CC.png"
                mlo_path = patient_folder / "MLO.png"

                if cc_path.exists() and mlo_path.exists():
                    self.samples.append(
                        (cc_path, mlo_path, label_idx, patient_folder.name)
                    )

        print(f"Loaded {len(self.samples)} paired patients from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cc_path, mlo_path, label, patient_id = self.samples[idx]

        cc_img = Image.open(cc_path).convert("RGB")
        mlo_img = Image.open(mlo_path).convert("RGB")

        if self.transform:
            cc_img = self.transform(cc_img)
            mlo_img = self.transform(mlo_img)

        return cc_img, mlo_img, label, patient_id


# =========================
# TRANSFORM
# =========================

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# CC MODEL: ResNet18
# =========================

class CCResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(CCResNet18, self).__init__()

        self.backbone = models.resnet18(weights=None)

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
# MLO MODEL: ResNet50
# =========================

class MLOResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(MLOResNet50, self).__init__()

        self.backbone = models.resnet50(weights=None)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# =========================
# LOAD MODELS
# =========================

def load_models():
    cc_model = CCResNet18(num_classes=2).to(DEVICE)
    mlo_model = MLOResNet50(num_classes=2).to(DEVICE)

    cc_model.load_state_dict(torch.load(CC_MODEL_PATH, map_location=DEVICE))
    mlo_model.load_state_dict(torch.load(MLO_MODEL_PATH, map_location=DEVICE))

    cc_model.eval()
    mlo_model.eval()

    print("✅ CC model loaded:", CC_MODEL_PATH)
    print("✅ MLO model loaded:", MLO_MODEL_PATH)

    return cc_model, mlo_model


# =========================
# LATE FUSION EVALUATION
# =========================

def evaluate_late_fusion(cc_model, mlo_model, dataloader):
    all_labels = []
    all_preds = []
    all_patient_ids = []

    all_cc_probs = []
    all_mlo_probs = []
    all_fused_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for cc_imgs, mlo_imgs, labels, patient_ids in dataloader:
            cc_imgs = cc_imgs.to(DEVICE)
            mlo_imgs = mlo_imgs.to(DEVICE)

            cc_outputs = cc_model(cc_imgs)
            mlo_outputs = mlo_model(mlo_imgs)

            cc_probs = softmax(cc_outputs)
            mlo_probs = softmax(mlo_outputs)

            fused_probs = (CC_WEIGHT * cc_probs) + (MLO_WEIGHT * mlo_probs)

            preds = torch.argmax(fused_probs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_patient_ids.extend(patient_ids)

            all_cc_probs.extend(cc_probs.cpu().numpy())
            all_mlo_probs.extend(mlo_probs.cpu().numpy())
            all_fused_probs.extend(fused_probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        all_patient_ids,
        np.array(all_cc_probs),
        np.array(all_mlo_probs),
        np.array(all_fused_probs)
    )


# =========================
# SAVE RESULTS
# =========================

def save_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Late Fusion CC + MLO")
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_predictions_csv(patient_ids, labels, preds, cc_probs, mlo_probs, fused_probs):
    output_path = RESULTS_DIR / "late_fusion_predictions.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient_id",
            "true_label",
            "predicted_label",

            "cc_prob_benign",
            "cc_prob_malignant",

            "mlo_prob_benign",
            "mlo_prob_malignant",

            "fused_prob_benign",
            "fused_prob_malignant"
        ])

        for i in range(len(patient_ids)):
            writer.writerow([
                patient_ids[i],
                CLASS_NAMES[labels[i]],
                CLASS_NAMES[preds[i]],

                cc_probs[i][0],
                cc_probs[i][1],

                mlo_probs[i][0],
                mlo_probs[i][1],

                fused_probs[i][0],
                fused_probs[i][1]
            ])

    print("✅ Predictions saved:", output_path)


def save_report(labels, preds):
    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES
    )

    print("\nLate Fusion Classification Report")
    print(report)

    output_path = RESULTS_DIR / "classification_report.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Late Fusion: CC ResNet18 + MLO ResNet50\n")
        f.write("=======================================\n\n")

        f.write(f"Seed: {SEED}\n")
        f.write(f"Device: {DEVICE}\n\n")

        f.write("Models:\n")
        f.write(f"CC model: {CC_MODEL_PATH}\n")
        f.write(f"MLO model: {MLO_MODEL_PATH}\n\n")

        f.write("Fusion strategy:\n")
        f.write(f"Final probability = {CC_WEIGHT} * CC probability + {MLO_WEIGHT} * MLO probability\n\n")

        f.write("Test dataset:\n")
        f.write("CNN_Both/test\n")
        f.write("135 patients | Benign: 76 | Malignant: 59\n\n")

        f.write(report)

    print("✅ Report saved:", output_path)


# =========================
# MAIN
# =========================

def main():
    print(f"Using device: {DEVICE}")
    print(f"CC weight: {CC_WEIGHT}")
    print(f"MLO weight: {MLO_WEIGHT}")

    test_dataset = PairedCCMLODataset(
        DATA_DIR / "test",
        transform=eval_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    cc_model, mlo_model = load_models()

    labels, preds, patient_ids, cc_probs, mlo_probs, fused_probs = evaluate_late_fusion(
        cc_model,
        mlo_model,
        test_loader
    )

    save_report(labels, preds)
    save_confusion_matrix(labels, preds)
    save_predictions_csv(patient_ids, labels, preds, cc_probs, mlo_probs, fused_probs)

    print("\n✅ Late fusion evaluation finished!")
    print("📁 Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()