import csv
import random
import numpy as np
from pathlib import Path
from PIL import Image

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
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data" / "CNN_Training" / "CNN_Both"

CC_MODEL_PATH = PROJECT_ROOT / "RESULTS" / "CC_ResNet18_with_Augmentation" / "best_cc_resnet18_with_aug.pth"
MLO_MODEL_PATH = PROJECT_ROOT / "RESULTS" / "MLO_ResNet50_with_Augmentation" / "best_mlo_resnet50_with_aug.pth"

OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "CNN_Probs_For_Hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = ["Benign", "Malignant"]
SPLITS = ["train", "val", "test"]

CC_WEIGHT = 0.5
MLO_WEIGHT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASET
# =========================

class PairedCCMLODataset(Dataset):
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


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# MODELS
# =========================

class CCResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

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


class MLOResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

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


def load_models():
    cc_model = CCResNet18().to(DEVICE)
    mlo_model = MLOResNet50().to(DEVICE)

    cc_model.load_state_dict(torch.load(CC_MODEL_PATH, map_location=DEVICE))
    mlo_model.load_state_dict(torch.load(MLO_MODEL_PATH, map_location=DEVICE))

    cc_model.eval()
    mlo_model.eval()

    print("✅ CC model loaded")
    print("✅ MLO model loaded")

    return cc_model, mlo_model


# =========================
# GENERATE PROBS
# =========================

def generate_probs_for_split(split, cc_model, mlo_model):
    dataset = PairedCCMLODataset(DATA_DIR / split, transform=eval_transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    softmax = nn.Softmax(dim=1)

    rows = []

    with torch.no_grad():
        for cc_imgs, mlo_imgs, labels, patient_ids in loader:
            cc_imgs = cc_imgs.to(DEVICE)
            mlo_imgs = mlo_imgs.to(DEVICE)

            cc_outputs = cc_model(cc_imgs)
            mlo_outputs = mlo_model(mlo_imgs)

            cc_probs = softmax(cc_outputs).cpu().numpy()
            mlo_probs = softmax(mlo_outputs).cpu().numpy()

            fused_probs = (CC_WEIGHT * cc_probs) + (MLO_WEIGHT * mlo_probs)

            for i, patient_id in enumerate(patient_ids):
                rows.append({
                    "patient_id": patient_id,
                    "label": int(labels[i]),

                    "cc_prob_benign": float(cc_probs[i][0]),
                    "cc_prob_malignant": float(cc_probs[i][1]),

                    "mlo_prob_benign": float(mlo_probs[i][0]),
                    "mlo_prob_malignant": float(mlo_probs[i][1]),

                    "cnn_prob_benign": float(fused_probs[i][0]),
                    "cnn_prob_malignant": float(fused_probs[i][1]),
                })

    output_path = OUTPUT_DIR / f"{split}_cnn_probs.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "label",
                "cc_prob_benign",
                "cc_prob_malignant",
                "mlo_prob_benign",
                "mlo_prob_malignant",
                "cnn_prob_benign",
                "cnn_prob_malignant",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {split}: {output_path}")
    print(f"   Patients: {len(rows)}")


def main():
    print(f"Using device: {DEVICE}")
    print(f"Fusion weight: CC={CC_WEIGHT}, MLO={MLO_WEIGHT}")

    cc_model, mlo_model = load_models()

    for split in SPLITS:
        generate_probs_for_split(split, cc_model, mlo_model)

    print("\n✅ CNN probability generation finished!")
    print(f"📁 Saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()