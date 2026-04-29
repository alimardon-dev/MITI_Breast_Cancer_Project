import os
import shutil
import random
from collections import defaultdict
from PIL import Image
import pydicom
import hashlib
import sys

# Fix encoding (Windows safe)
sys.stdout.reconfigure(encoding='utf-8')

# ===== PATHS (EDIT IF NEEDED) =====
INPUT_TRAIN = "Data_Mass_Train_Cropped"
INPUT_TEST = "Data_Mass_Test_Cropped"
PNG_TRAIN = "Data_Mass_Train_Cropped_PNG"
PNG_TEST = "Data_Mass_Test_Cropped_PNG"
CNN_ROOT = "cnn_data"

IMG_SIZE = (224, 224)
SPLIT_RATIO = (0.7, 0.15, 0.15)


# ----------------------------
# STEP 1: DICOM → PNG
# ----------------------------
def dicom_to_png(input_root, output_root):
    print(f"Converting {input_root} -> {output_root}")

    for root, _, files in os.walk(input_root):
        for f in files:
            if f.endswith(".dcm"):
                dcm_path = os.path.join(root, f)
                rel = os.path.relpath(root, input_root)
                out_dir = os.path.join(output_root, rel)
                os.makedirs(out_dir, exist_ok=True)

                try:
                    dcm = pydicom.dcmread(dcm_path)
                    img = dcm.pixel_array

                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = (img * 255).astype("uint8")

                    img = Image.fromarray(img).resize(IMG_SIZE)

                    out_path = os.path.join(out_dir, f.replace(".dcm", ".png"))
                    img.save(out_path)

                except Exception as e:
                    print(f"ERROR converting: {dcm_path}")


# ----------------------------
# STEP 2: HELPERS
# ----------------------------
def extract_patient_id(name):
    for part in name.split("_"):
        if part.startswith("P"):
            return part
    return None


def collect_images(root):
    data = defaultdict(list)

    for path, _, files in os.walk(root):
        for f in files:
            if f.endswith(".png") and "_CC_" in f:
                pid = extract_patient_id(f)

                if "benign" in f.lower():
                    label = "Benign"
                else:
                    label = "Malignant"

                full = os.path.join(path, f)

                if pid:
                    data[pid].append((full, label))

    return data


# ----------------------------
# STEP 3: SPLIT PATIENT-WISE
# ----------------------------
def split_patients(patients):
    keys = list(patients.keys())
    random.shuffle(keys)

    n = len(keys)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    train = keys[:n_train]
    val = keys[n_train:n_train + n_val]
    test = keys[n_train + n_val:]

    return train, val, test


# ----------------------------
# STEP 4: BUILD CNN DATASET
# ----------------------------
def build_dataset(patients, splits):
    print("Building cnn_data folder...")

    if os.path.exists(CNN_ROOT):
        shutil.rmtree(CNN_ROOT)

    for split_name, pids in zip(["train", "val", "test"], splits):
        for pid in pids:
            for img_path, label in patients[pid]:
                dest = os.path.join(CNN_ROOT, split_name, "CC", label)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(img_path, dest)


# ----------------------------
# STEP 5: VALIDATION
# ----------------------------
def validate():
    print("\nStarting validation...")

    patients = defaultdict(set)
    sizes = set()
    hashes = set()
    errors = []
    counts = defaultdict(int)

    for split in ["train", "val", "test"]:
        for label in ["Benign", "Malignant"]:
            folder = os.path.join(CNN_ROOT, split, "CC", label)

            if not os.path.exists(folder):
                errors.append(f"Missing folder: {folder}")
                continue

            for f in os.listdir(folder):
                path = os.path.join(folder, f)

                # PNG check
                if not f.endswith(".png"):
                    errors.append(f"Not PNG: {f}")

                # patient leakage
                pid = extract_patient_id(f)
                if pid:
                    patients[pid].add(split)

                # size check
                try:
                    img = Image.open(path)
                    sizes.add(img.size)

                    if img.size != IMG_SIZE:
                        errors.append(f"Wrong size: {f} -> {img.size}")

                except:
                    errors.append(f"Corrupted image: {f}")

                # duplicate content
                try:
                    h = hashlib.md5(open(path, 'rb').read()).hexdigest()
                    if h in hashes:
                        errors.append(f"Duplicate image: {f}")
                    hashes.add(h)
                except:
                    errors.append(f"Hash error: {f}")

                counts[(split, label)] += 1

    # leakage check
    for pid, s in patients.items():
        if len(s) > 1:
            errors.append(f"DATA LEAKAGE: {pid} in {s}")

    print("\nImage sizes found:", sizes)

    print("\nCounts:")
    for k, v in counts.items():
        print(f"{k}: {v}")

    print("\nErrors:")
    if not errors:
        print("Dataset clean and ready")
    else:
        for e in errors:
            print(e)


# ----------------------------
# RUN ALL
# ----------------------------
if __name__ == "__main__":
    dicom_to_png(INPUT_TRAIN, PNG_TRAIN)
    dicom_to_png(INPUT_TEST, PNG_TEST)

    patients = collect_images(PNG_TRAIN)
    splits = split_patients(patients)

    build_dataset(patients, splits)

    validate()