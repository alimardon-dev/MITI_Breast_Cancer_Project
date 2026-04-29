import os
import shutil
import random
from collections import defaultdict

SRC_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG"
DST_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\cnn_data_mlo"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VIEW = "MLO"
LABELS = ["Benign", "Malignant"]


def extract_patient_id(filename):
    parts = filename.split("_")
    for i in range(len(parts) - 1):
        if parts[i] == "P":
            return f"P_{parts[i+1]}"
    return None


def collect_files_by_patient():
    patient_to_files = defaultdict(list)

    for label in LABELS:
        folder = os.path.join(SRC_ROOT, VIEW, label)
        if not os.path.exists(folder):
            print(f"Missing source folder: {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".png"):
                continue

            pid = extract_patient_id(fname)
            if pid is None:
                continue

            full_path = os.path.join(folder, fname)
            patient_to_files[pid].append((full_path, label, fname))

    return patient_to_files


def make_split_lists(patient_ids):
    random.seed(RANDOM_SEED)
    patient_ids = sorted(patient_ids)
    random.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def create_folders():
    if os.path.exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)

    for split in ["train", "val", "test"]:
        for label in LABELS:
            os.makedirs(os.path.join(DST_ROOT, split, VIEW, label), exist_ok=True)


def copy_split(patient_to_files, split_name, patient_ids):
    count = 0
    for pid in patient_ids:
        for src_path, label, fname in patient_to_files[pid]:
            dst_path = os.path.join(DST_ROOT, split_name, VIEW, label, fname)
            shutil.copy2(src_path, dst_path)
            count += 1
    return count


def main():
    print("Building patient-wise MLO split (NO resizing)...")

    patient_to_files = collect_files_by_patient()
    patient_ids = list(patient_to_files.keys())

    if len(patient_ids) == 0:
        print("ERROR: No patients found.")
        return

    train_ids, val_ids, test_ids = make_split_lists(patient_ids)

    create_folders()

    train_count = copy_split(patient_to_files, "train", train_ids)
    val_count = copy_split(patient_to_files, "val", val_ids)
    test_count = copy_split(patient_to_files, "test", test_ids)

    print("\nDone.")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    print("Created here:")
    print(DST_ROOT)


if __name__ == "__main__":
    main()