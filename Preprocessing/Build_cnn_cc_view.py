import os
import shutil
import random
from collections import defaultdict, Counter

# =========================
# PATHS
# =========================
PROJECT_ROOT = r"C:\Users\USER\Desktop\CNN_TRAINING_2.0"

TRAIN_SRC_ROOT = os.path.join(PROJECT_ROOT, "Data_Mass_Train_Cropped_PNG")
TEST_SRC_ROOT  = os.path.join(PROJECT_ROOT, "Data_Mass_Test_Cropped_PNG")

DST_ROOT = os.path.join(PROJECT_ROOT, "cnn_cc_view")

VIEW = "CC"
LABELS = ["Benign", "Malignant"]

RANDOM_SEED = 42

# From original train folder:
# 87.5% stays train, 12.5% becomes validation
TRAIN_RATIO_FROM_ORIGINAL_TRAIN = 0.875


# =========================
# HELPERS
# =========================
def extract_patient_id(filename):
    """
    Example:
    Mass-Training_P_00004_LEFT_CC_1_1-2.png
    -> P_00004
    """
    parts = filename.split("_")

    for i in range(len(parts) - 1):
        if parts[i] == "P":
            return f"P_{parts[i+1]}"

    return None


def collect_files_by_patient(src_root):
    """
    Collect CC files grouped by patient ID.
    Expected structure:
    src_root/CC/Benign/*.png
    src_root/CC/Malignant/*.png
    """
    patient_to_files = defaultdict(list)

    for label in LABELS:
        folder = os.path.join(src_root, VIEW, label)

        if not os.path.exists(folder):
            print(f"Missing folder: {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".png"):
                continue

            pid = extract_patient_id(fname)

            if pid is None:
                print(f"Could not extract patient ID from: {fname}")
                continue

            src_path = os.path.join(folder, fname)

            patient_to_files[pid].append({
                "patient_id": pid,
                "label": label,
                "filename": fname,
                "src_path": src_path
            })

    return patient_to_files


def split_train_val(patient_to_files):
    patient_ids = sorted(patient_to_files.keys())

    random.seed(RANDOM_SEED)
    random.shuffle(patient_ids)

    n_train = int(len(patient_ids) * TRAIN_RATIO_FROM_ORIGINAL_TRAIN)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:]

    return train_ids, val_ids


def reset_output_folder():
    if os.path.exists(DST_ROOT):
        print(f"Removing old folder: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    for split in ["train", "val", "test"]:
        for label in LABELS:
            os.makedirs(os.path.join(DST_ROOT, split, label), exist_ok=True)


def copy_patients(patient_to_files, patient_ids, split_name):
    count = 0
    label_counter = Counter()

    for pid in patient_ids:
        for item in patient_to_files[pid]:
            label = item["label"]
            src_path = item["src_path"]
            fname = item["filename"]

            dst_path = os.path.join(DST_ROOT, split_name, label, fname)
            shutil.copy2(src_path, dst_path)

            count += 1
            label_counter[label] += 1

    return count, label_counter


def copy_all_test_files(test_patient_to_files):
    count = 0
    label_counter = Counter()

    for pid, files in test_patient_to_files.items():
        for item in files:
            label = item["label"]
            src_path = item["src_path"]
            fname = item["filename"]

            dst_path = os.path.join(DST_ROOT, "test", label, fname)
            shutil.copy2(src_path, dst_path)

            count += 1
            label_counter[label] += 1

    return count, label_counter


def check_patient_leakage(train_ids, val_ids, test_ids):
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    leakage = {
        "train_val": train_set & val_set,
        "train_test": train_set & test_set,
        "val_test": val_set & test_set
    }

    return leakage


# =========================
# MAIN
# =========================
def main():
    print("Building CC CNN dataset...")
    print(f"Train source: {TRAIN_SRC_ROOT}")
    print(f"Test source:  {TEST_SRC_ROOT}")
    print(f"Destination:  {DST_ROOT}")

    train_patient_to_files = collect_files_by_patient(TRAIN_SRC_ROOT)
    test_patient_to_files = collect_files_by_patient(TEST_SRC_ROOT)

    print(f"\nOriginal train patients found: {len(train_patient_to_files)}")
    print(f"Original test patients found:  {len(test_patient_to_files)}")

    if len(train_patient_to_files) == 0:
        print("ERROR: No train patients found.")
        return

    if len(test_patient_to_files) == 0:
        print("ERROR: No test patients found.")
        return

    train_ids, val_ids = split_train_val(train_patient_to_files)
    test_ids = sorted(test_patient_to_files.keys())

    reset_output_folder()

    train_count, train_labels = copy_patients(train_patient_to_files, train_ids, "train")
    val_count, val_labels = copy_patients(train_patient_to_files, val_ids, "val")
    test_count, test_labels = copy_all_test_files(test_patient_to_files)

    leakage = check_patient_leakage(train_ids, val_ids, test_ids)

    print("\n==============================")
    print("CC DATASET CREATED")
    print("==============================")

    print("\nPatient counts:")
    print(f"Train patients: {len(train_ids)}")
    print(f"Val patients:   {len(val_ids)}")
    print(f"Test patients:  {len(test_ids)}")

    print("\nImage counts:")
    print(f"Train images: {train_count}")
    print(f"Val images:   {val_count}")
    print(f"Test images:  {test_count}")

    print("\nLabel counts:")
    print(f"Train: {dict(train_labels)}")
    print(f"Val:   {dict(val_labels)}")
    print(f"Test:  {dict(test_labels)}")

    print("\nLeakage check:")
    if any(len(v) > 0 for v in leakage.values()):
        print("WARNING: Patient leakage found!")
        for k, v in leakage.items():
            if len(v) > 0:
                print(f"{k}: {sorted(list(v))[:10]}")
    else:
        print("No patient leakage found.")

    print("\nFinal folder created here:")
    print(DST_ROOT)


if __name__ == "__main__":
    main()