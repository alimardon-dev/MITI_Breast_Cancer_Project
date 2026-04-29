import os
import shutil
import random
from collections import defaultdict, Counter

PROJECT_ROOT = r"C:\Users\USER\Desktop\CNN_TRAINING_2.0"

TRAIN_SRC_ROOT = os.path.join(PROJECT_ROOT, "Data_Mass_Train_Cropped_PNG")
TEST_SRC_ROOT  = os.path.join(PROJECT_ROOT, "Data_Mass_Test_Cropped_PNG")

DST_ROOT = os.path.join(PROJECT_ROOT, "cnn_both")

LABELS = ["Benign", "Malignant"]
RANDOM_SEED = 42
TRAIN_RATIO_FROM_ORIGINAL_TRAIN = 0.80


def extract_patient_id(filename):
    parts = filename.split("_")
    for i in range(len(parts) - 1):
        if parts[i] == "P":
            return f"P_{parts[i+1]}"
    return None


def collect_view_files(src_root, view):
    patient_files = defaultdict(list)

    for label in LABELS:
        folder = os.path.join(src_root, view, label)

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

            patient_files[pid].append({
                "patient_id": pid,
                "label": label,
                "filename": fname,
                "src_path": os.path.join(folder, fname)
            })

    return patient_files


def choose_one_file(files):
    return sorted(files, key=lambda x: x["filename"])[0]


def build_pairs(src_root):
    cc_files = collect_view_files(src_root, "CC")
    mlo_files = collect_view_files(src_root, "MLO")

    paired = []
    skipped_missing = []
    skipped_label_mismatch = []

    all_patients = sorted(set(cc_files.keys()) | set(mlo_files.keys()))

    for pid in all_patients:
        if pid not in cc_files or pid not in mlo_files:
            skipped_missing.append(pid)
            continue

        cc_item = choose_one_file(cc_files[pid])
        mlo_item = choose_one_file(mlo_files[pid])

        if cc_item["label"] != mlo_item["label"]:
            skipped_label_mismatch.append(pid)
            continue

        paired.append({
            "patient_id": pid,
            "label": cc_item["label"],
            "cc_path": cc_item["src_path"],
            "mlo_path": mlo_item["src_path"],
            "cc_filename": cc_item["filename"],
            "mlo_filename": mlo_item["filename"]
        })

    return paired, skipped_missing, skipped_label_mismatch


def split_train_val(pairs):
    random.seed(RANDOM_SEED)
    pairs = sorted(pairs, key=lambda x: x["patient_id"])
    random.shuffle(pairs)

    n_train = int(len(pairs) * TRAIN_RATIO_FROM_ORIGINAL_TRAIN)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    return train_pairs, val_pairs


def reset_output_folder():
    if os.path.exists(DST_ROOT):
        print(f"Removing old folder: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    for split in ["train", "val", "test"]:
        for label in LABELS:
            os.makedirs(os.path.join(DST_ROOT, split, label), exist_ok=True)


def copy_pairs(pairs, split_name):
    count = 0
    label_counter = Counter()

    for item in pairs:
        patient_dir = os.path.join(
            DST_ROOT,
            split_name,
            item["label"],
            item["patient_id"]
        )

        os.makedirs(patient_dir, exist_ok=True)

        shutil.copy2(item["cc_path"], os.path.join(patient_dir, "CC.png"))
        shutil.copy2(item["mlo_path"], os.path.join(patient_dir, "MLO.png"))

        count += 1
        label_counter[item["label"]] += 1

    return count, label_counter


def check_patient_leakage(train_pairs, val_pairs, test_pairs):
    train_ids = {x["patient_id"] for x in train_pairs}
    val_ids = {x["patient_id"] for x in val_pairs}
    test_ids = {x["patient_id"] for x in test_pairs}

    return {
        "train_val": train_ids & val_ids,
        "train_test": train_ids & test_ids,
        "val_test": val_ids & test_ids
    }


def main():
    print("Building BOTH-view CNN dataset: CC + MLO paired only")
    print(f"Train source: {TRAIN_SRC_ROOT}")
    print(f"Test source:  {TEST_SRC_ROOT}")
    print(f"Destination:  {DST_ROOT}")

    train_pairs_all, train_missing, train_mismatch = build_pairs(TRAIN_SRC_ROOT)
    test_pairs, test_missing, test_mismatch = build_pairs(TEST_SRC_ROOT)

    train_pairs, val_pairs = split_train_val(train_pairs_all)

    reset_output_folder()

    train_count, train_labels = copy_pairs(train_pairs, "train")
    val_count, val_labels = copy_pairs(val_pairs, "val")
    test_count, test_labels = copy_pairs(test_pairs, "test")

    leakage = check_patient_leakage(train_pairs, val_pairs, test_pairs)

    print("\n==============================")
    print("BOTH-VIEW DATASET CREATED")
    print("==============================")

    print("\nPaired patient counts:")
    print(f"Train pairs: {train_count}")
    print(f"Val pairs:   {val_count}")
    print(f"Test pairs:  {test_count}")

    print("\nLabel counts:")
    print(f"Train: {dict(train_labels)}")
    print(f"Val:   {dict(val_labels)}")
    print(f"Test:  {dict(test_labels)}")

    print("\nSkipped from original train:")
    print(f"Missing one view: {len(train_missing)}")
    print(f"Label mismatch:   {len(train_mismatch)}")

    print("\nSkipped from original test:")
    print(f"Missing one view: {len(test_missing)}")
    print(f"Label mismatch:   {len(test_mismatch)}")

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