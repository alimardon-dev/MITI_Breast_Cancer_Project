import os
import shutil
import random

SRC_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\paired_data"
DST_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\paired_data_split"

LABELS = ["Benign", "Malignant"]

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def collect_patients():
    patients = []

    for label in LABELS:
        label_folder = os.path.join(SRC_ROOT, label)

        if not os.path.exists(label_folder):
            print(f"Missing folder: {label_folder}")
            continue

        for pid in os.listdir(label_folder):
            patient_path = os.path.join(label_folder, pid)

            if not os.path.isdir(patient_path):
                continue

            # check CC and MLO exist
            cc_path = os.path.join(patient_path, "CC.png")
            mlo_path = os.path.join(patient_path, "MLO.png")

            if not (os.path.exists(cc_path) and os.path.exists(mlo_path)):
                print(f"Skipping incomplete pair: {pid}")
                continue

            patients.append({
                "patient_id": pid,
                "label": label,
                "path": patient_path
            })

    return patients


def split_patients(patients):
    random.seed(RANDOM_SEED)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = patients[:n_train]
    val = patients[n_train:n_train + n_val]
    test = patients[n_train + n_val:]

    return train, val, test


def create_structure():
    if os.path.exists(DST_ROOT):
        print(f"Removing old split: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    for split in ["train", "val", "test"]:
        for label in LABELS:
            os.makedirs(os.path.join(DST_ROOT, split, label), exist_ok=True)


def copy_split(split_name, patients):
    count = 0

    for item in patients:
        src = item["path"]
        dst = os.path.join(DST_ROOT, split_name, item["label"], item["patient_id"])

        shutil.copytree(src, dst)
        count += 1

    return count


def main():
    print("Splitting paired dataset (patient-wise)...")

    patients = collect_patients()
    print(f"\nTotal paired patients: {len(patients)}")

    train, val, test = split_patients(patients)

    print("\nSplit sizes:")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")

    create_structure()

    train_count = copy_split("train", train)
    val_count = copy_split("val", val)
    test_count = copy_split("test", test)

    print("\nCopied:")
    print(f"Train: {train_count}")
    print(f"Val:   {val_count}")
    print(f"Test:  {test_count}")

    print("\nDone.")
    print("New dataset here:")
    print(DST_ROOT)


if __name__ == "__main__":
    main()