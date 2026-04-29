import os
import re
import random
import shutil

# =========================
# CONFIG
# =========================

SOURCE_ROOT = "Data/Cropped/Data_Mass_Train_Cropped_PNG"
DEST_ROOT = "Data/CNN_Training/CNN_Both"

TRAIN_RATIO = 0.80
SEED = 42

CLASSES = ["Benign", "Malignant"]
VIEWS = ["CC", "MLO"]


def extract_patient_id(filename):
    match = re.search(r"P_\d+", filename)
    return match.group() if match else None


def collect_patient_pairs(class_name):
    """
    Collect patients that have BOTH CC and MLO images.
    """
    patients = {}

    for view in VIEWS:
        folder = os.path.join(SOURCE_ROOT, view, class_name)

        if not os.path.exists(folder):
            print(f"⚠️ Missing folder: {folder}")
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            patient_id = extract_patient_id(file)
            if patient_id is None:
                continue

            if patient_id not in patients:
                patients[patient_id] = {}

            patients[patient_id][view] = os.path.join(folder, file)

    complete_patients = {
        pid: paths
        for pid, paths in patients.items()
        if "CC" in paths and "MLO" in paths
    }

    missing_pairs = {
        pid: paths
        for pid, paths in patients.items()
        if not ("CC" in paths and "MLO" in paths)
    }

    return complete_patients, missing_pairs


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def copy_patient(patient_id, paths, split, class_name):
    patient_folder = os.path.join(DEST_ROOT, split, class_name, patient_id)
    create_folder(patient_folder)

    shutil.copy2(paths["CC"], os.path.join(patient_folder, "CC.png"))
    shutil.copy2(paths["MLO"], os.path.join(patient_folder, "MLO.png"))


def main():
    random.seed(SEED)

    # Remove old CNN_Both folder to avoid duplicated/mixed data
    if os.path.exists(DEST_ROOT):
        print(f"🧹 Removing old folder: {DEST_ROOT}")
        shutil.rmtree(DEST_ROOT)

    report_lines = []
    report_lines.append("CNN_Both Train/Validation Split Report")
    report_lines.append("=====================================")
    report_lines.append(f"Train ratio: {TRAIN_RATIO}")
    report_lines.append(f"Validation ratio: {1 - TRAIN_RATIO}")
    report_lines.append("")

    for split in ["train", "val"]:
        for class_name in CLASSES:
            create_folder(os.path.join(DEST_ROOT, split, class_name))

    for class_name in CLASSES:
        complete_patients, missing_pairs = collect_patient_pairs(class_name)

        patient_ids = list(complete_patients.keys())
        random.shuffle(patient_ids)

        train_count = int(len(patient_ids) * TRAIN_RATIO)

        train_ids = patient_ids[:train_count]
        val_ids = patient_ids[train_count:]

        for pid in train_ids:
            copy_patient(pid, complete_patients[pid], "train", class_name)

        for pid in val_ids:
            copy_patient(pid, complete_patients[pid], "val", class_name)

        report_lines.append(f"{class_name}")
        report_lines.append("-" * len(class_name))
        report_lines.append(f"Total complete patients: {len(patient_ids)}")
        report_lines.append(f"Train patients: {len(train_ids)}")
        report_lines.append(f"Validation patients: {len(val_ids)}")
        report_lines.append(f"Missing CC/MLO pairs skipped: {len(missing_pairs)}")
        report_lines.append("")

    report_path = os.path.join(DEST_ROOT, "split_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("✅ CNN_Both train/validation split created successfully!")
    print(f"📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()