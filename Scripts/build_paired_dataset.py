import os
import shutil
from collections import defaultdict

SRC_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\Data_Mass_Train_Cropped_PNG"
DST_ROOT = r"C:\Users\USER\Desktop\CNN_training_breast_cancer\paired_data"

VIEWS = ["CC", "MLO"]
LABELS = ["Benign", "Malignant"]


def extract_patient_id(filename):
    parts = filename.split("_")
    for i in range(len(parts) - 1):
        if parts[i] == "P":
            return f"P_{parts[i+1]}"
    return None


def collect_view_files(view):
    patient_files = defaultdict(list)

    for label in LABELS:
        folder = os.path.join(SRC_ROOT, view, label)
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
                "path": os.path.join(folder, fname),
                "label": label,
                "filename": fname
            })

    return patient_files


def choose_one_file(file_list):
    # if a patient has multiple files for a view, pick the first sorted by filename
    file_list = sorted(file_list, key=lambda x: x["filename"])
    return file_list[0]


def build_pairs():
    cc_files = collect_view_files("CC")
    mlo_files = collect_view_files("MLO")

    paired = []
    skipped_label_mismatch = []
    skipped_missing = []

    all_patients = sorted(set(cc_files.keys()) | set(mlo_files.keys()))

    for pid in all_patients:
        has_cc = pid in cc_files
        has_mlo = pid in mlo_files

        if not (has_cc and has_mlo):
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
            "cc_path": cc_item["path"],
            "mlo_path": mlo_item["path"],
            "cc_filename": cc_item["filename"],
            "mlo_filename": mlo_item["filename"],
        })

    return paired, skipped_missing, skipped_label_mismatch


def create_output_structure():
    if os.path.exists(DST_ROOT):
        print(f"Removing old folder: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    for label in LABELS:
        os.makedirs(os.path.join(DST_ROOT, label), exist_ok=True)


def copy_pairs(paired):
    counts = defaultdict(int)

    for item in paired:
        patient_dir = os.path.join(DST_ROOT, item["label"], item["patient_id"])
        os.makedirs(patient_dir, exist_ok=True)

        cc_dst = os.path.join(patient_dir, "CC.png")
        mlo_dst = os.path.join(patient_dir, "MLO.png")

        shutil.copy2(item["cc_path"], cc_dst)
        shutil.copy2(item["mlo_path"], mlo_dst)

        counts[item["label"]] += 1

    return counts


def main():
    print("Building paired A-type dataset...")
    print(f"Source:      {SRC_ROOT}")
    print(f"Destination: {DST_ROOT}")

    paired, skipped_missing, skipped_label_mismatch = build_pairs()

    create_output_structure()
    counts = copy_pairs(paired)

    print("\nPaired dataset summary:")
    print(f"Total paired patients (A type): {len(paired)}")
    print(f"Benign pairs:    {counts['Benign']}")
    print(f"Malignant pairs: {counts['Malignant']}")

    print(f"\nSkipped (missing one view): {len(skipped_missing)}")
    print(f"Skipped (label mismatch):   {len(skipped_label_mismatch)}")

    print("\nExample paired patients:")
    for item in paired[:10]:
        print(f"{item['patient_id']} | {item['label']} | {item['cc_filename']} | {item['mlo_filename']}")

    print("\nDone.")
    print("Paired dataset created here:")
    print(DST_ROOT)


if __name__ == "__main__":
    main()