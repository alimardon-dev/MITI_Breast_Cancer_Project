import os

BASE_PATH = "Data/CNN_Training/CNN_Both"
OUTPUT_FILE = "cnn_both_patient_counts.txt"

SPLITS = ["train", "val", "test"]
CLASSES = ["Benign", "Malignant"]


def count_patient_folders(class_path):
    if not os.path.exists(class_path):
        return 0, []

    patient_folders = [
        name for name in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, name))
    ]

    return len(patient_folders), sorted(patient_folders)


def main():
    all_patients = set()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("CNN_Both Patient Counts\n")
        f.write("=======================\n\n")

        for split in SPLITS:
            f.write(f"{split.upper()}\n")
            f.write("-" * len(split) + "\n")

            split_total = 0

            for cls in CLASSES:
                class_path = os.path.join(BASE_PATH, split, cls)
                count, patients = count_patient_folders(class_path)

                split_total += count
                all_patients.update(patients)

                f.write(f"{cls}: {count}\n")

            f.write(f"Total {split}: {split_total}\n\n")

        f.write("=======================\n")
        f.write(f"Total unique patients in CNN_Both: {len(all_patients)}\n")

    print(f"✅ Patient counts saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()