import os

BASE_PATH = "Data/CNN_Training/CNN_Both"
OUTPUT_FILE = os.path.join(BASE_PATH, "cnn_both_detailed_counts.txt")

SPLITS = ["train", "val", "test"]
CLASSES = ["Benign", "Malignant"]


def count_patients_and_files(class_path):
    patient_count = 0
    file_count = 0

    if not os.path.exists(class_path):
        return 0, 0

    for patient in os.listdir(class_path):
        patient_path = os.path.join(class_path, patient)

        if os.path.isdir(patient_path):
            patient_count += 1

            files = [
                f for f in os.listdir(patient_path)
                if os.path.isfile(os.path.join(patient_path, f))
            ]

            file_count += len(files)

    return patient_count, file_count


def main():
    with open(OUTPUT_FILE, "w") as f:
        f.write("CNN_Both Detailed Counts\n")
        f.write("========================\n\n")

        total_patients_all = 0
        total_files_all = 0

        for split in SPLITS:
            f.write(f"{split.upper()}\n")
            f.write("-" * len(split) + "\n")

            split_patients = 0
            split_files = 0

            for cls in CLASSES:
                class_path = os.path.join(BASE_PATH, split, cls)

                patients, files = count_patients_and_files(class_path)

                split_patients += patients
                split_files += files

                f.write(f"{cls}:\n")
                f.write(f"  Patients: {patients}\n")
                f.write(f"  Files: {files}\n")

            f.write(f"Total {split} Patients: {split_patients}\n")
            f.write(f"Total {split} Files: {split_files}\n\n")

            total_patients_all += split_patients
            total_files_all += split_files

        f.write("========================\n")
        f.write(f"Total Patients (All): {total_patients_all}\n")
        f.write(f"Total Files (All): {total_files_all}\n")

    print(f"✅ Detailed counts saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()