import os

BASE_PATH = "Data/Cropped"

DATASETS = {
    "Mass_Test": "Data_Mass_Test_Cropped_PNG",
    "Mass_Train": "Data_Mass_Train_Cropped_PNG"
}

VIEWS = ["CC", "MLO"]
CLASSES = ["Benign", "Malignant"]

OUTPUT_FILE = "mass_png_counts.txt"


def count_files(folder_path):
    if not os.path.exists(folder_path):
        return 0

    return len([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ])


def main():
    with open(OUTPUT_FILE, "w") as f:
        f.write("Mass PNG File Counts\n")
        f.write("====================\n\n")

        for dataset_name, dataset_folder in DATASETS.items():
            dataset_path = os.path.join(BASE_PATH, dataset_folder)

            f.write(f"{dataset_name}\n")
            f.write("-" * len(dataset_name) + "\n")

            total_dataset = 0

            for view in VIEWS:
                f.write(f"  {view}:\n")

                for cls in CLASSES:
                    folder = os.path.join(dataset_path, view, cls)
                    count = count_files(folder)

                    total_dataset += count
                    f.write(f"    {cls}: {count}\n")

            f.write(f"  Total {dataset_name}: {total_dataset}\n\n")

    print(f"✅ Counts saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()