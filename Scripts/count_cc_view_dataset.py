import os

BASE_PATH = "Data/CNN_Training/CNN_CC_View"
OUTPUT_FILE = os.path.join(BASE_PATH, "cc_view_counts.txt")

SPLITS = ["train", "val", "test"]
CLASSES = ["Benign", "Malignant"]


def count_files(folder_path):
    if not os.path.exists(folder_path):
        return 0

    return len([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ])


def main():
    total_all = 0

    with open(OUTPUT_FILE, "w") as f:
        f.write("CNN_CC_View File Counts\n")
        f.write("========================\n\n")

        for split in SPLITS:
            f.write(f"{split.upper()}\n")
            f.write("-" * len(split) + "\n")

            split_total = 0

            for cls in CLASSES:
                class_path = os.path.join(BASE_PATH, split, cls)

                count = count_files(class_path)
                split_total += count

                f.write(f"{cls}: {count}\n")

            f.write(f"Total {split}: {split_total}\n\n")

            total_all += split_total

        f.write("========================\n")
        f.write(f"Total Files (All): {total_all}\n")

    print(f"✅ Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()