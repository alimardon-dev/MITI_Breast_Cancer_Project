import os

BASE_PATH = "Data/CNN_Training"   # adjust if needed
OUTPUT_FILE = "dataset_counts.txt"


def count_files(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        # count only files (ignore hidden/system files if needed)
        count += len([f for f in files if not f.startswith('.')])
    return count


def process_view(view_path, view_name, file):
    file.write(f"\n{view_name}\n")

    for split in ["train", "val", "test"]:
        split_path = os.path.join(view_path, split)

        if not os.path.exists(split_path):
            continue

        file.write(f"  {split}:\n")

        for cls in ["Benign", "Malignant"]:
            cls_path = os.path.join(split_path, cls)

            if os.path.exists(cls_path):
                count = count_files(cls_path)
                file.write(f"    {cls}: {count}\n")
            else:
                file.write(f"    {cls}: (missing)\n")


def main():
    with open(OUTPUT_FILE, "w") as f:
        f.write("Dataset File Counts\n")
        f.write("====================\n")

        for view in os.listdir(BASE_PATH):
            view_path = os.path.join(BASE_PATH, view)

            if os.path.isdir(view_path):
                process_view(view_path, view, f)

    print(f"✅ Counts saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()