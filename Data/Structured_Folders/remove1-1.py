import os

base_dir = r"project/Data/"

folders = [
    "Benign",
    "Benign_without_callback",
    "Malignant"
]

total_deleted = 0

for folder in folders:
    folder_path = os.path.join(base_dir, folder)

    print(f"\nProcessing folder: {folder}")

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".dcm"):
            continue

        # target only 1-1 files
        if "1-1" in file:
            file_path = os.path.join(folder_path, file)

            try:
                os.remove(file_path)
                print(f"Deleted: {file}")
                total_deleted += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")

print(f"\nTotal deleted files: {total_deleted}")