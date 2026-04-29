import os
import shutil

# =========================================================
# PATHS
# =========================================================

project_root = r"C:\Users\USER\Desktop\CBIS-DDSM Dataset\manifest-ZkhPvrLo5216730872708713142\project"
archive_root = os.path.join(project_root, "_ARCHIVE_OLD")

# Folders we want to archive safely
folders_to_archive = [
    "OLD_Data_Mass_Train",
    "OLD_Data_Mass_Test",
    "train_bwc",
    "test_bwc",
]

# Final folders we want to keep in project root
folders_to_keep = [
    "Data_Mass_Train_Full",
    "Data_Mass_Train_Cropped",
    "Data_Mass_Train_ROI",
    "Data_Mass_Test_Full",
    "Data_Mass_Test_Cropped",
    "Data_Mass_Test_ROI",
    "data_mass_train_bwc_full",
    "data_mass_train_bwc_cropped",
    "data_mass_train_bwc_roi",
    "data_mass_test_bwc_full",
    "data_mass_test_bwc_roi",
]


# =========================================================
# HELPERS
# =========================================================

def get_safe_destination(base_dir, folder_name):
    """
    If folder_name already exists in base_dir,
    return folder_name_1, folder_name_2, etc.
    """
    candidate = os.path.join(base_dir, folder_name)
    if not os.path.exists(candidate):
        return candidate

    i = 1
    while True:
        new_candidate = os.path.join(base_dir, f"{folder_name}_{i}")
        if not os.path.exists(new_candidate):
            return new_candidate
        i += 1


def move_folder_safely(src_path, dst_base_dir):
    """
    Move entire folder into dst_base_dir without overwriting.
    """
    folder_name = os.path.basename(src_path)
    dst_path = get_safe_destination(dst_base_dir, folder_name)

    print(f"[MOVE] {src_path}")
    print(f"   -> {dst_path}")
    shutil.move(src_path, dst_path)


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 70)
    print("SAFE PROJECT CLEANUP")
    print("=" * 70)

    if not os.path.exists(project_root):
        print(f"[ERROR] Project root not found: {project_root}")
        return

    os.makedirs(archive_root, exist_ok=True)
    print(f"[INFO] Archive folder ready: {archive_root}")
    print()

    # 1. Show final folders that are expected to remain
    print("[KEEP THESE FINAL FOLDERS]")
    for name in folders_to_keep:
        path = os.path.join(project_root, name)
        print(f" - {name} | exists: {os.path.exists(path)}")
    print()

    # 2. Move old folders into archive
    moved_count = 0
    skipped_count = 0

    print("[ARCHIVING OLD FOLDERS]")
    for name in folders_to_archive:
        src_path = os.path.join(project_root, name)

        if not os.path.exists(src_path):
            print(f"[SKIP] Not found: {src_path}")
            skipped_count += 1
            continue

        move_folder_safely(src_path, archive_root)
        moved_count += 1

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Moved folders : {moved_count}")
    print(f"Skipped       : {skipped_count}")
    print()
    print("Old folders are now inside:")
    print(archive_root)
    print()
    print("Your final dataset folders stay in project root unchanged.")


if __name__ == "__main__":
    main()